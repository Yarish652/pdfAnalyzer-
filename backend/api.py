from contextvars import ContextVar
import logging
from uuid import uuid4

from chromadb.errors import ChromaError
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from openai import OpenAIError
from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from config import MAX_PDF_PAGES, MAX_UPLOAD_SIZE_BYTES

from pdf_reader import extract_document
from chunker import chunk_document
from embedding import embed_texts
from hybrid_retriever import reciprocal_rank_fusion
from keyword_retriever import retrieve_keyword_chunks
from reranker import rerank_chunks
from vector_store import add_documents, search
from generator import generate_answer, rewrite_query


logger = logging.getLogger(__name__)
correlation_id_context: ContextVar[str | None] = ContextVar(
    "correlation_id",
    default=None,
)


class AskRequest(BaseModel):
    question: str
    document_id: str
    history: list[dict] = []


app = FastAPI()

# This in-memory state is suitable for a single-process development server.
# Move it to Redis or a database before running multiple application workers.
upload_jobs: dict[str, str] = {}


class NonPdfUploadError(Exception):
    pass


class InvalidUploadError(Exception):
    pass


class UploadTooLargeError(Exception):
    pass


class PageLimitExceededError(Exception):
    pass


def retrieve_context_chunks(query: str, document_id: str) -> list[dict]:
    """Retrieve, fuse, and rerank candidates for one question."""
    query_embedding = embed_texts([query])
    vector_results = search(query_embedding, document_id, top_k=10)

    documents = (vector_results.get("documents") or [[]])[0] or []
    metadatas = (vector_results.get("metadatas") or [[]])[0] or []
    vector_chunks = []

    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else None
        vector_chunks.append({
            "text": text,
            "metadata": metadata or {
                "document_id": document_id,
                "chunk_index": index,
            },
        })

    keyword_chunks = retrieve_keyword_chunks(query, document_id, top_k=10)
    fused_chunks = reciprocal_rank_fusion([vector_chunks, keyword_chunks])

    if not fused_chunks:
        return []

    return rerank_chunks(query, fused_chunks)[:3]


def current_correlation_id() -> str:
    """Return the request correlation ID, including for direct function calls."""
    return correlation_id_context.get() or str(uuid4())


def error_response(status_code: int, detail: str, correlation_id: str) -> JSONResponse:
    """Create a client-safe error response that can be correlated with logs."""
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "request_id": correlation_id},
    )


def validate_pdf(file, content_type: str | None = None) -> None:
    """Validate upload metadata and PDF bounds before scheduling processing."""
    if content_type != "application/pdf":
        raise NonPdfUploadError

    file.seek(0, 2)
    if file.tell() > MAX_UPLOAD_SIZE_BYTES:
        raise UploadTooLargeError

    file.seek(0)
    if file.read(5) != b"%PDF-":
        raise InvalidUploadError("The uploaded file is not a PDF.")

    file.seek(0)
    reader = PdfReader(file)
    if len(reader.pages) > MAX_PDF_PAGES:
        raise PageLimitExceededError

    file.seek(0)


@app.middleware("http")
async def add_correlation_id(request, call_next):
    correlation_id = str(uuid4())
    token = correlation_id_context.set(correlation_id)

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled API error request_id=%s", correlation_id)
        response = error_response(
            500,
            "An unexpected internal error occurred.",
            correlation_id,
        )
    finally:
        correlation_id_context.reset(token)

    response.headers["X-Request-ID"] = correlation_id
    return response


async def process_upload(document_id: str, file, correlation_id: str) -> None:
    """Run the synchronous PDF ingestion pipeline outside the event loop."""
    upload_jobs[document_id] = "processing"

    try:
        document = await run_in_threadpool(extract_document, file)
    except PdfReadError:
        logger.exception("Upload extraction failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
        return
    except Exception:
        logger.exception("Upload extraction failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
        return

    try:
        chunks = await run_in_threadpool(chunk_document, document)
        for chunk in chunks:
            chunk["metadata"]["document_id"] = document_id

        texts = [chunk["text"] for chunk in chunks]
    except Exception:
        logger.exception("Upload chunking failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
        return

    try:
        embeddings = await run_in_threadpool(embed_texts, texts)
    except Exception:
        logger.exception("Upload embedding failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
        return

    try:
        await run_in_threadpool(add_documents, chunks, embeddings)
    except ChromaError:
        logger.exception("Upload storage failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
    except Exception:
        logger.exception("Upload storage failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
    else:
        upload_jobs[document_id] = "ready"


@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    correlation_id = current_correlation_id()

    try:
        await run_in_threadpool(validate_pdf, file.file, file.content_type)
    except NonPdfUploadError:
        logger.warning("Rejected non-PDF upload request_id=%s", correlation_id)
        return error_response(400, "Only PDF files are accepted.", correlation_id)
    except InvalidUploadError:
        logger.warning("Rejected invalid PDF upload request_id=%s", correlation_id)
        return error_response(400, "The uploaded file is not a readable PDF.", correlation_id)
    except UploadTooLargeError:
        logger.warning("Rejected oversized upload request_id=%s", correlation_id)
        return error_response(413, "The uploaded file is too large.", correlation_id)
    except PageLimitExceededError:
        logger.warning("Rejected PDF page limit request_id=%s", correlation_id)
        return error_response(400, "The uploaded PDF has too many pages.", correlation_id)
    except PdfReadError:
        logger.exception("Malformed PDF upload request_id=%s", correlation_id)
        return error_response(400, "The uploaded file is not a readable PDF.", correlation_id)
    except Exception:
        logger.exception("PDF preflight failed request_id=%s", correlation_id)
        return error_response(500, "An unexpected internal error occurred.", correlation_id)

    document_id = str(uuid4())
    upload_jobs[document_id] = "pending"
    background_tasks.add_task(process_upload, document_id, file.file, correlation_id)

    return JSONResponse(status_code=202, content={
        "success": True,
        "message": "PDF processing started",
        "filename": file.filename,
        "document_id": document_id,
        "status": "pending",
    })


@app.get("/upload/{document_id}/status")
async def upload_status(document_id: str):
    status = upload_jobs.get(document_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Upload not found.")

    return {"document_id": document_id, "status": status}


@app.post("/ask")
async def ask_question(request: AskRequest):
    correlation_id = current_correlation_id()

    try:
        rewritten_query = await run_in_threadpool(
            rewrite_query,
            request.question,
            request.history,
        )
    except OpenAIError:
        logger.exception("Query rewrite failed request_id=%s", correlation_id)
        return error_response(502, "Unable to contact the answer service.", correlation_id)
    except Exception:
        logger.exception("Query rewrite failed request_id=%s", correlation_id)
        return error_response(500, "An unexpected internal error occurred.", correlation_id)

    try:
        context_chunks = await run_in_threadpool(
            retrieve_context_chunks,
            rewritten_query,
            request.document_id,
        )
    except ChromaError:
        logger.exception("Retrieval failed request_id=%s", correlation_id)
        return error_response(502, "Unable to retrieve document context.", correlation_id)
    except Exception:
        logger.exception("Retrieval failed request_id=%s", correlation_id)
        return error_response(500, "An unexpected internal error occurred.", correlation_id)

    context = "\n\n".join(chunk["text"] for chunk in context_chunks)

    try:
        answer = await run_in_threadpool(
            generate_answer,
            request.question,
            context,
            request.history,
        )
    except OpenAIError:
        logger.exception("Answer generation failed request_id=%s", correlation_id)
        return error_response(502, "Unable to generate an answer at this time.", correlation_id)
    except Exception:
        logger.exception("Answer generation failed request_id=%s", correlation_id)
        return error_response(500, "An unexpected internal error occurred.", correlation_id)

    return {
        "question": request.question,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "sources": [chunk["text"] for chunk in context_chunks],
    }
