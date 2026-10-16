from contextlib import asynccontextmanager
from contextvars import ContextVar
import json
import logging
from uuid import uuid4

from chromadb.errors import ChromaError
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAIError
from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from config import MAX_PDF_PAGES, MAX_UPLOAD_SIZE_BYTES, OCR_ENABLED

from pdf_reader import extract_document
import ocr
from chunker import chunk_document
from document_classifier import classify_document
from embedding import embed_documents, embed_texts
from hybrid_retriever import reciprocal_rank_fusion
from keyword_retriever import retrieve_keyword_chunks
from reranker import rerank_chunks
from vector_store import add_documents, get_chunks, search
from context_builder import expand_with_neighbors, format_context
from generator import REFUSAL, generate_answer, is_daily_quota_error, rewrite_query, stream_answer


logger = logging.getLogger(__name__)
correlation_id_context: ContextVar[str | None] = ContextVar(
    "correlation_id",
    default=None,
)


class AskRequest(BaseModel):
    question: str
    document_id: str
    history: list[dict] = []


def warm_up_models() -> None:
    """Run each local model once so the first question does not pay setup cost."""
    embed_texts(["warm up"])
    rerank_chunks("warm up", [{"text": "warm up", "metadata": {}}])


@asynccontextmanager
async def lifespan(_app):
    await run_in_threadpool(warm_up_models)
    yield


app = FastAPI(lifespan=lifespan)

# This in-memory state is suitable for a single-process development server.
# Move it to Redis or a database before running multiple application workers.
upload_jobs: dict[str, str] = {}
# Details shown to the user: document type and size once ready, or why it failed.
upload_details: dict[str, dict] = {}

NO_TEXT_REASON = (
    "This PDF has no selectable text. It is probably a scanned image, "
    "and text recognition (OCR) is turned off or not installed."
)
NO_TEXT_AFTER_OCR_REASON = (
    "We couldn't find any readable text in this PDF, even with text "
    "recognition. The scan may be too blurry or contain only images."
)
GENERIC_FAILURE_REASON = "We couldn't process this PDF. Please try another file."

# Candidates passed to the cross-encoder. Its cost grows linearly with this,
# and on the benchmarks the top 6 fused results kept the same accuracy as ~16.
RERANK_CANDIDATES = 6
CONTEXT_CHUNKS = 3


class NonPdfUploadError(Exception):
    pass


class InvalidUploadError(Exception):
    pass


class UploadTooLargeError(Exception):
    pass


class PageLimitExceededError(Exception):
    pass


def retrieve_context_chunks(query: str, document_id: str) -> list[dict]:
    """Retrieve, fuse, and rerank candidates, then widen them into passages."""
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

    reranked = rerank_chunks(query, fused_chunks[:RERANK_CANDIDATES])
    return expand_with_neighbors(reranked[:CONTEXT_CHUNKS], get_chunks)


def source_payload(chunks: list[dict]) -> list[dict]:
    """Sources numbered to match the [n] citations in the answer."""
    sources = []
    for number, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        sources.append({
            "id": number,
            "page": metadata.get("page"),
            "section": metadata.get("section") or metadata.get("subsection") or "",
            "text": chunk["text"],
        })
    return sources


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


def ocr_quality(document) -> str | None:
    """"low" when recognized text is unreliable (for example handwriting)."""
    confidences = [page["ocr_confidence"] for page in document if page.get("ocr_confidence") is not None]
    if not confidences:
        return None
    return "low" if sum(confidences) / len(confidences) < ocr.LOW_CONFIDENCE else "good"


# Share of the progress bar for reading pages. OCR dominates ingestion time
# when it runs; reading a text layer is nearly instant.
TEXT_READ_SHARE = 10
OCR_READ_SHARE = 60
EMBEDDING_DONE = 95


def report_progress(document_id: str, stage: str, percent: float, **extra) -> None:
    upload_details[document_id] = {"stage": stage, "progress": int(percent), **extra}


async def process_upload(document_id: str, file, correlation_id: str) -> None:
    """Run the synchronous PDF ingestion pipeline outside the event loop.

    Progress is published through upload_details as a stage ("reading",
    "recognizing", "embedding", "saving") and an overall percentage.
    """
    upload_jobs[document_id] = "processing"
    report_progress(document_id, "reading", 0)

    def report_ocr_progress(done: int, total: int) -> None:
        report_progress(
            document_id,
            "recognizing",
            OCR_READ_SHARE * done / total,
            ocr_pages_done=done,
            ocr_pages_total=total,
        )

    try:
        document = await run_in_threadpool(
            extract_document,
            file,
            ocr=OCR_ENABLED,
            on_progress=report_ocr_progress,
        )
    except PdfReadError:
        logger.exception("Upload extraction failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
        return
    except Exception:
        logger.exception("Upload extraction failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
        return

    if not document:
        # Image-only (scanned) PDFs have no text layer; there is nothing to index.
        logger.warning("Upload has no extractable text request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
        ocr_was_used = OCR_ENABLED and ocr.is_available()
        upload_details[document_id] = {
            "reason": NO_TEXT_AFTER_OCR_REASON if ocr_was_used else NO_TEXT_REASON,
        }
        return

    try:
        classification = classify_document(document)
        logger.info(
            "Classified upload as %s (%s) request_id=%s",
            classification.document_type,
            "; ".join(classification.reasons),
            correlation_id,
        )
        chunks = await run_in_threadpool(
            chunk_document,
            document,
            classification.document_type,
        )
        for chunk in chunks:
            chunk["metadata"]["document_id"] = document_id

        texts = [chunk["text"] for chunk in chunks]
    except Exception:
        logger.exception("Upload chunking failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
        return

    if not chunks:
        logger.warning("Upload produced no chunks request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
        return

    read_share = OCR_READ_SHARE if any(page.get("ocr") for page in document) else TEXT_READ_SHARE
    report_progress(document_id, "embedding", read_share)

    def report_embedding_progress(done: int, total: int) -> None:
        report_progress(
            document_id,
            "embedding",
            read_share + (EMBEDDING_DONE - read_share) * done / total,
        )

    try:
        embeddings = await run_in_threadpool(
            embed_documents,
            texts,
            on_progress=report_embedding_progress,
        )
    except Exception:
        logger.exception("Upload embedding failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
        return

    report_progress(document_id, "saving", EMBEDDING_DONE)
    try:
        await run_in_threadpool(add_documents, chunks, embeddings)
    except ChromaError:
        logger.exception("Upload storage failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
    except Exception:
        logger.exception("Upload storage failed request_id=%s", correlation_id)
        upload_jobs[document_id] = "failed"
    else:
        upload_details[document_id] = {
            "document_type": classification.document_type,
            "pages": max(page["page"] for page in document),
            "chunks": len(chunks),
            "ocr_pages": sum(1 for page in document if page.get("ocr")),
            "ocr_quality": ocr_quality(document),
        }
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

    details = upload_details.get(document_id, {})
    if status == "failed":
        # Progress from the step that failed is not useful to the client.
        details = {"reason": details.get("reason", GENERIC_FAILURE_REASON)}

    return {"document_id": document_id, "status": status, **details}


DAILY_QUOTA_DETAIL = (
    "The free model's daily request limit has been reached. "
    "It resets at midnight UTC."
)


def provider_error_detail(error: Exception, default: str) -> str:
    """A client-safe message; only the daily quota is worth explaining."""
    return DAILY_QUOTA_DETAIL if is_daily_quota_error(error) else default


async def prepare_answer_context(request: AskRequest, correlation_id: str):
    """Rewrite the question if needed and retrieve its passages.

    Returns (rewritten_query, passages), or an error response to send as is.
    """
    try:
        rewritten_query = await run_in_threadpool(
            rewrite_query,
            request.question,
            request.history,
        )
    except OpenAIError as error:
        logger.exception("Query rewrite failed request_id=%s", correlation_id)
        return error_response(
            502,
            provider_error_detail(error, "Unable to contact the answer service."),
            correlation_id,
        )
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

    return rewritten_query, context_chunks


@app.post("/ask")
async def ask_question(request: AskRequest):
    correlation_id = current_correlation_id()

    prepared = await prepare_answer_context(request, correlation_id)
    if isinstance(prepared, JSONResponse):
        return prepared
    rewritten_query, context_chunks = prepared

    try:
        answer = await run_in_threadpool(
            generate_answer,
            request.question,
            format_context(context_chunks),
            request.history,
        )
    except OpenAIError as error:
        logger.exception("Answer generation failed request_id=%s", correlation_id)
        return error_response(
            502,
            provider_error_detail(error, "Unable to generate an answer at this time."),
            correlation_id,
        )
    except Exception:
        logger.exception("Answer generation failed request_id=%s", correlation_id)
        return error_response(500, "An unexpected internal error occurred.", correlation_id)

    return {
        "question": request.question,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "sources": source_payload(context_chunks),
    }


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/ask/stream")
async def ask_question_stream(request: AskRequest):
    """Stream an answer as server-sent events.

    Events: "sources" (sent before generation starts), then "token" deltas,
    then "done" with the full answer, or "error" if generation fails.
    Failures before generation return a normal JSON error response.
    """
    correlation_id = current_correlation_id()

    prepared = await prepare_answer_context(request, correlation_id)
    if isinstance(prepared, JSONResponse):
        return prepared
    rewritten_query, context_chunks = prepared
    context = format_context(context_chunks)

    def events():
        yield sse_event("sources", {
            "rewritten_query": rewritten_query,
            "sources": source_payload(context_chunks),
        })

        parts = []
        try:
            for delta in stream_answer(request.question, context, request.history):
                parts.append(delta)
                yield sse_event("token", {"text": delta})
        except OpenAIError as error:
            logger.exception("Answer streaming failed request_id=%s", correlation_id)
            yield sse_event("error", {
                "detail": provider_error_detail(error, "Unable to generate an answer at this time."),
                "request_id": correlation_id,
            })
            return
        except Exception:
            logger.exception("Answer streaming failed request_id=%s", correlation_id)
            yield sse_event("error", {
                "detail": "An unexpected internal error occurred.",
                "request_id": correlation_id,
            })
            return

        answer = "".join(parts).strip() or REFUSAL
        yield sse_event("done", {"answer": answer})

    # The generator is synchronous; Starlette iterates it in a worker thread,
    # so a slow model never blocks the event loop.
    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
