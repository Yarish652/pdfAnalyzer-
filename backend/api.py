from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File

from pdf_reader import extract_document
from chunker import chunk_document
from embedding import embed_texts
from vector_store import add_documents, search
from generator import generate_answer, rewrite_query

class AskRequest(BaseModel):
    question: str
    history: list[dict] = []

app = FastAPI()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # -----------------------------------------------------
    # 1. Extract the document
    # -----------------------------------------------------

    document = extract_document(file.file)

    # -----------------------------------------------------
    # 2. Chunk the document
    # -----------------------------------------------------

    chunks = chunk_document(document)

    # -----------------------------------------------------
    # 3. Extract text for embedding
    # -----------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # -----------------------------------------------------
    # 4. Generate embeddings
    # -----------------------------------------------------

    embeddings = embed_texts(texts)

    # -----------------------------------------------------
    # 5. Store chunks, embeddings, and metadata
    #    in ChromaDB
    # -----------------------------------------------------

    add_documents(
        chunks,
        embeddings
    )

    return {
        "success": True,
        "message": "PDF processed successfully",
        "filename": file.filename,
        "chunks": len(chunks),
    }



@app.post("/ask")
async def ask_question(request: AskRequest):
    rewritten_query = rewrite_query(
        request.question,
        request.history
    )
    query_embedding = embed_texts([rewritten_query])

    results = search(query_embedding, top_k=3)

    context = "\n\n".join(results["documents"][0])

    try:
        answer = generate_answer(
            request.question,
            context,
            request.history
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="Unable to generate an answer at this time."
        ) from error

    return {
        "question": request.question,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "sources": results["documents"][0]
    }
