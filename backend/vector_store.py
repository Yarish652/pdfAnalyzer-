from pathlib import Path

import hashlib
import re

import chromadb

try:
    from backend.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODEL
except ModuleNotFoundError:
    from config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODEL


CHROMA_DB_PATH = Path(__file__).resolve().parent / "chroma_db"
client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


def collection_name(model_name: str) -> str:
    """One collection per embedding model: vectors from different models are
    not comparable, and their dimensions may differ."""
    if model_name == DEFAULT_EMBEDDING_MODEL:
        return "pdf_documents"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_")
    if len(slug) > 40:
        # Truncated names could collide; a hash of the full name keeps them apart.
        digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
        slug = slug[:31].rstrip("_") + "_" + digest
    return "pdf_documents__" + slug


collection = client.get_or_create_collection(
    name=collection_name(EMBEDDING_MODEL)
)


def add_documents(chunks, embeddings):
    """
    Store document chunks, embeddings, and metadata in Chroma.

    Each chunk has the normalized format:

    {
        "text": "...",
        "metadata": {
            "page": 1,
            "section": "...",
            "subsection": "...",
            "chunk_index": 0,
            "document_id": "..."
        }
    }
    """

    collection.add(
        ids=[
            f"{chunk['metadata']['document_id']}:{chunk['metadata']['chunk_index']}"
            for chunk in chunks
        ],
        documents=[
            chunk["text"]
            for chunk in chunks
        ],
        embeddings=embeddings.tolist(),
        metadatas=[
            chunk["metadata"]
            for chunk in chunks
        ],
    )


def search(query_embedding, document_id, top_k=3):
    """
    Search one document in Chroma using the query embedding.

    Chroma returns both the retrieved documents and their
    associated metadata.
    """

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        where={"document_id": document_id},
    )

    return results


def get_chunks(document_id, chunk_indexes):
    """Fetch stored chunks of one document by chunk index."""
    results = collection.get(
        ids=[f"{document_id}:{index}" for index in chunk_indexes],
        include=["documents", "metadatas"],
    )

    return [
        {"text": text, "metadata": metadata or {}}
        for text, metadata in zip(results.get("documents") or [], results.get("metadatas") or [])
    ]


def print_document_diagnostics():
    """Print document IDs, chunk counts, and one sample chunk per document."""
    results = collection.get(include=["documents", "metadatas"])
    documents_by_id = {}

    for text, metadata in zip(results.get("documents", []), results.get("metadatas", [])):
        document_id = (metadata or {}).get("document_id", "(missing document_id)")
        documents_by_id.setdefault(document_id, []).append(text or "")

    for document_id in sorted(documents_by_id):
        chunks = documents_by_id[document_id]
        print(f"Document ID: {document_id}")
        print(f"Chunk count: {len(chunks)}")
        print(f"Sample chunk: {chunks[0][:150]}")
