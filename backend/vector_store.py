from pathlib import Path

import chromadb


CHROMA_DB_PATH = Path(__file__).resolve().parent / "chroma_db"
client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

collection = client.get_or_create_collection(
    name="pdf_documents"
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
