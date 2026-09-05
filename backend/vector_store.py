import chromadb


client = chromadb.PersistentClient(path="./chroma_db")

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
            "chunk_index": 0
        }
    }
    """

    collection.add(
        ids=[
            str(chunk["metadata"]["chunk_index"])
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


def search(query_embedding, top_k=3):
    """
    Search Chroma using the query embedding.

    Chroma returns both the retrieved documents and their
    associated metadata.
    """

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )

    return results
