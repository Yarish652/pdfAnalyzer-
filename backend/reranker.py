"""Local cross-encoder reranking for retrieved document chunks."""

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
model = CrossEncoder(MODEL_NAME)


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Score and sort retrieved chunks without changing their source fields."""
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)

    reranked = []
    for chunk, score in zip(chunks, scores):
        reranked.append({
            **chunk,
            "metadata": dict(chunk.get("metadata", {})),
            "reranker_score": float(score),
        })

    return sorted(
        reranked,
        key=lambda chunk: chunk["reranker_score"],
        reverse=True,
    )