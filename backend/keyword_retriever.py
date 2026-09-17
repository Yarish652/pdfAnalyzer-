"""BM25 keyword retrieval over chunks stored in Chroma."""

import math
import re
from collections import Counter

try:
    from backend.vector_store import collection
except ModuleNotFoundError:
    from vector_store import collection


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
K1 = 1.5
B = 0.75
_index_cache: dict[str, tuple[int, list[dict]]] = {}


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _get_document_index(document_id: str) -> list[dict]:
    collection_size = collection.count()
    cached_index = _index_cache.get(document_id)
    if cached_index and cached_index[0] == collection_size:
        return cached_index[1]

    results = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    tokenized_chunks = [
        {
            "text": document or "",
            "metadata": dict(metadatas[index] or {})
            if index < len(metadatas)
            else {},
            "tokens": tokenize(document or ""),
        }
        for index, document in enumerate(documents)
    ]
    _index_cache[document_id] = (collection_size, tokenized_chunks)
    return tokenized_chunks


def retrieve_keyword_chunks(
    query: str,
    document_id: str,
    top_k: int = 3,
) -> list[dict]:
    """Return document chunks ranked by descending BM25 score."""
    if top_k <= 0:
        return []

    indexed_chunks = _get_document_index(document_id)
    if not indexed_chunks:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return []

    document_count = len(indexed_chunks)
    average_length = sum(
        len(chunk["tokens"])
        for chunk in indexed_chunks
    ) / document_count
    document_frequencies = Counter(
        term
        for chunk in indexed_chunks
        for term in set(chunk["tokens"])
    )

    scored_chunks = []
    for chunk in indexed_chunks:
        tokens = chunk["tokens"]
        term_frequencies = Counter(tokens)
        document_length = len(tokens)
        score = 0.0

        for term in query_terms:
            document_frequency = document_frequencies.get(term, 0)
            if not document_frequency:
                continue

            inverse_document_frequency = math.log(
                1
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            term_frequency = term_frequencies[term]
            length_normalization = (
                1 - B + B * document_length / average_length
                if average_length
                else 1
            )
            score += inverse_document_frequency * (
                term_frequency * (K1 + 1)
                / (term_frequency + K1 * length_normalization)
            )

        scored_chunks.append({
            "text": chunk["text"],
            "metadata": dict(chunk["metadata"]),
            "bm25_score": score,
        })

    scored_chunks.sort(key=lambda chunk: chunk["bm25_score"], reverse=True)
    return scored_chunks[:top_k]
