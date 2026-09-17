"""Benchmark Chroma retrieval against local cross-encoder reranking."""

import os

from backend.embedding import embed_texts
from backend.generator import rewrite_query
from backend.reranker import rerank_chunks
from backend.vector_store import collection


TOP_K = 10

TEST_CASES = [
    {
        "question": "What technologies were used to build Speakzy?",
        "history": [],
    },
    {
        "question": "How does Speakzy perform vocabulary revision?",
        "history": [],
    },
    {
        "question": "What role did vector embeddings play in Speakzy?",
        "history": [],
    },
    {
        "question": "What technologies were used in the pronunciation assessment system?",
        "history": [],
    },
    {
        "question": "How does the pronunciation assessment system identify mistakes in speech?",
        "history": [],
    },
    {
        "question": "How did the pronunciation assessment system reduce hallucinations?",
        "history": [],
    },
    {
        "question": "How was the speech analysis pipeline made reliable?",
        "history": [],
    },
    {
        "question": "What backend technologies have you worked with?",
        "history": [],
    },
    {
        "question": "Which projects demonstrate your experience with retrieval augmented generation?",
        "history": [],
    },
    {
        "question": "What AI projects have you built and what was your role in each?",
        "history": [],
    },
]


def get_document_id() -> str:
    document_id = os.getenv("RERANKING_BENCHMARK_DOCUMENT_ID")
    if not document_id:
        raise SystemExit(
            "Set RERANKING_BENCHMARK_DOCUMENT_ID to the uploaded document ID."
        )
    return document_id


def retrieve_candidates(query: str, document_id: str) -> list[dict]:
    query_embedding = embed_texts([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K,
        where={"document_id": document_id},
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return [
        {"text": text, "metadata": metadata or {}}
        for text, metadata in zip(documents, metadatas)
    ]


def print_ranking(
    title: str,
    chunks: list[dict],
    score_key: str | None = None,
) -> None:
    print(title)

    for rank, chunk in enumerate(chunks, start=1):
        location = chunk["metadata"].get("page", "?")

        score = (
            f" | reranker_score={chunk[score_key]:.6f}"
            if score_key
            else ""
        )

        print(
            f"  {rank:>2}. page={location}{score} | {chunk['text']}"
        )


def main() -> None:
    document_id = get_document_id()

    print(f"DOCUMENT ID: {document_id}")
    print(f"RETRIEVAL CANDIDATES: top {TOP_K}")
    print(f"BENCHMARK QUESTIONS: {len(TEST_CASES)}")

    for case_number, case in enumerate(TEST_CASES, start=1):
        question = case["question"]

        rewritten_query = rewrite_query(
            question,
            case["history"],
        )

        retrieved_chunks = retrieve_candidates(
            rewritten_query,
            document_id,
        )

        reranked_chunks = rerank_chunks(
            rewritten_query,
            retrieved_chunks,
        )

        print("\n" + "=" * 100)
        print(f"TEST CASE {case_number}")
        print(f"ORIGINAL QUESTION: {question}")
        print(f"REWRITTEN QUERY: {rewritten_query}")

        print_ranking(
            "RETRIEVAL RANKING:",
            retrieved_chunks,
        )

        print_ranking(
            "RERANKED RANKING:",
            reranked_chunks,
            "reranker_score",
        )


if __name__ == "__main__":
    main()