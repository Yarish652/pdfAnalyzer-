"""Benchmark BM25 keyword retrieval over the resume chunks."""

import os

from backend.keyword_retriever import retrieve_keyword_chunks


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


def print_ranking(chunks: list[dict]) -> None:
    for rank, chunk in enumerate(chunks, start=1):
        location = chunk["metadata"].get("page", "?")
        print(
            f"  {rank:>2}. page={location} "
            f"| bm25_score={chunk['bm25_score']:.6f} "
            f"| {chunk['text']}"
        )


def main() -> None:
    document_id = get_document_id()

    print(f"DOCUMENT ID: {document_id}")
    print(f"BM25 CANDIDATES: top {TOP_K}")
    print(f"BENCHMARK QUESTIONS: {len(TEST_CASES)}")

    for case_number, case in enumerate(TEST_CASES, start=1):
        question = case["question"]
        ranked_chunks = retrieve_keyword_chunks(
            question,
            document_id,
            top_k=TOP_K,
        )

        print("\n" + "=" * 100)
        print(f"TEST CASE {case_number}")
        print(f"QUESTION: {question}")
        print("BM25 RANKING:")
        print_ranking(ranked_chunks)


if __name__ == "__main__":
    main()
