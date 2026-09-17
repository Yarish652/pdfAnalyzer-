"""Benchmark MPNet, BM25, and reciprocal-rank-fusion retrieval."""

from backend.embedding import embed_texts
from backend.generator import rewrite_query
from backend.hybrid_retriever import reciprocal_rank_fusion
from backend.keyword_retriever import retrieve_keyword_chunks
from backend.vector_store import collection


DOCUMENT_ID = "f0ca67f7-0602-41c2-a65f-ebc4914b8a87"
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


def retrieve_vector_chunks(query: str, document_id: str) -> list[dict]:
    query_embedding = embed_texts([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K,
        where={"document_id": document_id},
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []
    for index, (text, metadata) in enumerate(zip(documents, metadatas)):
        chunk = {
            "text": text,
            "metadata": metadata or {},
        }
        if index < len(distances):
            chunk["mpnet_distance"] = distances[index]
        chunks.append(chunk)

    return chunks


def print_ranking(
    title: str,
    chunks: list[dict],
    score_key: str | None = None,
) -> None:
    print(title)

    for rank, chunk in enumerate(chunks, start=1):
        location = chunk["metadata"].get("page", "?")
        score = (
            f" | {score_key}={chunk[score_key]:.6f}"
            if score_key and score_key in chunk
            else ""
        )
        print(f"  {rank:>2}. page={location}{score} | {chunk['text']}")


def main() -> None:
    print(f"DOCUMENT ID: {DOCUMENT_ID}")
    print(f"RETRIEVAL CANDIDATES: top {TOP_K}")
    print(f"BENCHMARK QUESTIONS: {len(TEST_CASES)}")

    for case_number, case in enumerate(TEST_CASES, start=1):
        question = case["question"]
        rewritten_query = rewrite_query(question, case["history"])
        vector_chunks = retrieve_vector_chunks(rewritten_query, DOCUMENT_ID)
        bm25_chunks = retrieve_keyword_chunks(
            rewritten_query,
            DOCUMENT_ID,
            top_k=TOP_K,
        )
        fused_chunks = reciprocal_rank_fusion([vector_chunks, bm25_chunks])

        print("\n" + "=" * 100)
        print(f"TEST CASE {case_number}")
        print(f"ORIGINAL QUESTION: {question}")
        print(f"REWRITTEN QUERY: {rewritten_query}")
        print_ranking("MPNET / VECTOR RANKING:", vector_chunks, "mpnet_distance")
        print_ranking("BM25 RANKING:", bm25_chunks, "bm25_score")
        print_ranking("RRF RANKING:", fused_chunks, "rrf_score")


if __name__ == "__main__":
    main()
