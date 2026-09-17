"""Compare the local MPNet-only and hybrid reranked retrieval pipelines."""

import math

from backend.embedding import embed_texts
from backend.generator import rewrite_query
from backend.hybrid_retriever import reciprocal_rank_fusion
from backend.keyword_retriever import retrieve_keyword_chunks
from backend.reranker import rerank_chunks
from backend.vector_store import collection

try:
    from backend.tests.evaluation_dataset import DOCUMENT_ID, EVALUATION_QUESTIONS
except ModuleNotFoundError:
    from evaluation_dataset import DOCUMENT_ID, EVALUATION_QUESTIONS


TOP_K = 10
FINAL_K = 3


def chunk_identity(chunk: dict) -> tuple[str, int]:
    metadata = chunk.get("metadata", {})
    return metadata["document_id"], int(metadata["chunk_index"])


def retrieve_mpnet_chunks(query: str, document_id: str) -> list[dict]:
    results = collection.query(
        query_embeddings=embed_texts([query]).tolist(),
        n_results=TOP_K,
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )
    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    return [
        {"text": text, "metadata": metadata or {}}
        for text, metadata in zip(documents, metadatas)
    ]


def retrieve_v1(query: str, document_id: str) -> list[dict]:
    return retrieve_mpnet_chunks(query, document_id)


def retrieve_v2(query: str, document_id: str) -> list[dict]:
    vector_chunks = retrieve_mpnet_chunks(query, document_id)
    keyword_chunks = retrieve_keyword_chunks(query, document_id, top_k=TOP_K)
    fused_chunks = reciprocal_rank_fusion([vector_chunks, keyword_chunks])
    return rerank_chunks(query, fused_chunks)


def recall_at_k(ranked_chunks: list[dict], gold_chunks: set[tuple[str, int]], k: int) -> float:
    if not gold_chunks:
        return 0.0
    retrieved = {chunk_identity(chunk) for chunk in ranked_chunks[:k]}
    return len(retrieved & gold_chunks) / len(gold_chunks)


def precision_at_k(ranked_chunks: list[dict], gold_chunks: set[tuple[str, int]], k: int) -> float:
    retrieved = ranked_chunks[:k]
    if not retrieved:
        return 0.0
    relevant = sum(chunk_identity(chunk) in gold_chunks for chunk in retrieved)
    return relevant / len(retrieved)


def reciprocal_rank(ranked_chunks: list[dict], gold_chunks: set[tuple[str, int]]) -> float:
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if chunk_identity(chunk) in gold_chunks:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_chunks: list[dict], gold_chunks: set[tuple[str, int]], k: int) -> float:
    def dcg(chunks: list[dict]) -> float:
        return sum(
            (1.0 if chunk_identity(chunk) in gold_chunks else 0.0)
            / math.log2(rank + 2)
            for rank, chunk in enumerate(chunks)
        )

    actual = dcg(ranked_chunks[:k])
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(gold_chunks))))
    return actual / ideal if ideal else 0.0


def calculate_metrics(ranked_chunks: list[dict], gold_chunks: set[tuple[str, int]]) -> dict[str, float]:
    return {
        "Recall@3": recall_at_k(ranked_chunks, gold_chunks, 3),
        "Recall@10": recall_at_k(ranked_chunks, gold_chunks, 10),
        "MRR": reciprocal_rank(ranked_chunks, gold_chunks),
        "NDCG@3": ndcg_at_k(ranked_chunks, gold_chunks, 3),
        "NDCG@10": ndcg_at_k(ranked_chunks, gold_chunks, 10),
        "Precision@3": precision_at_k(ranked_chunks, gold_chunks, 3),
    }


def evaluate_pipeline(retriever) -> list[dict]:
    results = []
    for case in EVALUATION_QUESTIONS:
        rewritten_query = rewrite_query(case["question"], case["history"])
        ranked_chunks = retriever(rewritten_query, DOCUMENT_ID)
        results.append({
            "question": case["question"],
            "rewritten_query": rewritten_query,
            "ranked_chunks": ranked_chunks,
            "metrics": calculate_metrics(ranked_chunks, case["gold_chunks"]),
        })
    return results


def aggregate_metrics(results: list[dict]) -> dict[str, float]:
    metric_names = list(results[0]["metrics"])
    return {
        name: sum(result["metrics"][name] for result in results) / len(results)
        for name in metric_names
    }


def print_question_results(version: str, results: list[dict]) -> None:
    print(f"\n{version} per-question results")
    for index, result in enumerate(results, start=1):
        metrics = result["metrics"]
        print(
            f"{index:>2}. {result['question']} | "
            f"R@3={metrics['Recall@3']:.3f} "
            f"R@10={metrics['Recall@10']:.3f} "
            f"MRR={metrics['MRR']:.3f} "
            f"NDCG@3={metrics['NDCG@3']:.3f} "
            f"NDCG@10={metrics['NDCG@10']:.3f} "
            f"P@3={metrics['Precision@3']:.3f}"
        )


def print_comparison(v1_metrics: dict[str, float], v2_metrics: dict[str, float]) -> None:
    print("\nFinal comparison")
    print(f"{'Metric':<12} {'V1 MPNet':>10} {'V2 Hybrid':>10}")
    print("-" * 36)
    for metric in v1_metrics:
        print(f"{metric:<12} {v1_metrics[metric]:>10.3f} {v2_metrics[metric]:>10.3f}")


def main() -> None:
    v1_results = evaluate_pipeline(retrieve_v1)
    v2_results = evaluate_pipeline(retrieve_v2)

    print_question_results("V1 MPNet-only", v1_results)
    print_question_results("V2 hybrid + reranking", v2_results)
    print_comparison(aggregate_metrics(v1_results), aggregate_metrics(v2_results))


if __name__ == "__main__":
    main()