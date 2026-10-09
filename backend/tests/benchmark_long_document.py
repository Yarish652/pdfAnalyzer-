"""End-to-end quality benchmark for long, technical PDFs.

Ingests a PDF into an isolated in-memory Chroma collection (the persistent
backend/chroma_db is never modified), then reports:

1. Ingestion diagnostics: timing, chunk size distribution, fragment and noise
   rates, and section metadata quality.
2. Retrieval quality for MPNet-only, BM25-only, and the V2 hybrid + rerank
   pipeline, using chunker-independent evidence-phrase gold labels.
3. Optional answer quality (--generate): calls the configured OpenRouter model
   with the top 3 V2 chunks and checks answer keywords and refusals.

Run from the repository root:

    python -m backend.tests.benchmark_long_document --pdf "path/to/Sequence Modeling.pdf"
    python -m backend.tests.benchmark_long_document --pdf "..." --generate --report report.json
    python -m backend.tests.benchmark_long_document --pdf resume.pdf --dataset resume
"""

import argparse
import importlib
import json
import re
import statistics
import time
import unicodedata
from uuid import uuid4

import chromadb

from backend import keyword_retriever, vector_store
from backend.chunker import chunk_document
from backend.context_builder import expand_with_neighbors, format_context
from backend.embedding import embed_texts
from backend.hybrid_retriever import reciprocal_rank_fusion
from backend.keyword_retriever import retrieve_keyword_chunks
from backend.pdf_reader import extract_document
from backend.reranker import rerank_chunks

DATASETS = {
    "long": "backend.tests.long_document_dataset",
    "resume": "backend.tests.resume_dataset",
}

# Populated by load_dataset() so the helpers below can share one question set.
EVALUATION_QUESTIONS: list[dict] = []
UNANSWERABLE_QUESTIONS: list[str] = []
REFUSAL_TEXT = ""


TOP_K = 10
FINAL_K = 3
# Must match api.RERANK_CANDIDATES (api.py cannot be imported from the repo root).
RERANK_CANDIDATES = 6
FRAGMENT_WORDS = 15
WORD_PATTERN = re.compile(r"[A-Za-z]{2,}")


def load_dataset(name: str) -> None:
    global EVALUATION_QUESTIONS, UNANSWERABLE_QUESTIONS, REFUSAL_TEXT
    dataset = importlib.import_module(DATASETS[name])
    EVALUATION_QUESTIONS = dataset.EVALUATION_QUESTIONS
    UNANSWERABLE_QUESTIONS = dataset.UNANSWERABLE_QUESTIONS
    REFUSAL_TEXT = dataset.REFUSAL_TEXT


# ---------------------------------------------------------
# Isolated ingestion
# ---------------------------------------------------------

def use_isolated_collection():
    """Point the vector store and BM25 retriever at a throwaway collection."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(f"benchmark_{uuid4().hex}")
    vector_store.collection = collection
    keyword_retriever.collection = collection
    keyword_retriever._index_cache.clear()
    return collection


def ingest(pdf_path: str, document_id: str, extract_fn=extract_document, chunk_fn=chunk_document):
    """Run the same extract -> chunk -> embed -> store steps as api.process_upload."""
    timings = {}

    start = time.perf_counter()
    document = extract_fn(pdf_path)
    timings["extract_s"] = time.perf_counter() - start

    start = time.perf_counter()
    chunks = chunk_fn(document)
    for chunk in chunks:
        chunk["metadata"]["document_id"] = document_id
    timings["chunk_s"] = time.perf_counter() - start

    start = time.perf_counter()
    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    timings["embed_s"] = time.perf_counter() - start

    start = time.perf_counter()
    vector_store.add_documents(chunks, embeddings)
    timings["store_s"] = time.perf_counter() - start

    return document, chunks, timings


# ---------------------------------------------------------
# Ingestion diagnostics
# ---------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase alphanumerics only, so lost spaces and ligatures still match."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in text if character.isalnum())


def noise_ratio(text: str) -> float:
    """Share of whitespace tokens that are not ordinary words (math, labels)."""
    tokens = text.split()
    if not tokens:
        return 1.0
    return sum(not WORD_PATTERN.search(token) for token in tokens) / len(tokens)


def ingestion_diagnostics(document, chunks) -> dict:
    word_counts = [len(chunk["text"].split()) for chunk in chunks]
    sections = [chunk["metadata"].get("section", "") for chunk in chunks]
    distinct_sections = sorted(set(filter(None, sections)))
    page_numbers = {page["page"] for page in document}

    # Repeated lines across pages are usually running headers or footers.
    line_pages = {}
    for page in document:
        for block in set(page["blocks"]):
            line_pages.setdefault(block, set()).add(page["page"])
    repeated_lines = [
        line for line, pages in line_pages.items()
        if len(pages) >= max(3, len(page_numbers) // 4) and len(line) > 10
    ]

    return {
        "pages_with_text": len(document),
        "blocks": sum(len(page["blocks"]) for page in document),
        "chunks": len(chunks),
        "words_min": min(word_counts, default=0),
        "words_median": statistics.median(word_counts) if word_counts else 0,
        "words_max": max(word_counts, default=0),
        "fragment_chunks": sum(count < FRAGMENT_WORDS for count in word_counts),
        "noisy_chunks": sum(noise_ratio(chunk["text"]) > 0.4 for chunk in chunks),
        "starts_mid_sentence": sum(chunk["text"][:1].islower() for chunk in chunks),
        "distinct_sections": len(distinct_sections),
        "suspicious_sections": [section for section in distinct_sections if len(section) <= 3],
        "repeated_header_lines": repeated_lines,
    }


def evidence_coverage(chunks) -> list[dict]:
    """Report evidence phrases that no single chunk contains.

    A phrase present in the full text but in no chunk was split across a chunk
    boundary, which makes it unretrievable as a unit.
    """
    full_text = normalize(" ".join(chunk["text"] for chunk in chunks))
    normalized_chunks = [normalize(chunk["text"]) for chunk in chunks]
    problems = []

    for case in EVALUATION_QUESTIONS:
        for phrase in case["evidence"]:
            needle = normalize(phrase)
            if any(needle in text for text in normalized_chunks):
                continue
            problems.append({
                "question": case["question"],
                "phrase": phrase,
                "reason": "split across chunks" if needle in full_text else "missing from extraction",
            })

    return problems


# ---------------------------------------------------------
# Retrievers
# ---------------------------------------------------------

def retrieve_mpnet(query: str, document_id: str) -> list[dict]:
    results = vector_store.search(embed_texts([query]), document_id, top_k=TOP_K)
    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    return [
        {"text": text, "metadata": metadata or {}}
        for text, metadata in zip(documents, metadatas)
    ]


def retrieve_bm25(query: str, document_id: str) -> list[dict]:
    return retrieve_keyword_chunks(query, document_id, top_k=TOP_K)


def retrieve_v2(query: str, document_id: str) -> list[dict]:
    """Mirror api.retrieve_context_chunks without truncating to FINAL_K.

    Like the API, only the top RERANK_CANDIDATES fused results are reranked;
    the rest keep their fused order so deeper metrics such as Hit@10 still work.
    """
    fused = reciprocal_rank_fusion([
        retrieve_mpnet(query, document_id),
        retrieve_bm25(query, document_id),
    ])
    if not fused:
        return []
    return rerank_chunks(query, fused[:RERANK_CANDIDATES]) + fused[RERANK_CANDIDATES:]


RETRIEVERS = {
    "MPNet": retrieve_mpnet,
    "BM25": retrieve_bm25,
    "V2 hybrid+rerank": retrieve_v2,
}


# ---------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------

def is_relevant(chunk: dict, case: dict) -> bool:
    text = normalize(chunk["text"])
    return any(normalize(phrase) in text for phrase in case["evidence"])


def retrieval_metrics(ranked: list[dict], case: dict) -> dict:
    relevance = [is_relevant(chunk, case) for chunk in ranked]
    first_hit = next((rank for rank, hit in enumerate(relevance, start=1) if hit), None)
    top_pages = {chunk["metadata"].get("page") for chunk in ranked[:FINAL_K]}

    return {
        "Hit@1": float(any(relevance[:1])),
        "Hit@3": float(any(relevance[:FINAL_K])),
        "Hit@10": float(any(relevance[:TOP_K])),
        "MRR": 1.0 / first_hit if first_hit else 0.0,
        "PageHit@3": float(bool(top_pages & case["pages"])),
    }


def evaluate_retrieval(document_id: str) -> dict:
    results = {}
    for name, retriever in RETRIEVERS.items():
        per_question = []
        start = time.perf_counter()
        for case in EVALUATION_QUESTIONS:
            ranked = retriever(case["question"], document_id)
            context = build_context(ranked)
            metrics = retrieval_metrics(ranked, case)
            # Whether the evidence reaches the model after neighbor expansion.
            metrics["ContextHit"] = float(any(is_relevant(passage, case) for passage in context))
            per_question.append({
                "question": case["question"],
                "metrics": metrics,
                "top_chunks": [
                    {
                        "page": chunk["metadata"].get("page"),
                        "chunk_index": chunk["metadata"].get("chunk_index"),
                        "relevant": is_relevant(chunk, case),
                        "text": chunk["text"][:200],
                    }
                    for chunk in ranked[:FINAL_K]
                ],
                "context": context,
            })
        elapsed = time.perf_counter() - start
        results[name] = {
            "per_question": per_question,
            "mean": mean_metrics(per_question),
            "avg_latency_s": elapsed / len(EVALUATION_QUESTIONS),
        }
    return results


def build_context(ranked: list[dict]) -> list[dict]:
    """The passages api.retrieve_context_chunks would hand to the model."""
    return expand_with_neighbors(ranked[:FINAL_K], vector_store.get_chunks)


def mean_metrics(per_question: list[dict]) -> dict:
    names = per_question[0]["metrics"]
    return {
        name: sum(item["metrics"][name] for item in per_question) / len(per_question)
        for name in names
    }


# ---------------------------------------------------------
# Optional generation check
# ---------------------------------------------------------

def contains_keywords(answer: str, keyword_groups: list[list[str]]) -> bool:
    """Every group must match, ignoring case, spacing and punctuation."""
    text = normalize(answer)
    return all(any(normalize(keyword) in text for keyword in group) for group in keyword_groups)


# Stop calling the provider after this many errors in a row (for example an
# exhausted daily free-tier quota) instead of failing every remaining question.
MAX_CONSECUTIVE_ERRORS = 3


def evaluate_generation(document_id: str, v2_results: list[dict], delay_s: float) -> list[dict]:
    # Imported lazily so retrieval-only runs never need an API key.
    from backend.generator import generate_answer

    cases = [
        (case["question"], "answerable", format_context(retrieved["context"]), case, retrieved)
        for case, retrieved in zip(EVALUATION_QUESTIONS, v2_results)
    ] + [
        (question, "unanswerable", None, None, None)
        for question in UNANSWERABLE_QUESTIONS
    ]

    outcomes = []
    consecutive_errors = 0
    for question, kind, context, case, retrieved in cases:
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            answer, error = "", "skipped after repeated provider errors"
        else:
            if context is None:
                context = format_context(build_context(retrieve_v2(question, document_id)))
            answer, error = _safe_generate(generate_answer, question, context)
            consecutive_errors = consecutive_errors + 1 if error else 0
            time.sleep(delay_s)

        if error:
            passed = False
        elif kind == "answerable":
            passed = contains_keywords(answer, case["answer_keywords"]) and REFUSAL_TEXT not in answer
        else:
            passed = REFUSAL_TEXT in answer

        outcomes.append({
            "question": question,
            "kind": kind,
            "answer": answer,
            "error": error,
            "retrieval_hit": bool(retrieved["metrics"]["Hit@3"]) if retrieved else None,
            "passed": passed,
        })

    return outcomes


def _safe_generate(generate_answer, question: str, context: str) -> tuple[str, str | None]:
    """Return (answer, error) so provider failures are reported, not graded."""
    try:
        return generate_answer(question, context, []), None
    except Exception as error:
        return "", f"{type(error).__name__}: {error}"


# ---------------------------------------------------------
# Reporting
# ---------------------------------------------------------

def print_ingestion(diagnostics: dict, timings: dict, coverage: list[dict]) -> None:
    print("\n=== Ingestion ===")
    print(" ".join(f"{name}={value:.2f}" for name, value in timings.items()))
    for name, value in diagnostics.items():
        if isinstance(value, list):
            preview = value[:8]
            print(f"{name:<22} {len(value)} {preview}")
        else:
            print(f"{name:<22} {value}")

    print("\nEvidence phrases not contained in any single chunk:")
    if not coverage:
        print("  none")
    for problem in coverage:
        print(f"  [{problem['reason']}] {problem['phrase']!r} ({problem['question']})")


def print_retrieval(results: dict) -> None:
    print("\n=== Retrieval ===")
    metric_names = list(next(iter(results.values()))["mean"])
    print(f"{'Pipeline':<18}" + "".join(f"{name:>11}" for name in metric_names) + f"{'latency':>10}")
    for name, result in results.items():
        row = "".join(f"{result['mean'][metric]:>11.3f}" for metric in metric_names)
        print(f"{name:<18}{row}{result['avg_latency_s']:>9.2f}s")

    print("\nV2 misses (no relevant chunk in final top 3):")
    for item in results["V2 hybrid+rerank"]["per_question"]:
        if item["metrics"]["Hit@3"]:
            continue
        print(f"- {item['question']}")
        for chunk in item["top_chunks"]:
            print(f"    p{chunk['page']} #{chunk['chunk_index']}: {chunk['text'][:110]!r}")


def print_generation(outcomes: list[dict]) -> None:
    print("\n=== Generation ===")
    for outcome in outcomes:
        status = "ERROR" if outcome["error"] else "PASS" if outcome["passed"] else "FAIL"
        print(f"[{status}] ({outcome['kind']}) {outcome['question']}")
        print(f"       {(outcome['error'] or outcome['answer'])[:300]!r}")

    for kind in ("answerable", "unanswerable"):
        subset = [outcome for outcome in outcomes if outcome["kind"] == kind]
        if subset:
            passed = sum(outcome["passed"] for outcome in subset)
            errors = sum(bool(outcome["error"]) for outcome in subset)
            print(f"{kind}: {passed}/{len(subset)} passed, {errors} provider errors")

    misses = [
        o for o in outcomes
        if o["kind"] == "answerable" and o["retrieval_hit"] and not o["passed"] and not o["error"]
    ]
    print(f"answerable failures despite a retrieval hit: {len(misses)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, help="Path to the PDF to benchmark.")
    parser.add_argument("--dataset", choices=DATASETS, default="long", help="Gold label set to evaluate.")
    parser.add_argument("--generate", action="store_true", help="Also call the LLM and grade answers.")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between LLM calls (free-tier rate limits).")
    parser.add_argument("--report", help="Optional path for a JSON report.")
    args = parser.parse_args()

    load_dataset(args.dataset)
    use_isolated_collection()
    document_id = f"benchmark-{uuid4()}"
    document, chunks, timings = ingest(args.pdf, document_id)

    diagnostics = ingestion_diagnostics(document, chunks)
    coverage = evidence_coverage(chunks)
    print_ingestion(diagnostics, timings, coverage)

    retrieval = evaluate_retrieval(document_id)
    print_retrieval(retrieval)

    generation = None
    if args.generate:
        generation = evaluate_generation(
            document_id,
            retrieval["V2 hybrid+rerank"]["per_question"],
            args.delay,
        )
        print_generation(generation)

    if args.report:
        for result in retrieval.values():
            for item in result["per_question"]:
                item.pop("context")
        with open(args.report, "w", encoding="utf-8") as report:
            json.dump({
                "pdf": args.pdf,
                "timings": timings,
                "ingestion": diagnostics,
                "evidence_coverage": coverage,
                "retrieval": retrieval,
                "generation": generation,
            }, report, indent=2, ensure_ascii=False)
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
