"""Turn reranked chunks into the passages sent to the answer model.

Retrieval works best on small, focused chunks, but an answer often needs the
sentences around the matching chunk: a definition that continues into the next
chunk, or the setup that precedes it. For prose documents each retrieved chunk
is therefore widened with its neighbors from the same section. Resume entries
and slides are self-contained units, so they are passed through unchanged.
"""

try:
    from backend.document_classifier import PROSE
except ModuleNotFoundError:
    from document_classifier import PROSE


NEIGHBORS = 1

# Longest text compared when removing the overlap sentence that adjacent
# prose chunks share.
MAX_OVERLAP_CHARS = 600


def merge_overlapping(first: str, second: str) -> str:
    """Join adjacent chunks, dropping the sentence they both contain."""
    for length in range(min(len(first), len(second), MAX_OVERLAP_CHARS), 19, -1):
        if first.endswith(second[:length]):
            return first + second[length:]
    return f"{first} {second}"


def expand_with_neighbors(chunks: list[dict], fetch_chunks) -> list[dict]:
    """Widen prose chunks with adjacent chunks from the same section.

    fetch_chunks(document_id, chunk_indexes) returns stored chunks. Passages
    keep the rank order of the chunks that produced them, and a chunk already
    included as a neighbor is not repeated.
    """

    included = set()
    passages = []

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        document_id = metadata.get("document_id")
        chunk_index = metadata.get("chunk_index")

        if metadata.get("document_type") != PROSE or document_id is None or chunk_index is None:
            passages.append(chunk)
            continue

        if (document_id, chunk_index) in included:
            continue

        wanted = [
            index
            for index in range(chunk_index - NEIGHBORS, chunk_index + NEIGHBORS + 1)
            if index >= 0 and index != chunk_index and (document_id, index) not in included
        ]
        neighbors = [
            neighbor
            for neighbor in (fetch_chunks(document_id, wanted) if wanted else [])
            if neighbor["metadata"].get("section") == metadata.get("section")
        ]
        run = sorted([chunk, *neighbors], key=lambda item: item["metadata"]["chunk_index"])
        included.update((document_id, item["metadata"]["chunk_index"]) for item in run)

        text = run[0]["text"]
        for item in run[1:]:
            text = merge_overlapping(text, item["text"])

        passages.append({
            **chunk,
            "text": text,
            "metadata": {**metadata, "page": run[0]["metadata"].get("page", metadata.get("page"))},
        })

    return passages


def format_context(passages: list[dict]) -> str:
    """Label each passage with its location so answers can be grounded."""
    formatted = []
    for number, passage in enumerate(passages, start=1):
        metadata = passage.get("metadata") or {}
        label = [f"[{number}]"]
        if metadata.get("page"):
            label.append(f"page {metadata['page']}")
        section = metadata.get("section") or metadata.get("subsection")
        if section:
            label.append(section)
        formatted.append(f"{' | '.join(label)}\n{passage['text']}")
    return "\n\n".join(formatted)
