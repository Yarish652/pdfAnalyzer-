def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    scores = {}
    chunks = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = (
                chunk["metadata"]["document_id"],
                chunk["metadata"]["chunk_index"],
            )

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk_id] = chunk

    fused = []

    for chunk_id, score in scores.items():
        chunk = chunks[chunk_id].copy()
        chunk["rrf_score"] = score
        fused.append(chunk)

    return sorted(
        fused,
        key=lambda chunk: chunk["rrf_score"],
        reverse=True,
    )