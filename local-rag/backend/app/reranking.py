"""Second-stage reranking using a lightweight local cross-encoder."""

from __future__ import annotations

from typing import Any

from fastembed.rerank.cross_encoder import TextCrossEncoder


MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"

_reranker: TextCrossEncoder | None = None


def get_reranker() -> TextCrossEncoder:
    global _reranker

    if _reranker is None:
        _reranker = TextCrossEncoder(
            model_name=MODEL_NAME
        )

    return _reranker


def rerank(
    query: str,
    results: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rerank retrieved candidates using query-document relevance."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not results:
        return []

    reranker = get_reranker()

    documents = [
        result["content"]
        for result in results
    ]

    scores = list(
        reranker.rerank(
            query,
            documents,
        )
    )

    reranked = []

    for result, score in zip(results, scores):
        item = dict(result)
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    for rank, result in enumerate(reranked, start=1):
        result["rerank_rank"] = rank

    return reranked[:top_k]