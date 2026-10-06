"""Hybrid retrieval using vector search + BM25."""

from __future__ import annotations

from typing import Any

from .retrieval import search_bm25, search_vector


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Combine vector and BM25 rankings using Reciprocal Rank Fusion."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    fused: dict[int, dict[str, Any]] = {}

    # Vector ranking
    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result["id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                **result,
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
            }

        fused[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)
        fused[chunk_id]["vector_rank"] = rank

    # BM25 ranking
    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                **result,
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
            }

        fused[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)
        fused[chunk_id]["bm25_rank"] = rank

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )[:top_k]


def search_hybrid(
    conn,
    query: str,
    *,
    retrieval_k: int = 10,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run vector + BM25 retrieval and combine using RRF."""

    vector_results = search_vector(
        conn,
        query,
        top_k=retrieval_k,
    )

    bm25_results = search_bm25(
        query,
        top_k=retrieval_k,
    )

    return reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        top_k=top_k,
    )