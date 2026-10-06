"""Context assembly for final RAG generation."""

from __future__ import annotations

from typing import Any

from .retrieval import fetch_parent_context


def build_context(
    conn,
    reranked_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Expand final reranked child chunks into unique parent contexts.
    """

    if not reranked_results:
        return []

    expanded = fetch_parent_context(
        conn,
        reranked_results,
    )

    seen_parent_ids: set[int] = set()
    context: list[dict[str, Any]] = []

    for result in expanded:
        parent = result.get("parent")

        if not parent:
            continue

        parent_id = parent["id"]

        # Multiple children can point to the same parent.
        if parent_id in seen_parent_ids:
            continue

        seen_parent_ids.add(parent_id)

        context.append(
            {
                "parent_id": parent_id,
                "filename": result["filename"],
                "page": parent["page"],
                "section_path": parent.get("section_path"),
                "chunk_type": parent["chunk_type"],
                "content": parent["content"],
                "source_child_chunk": result["chunk_index"],
            }
        )

    return context