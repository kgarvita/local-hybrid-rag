"""Retrieval functions for the RAG system."""

from __future__ import annotations
from .opensearch_index import INDEX_NAME, get_opensearch_client
from typing import Any

import requests

from .config import (
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
)


def search_bm25(
    query: str,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search child chunks using OpenSearch BM25."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    client = get_opensearch_client()

    response = client.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "query": {
                "match": {
                    "content": {
                        "query": query,
                    }
                }
            },
        },
    )

    results = []

    for hit in response["hits"]["hits"]:
        source = hit["_source"]

        results.append(
            {
                "id": source["chunk_id"],
                "document_id": source["document_id"],
                "filename": source["filename"],
                "chunk_index": source["chunk_index"],
                "content": source["content"],
                "page": source["page"],
                "section_path": source.get("section_path"),
                "chunk_type": source["chunk_type"],
                "parent_chunk_id": source["parent_chunk_id"],
                "score": float(hit["_score"]),
            }
        )

    return results

def embed_query(query: str) -> list[float]:
    """Create an embedding for the user's query using Ollama."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": query,
        },
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    body = response.json()
    embeddings = body.get("embeddings") or body.get("embedding")

    if isinstance(embeddings, list) and embeddings:
        if isinstance(embeddings[0], list):
            return [float(x) for x in embeddings[0]]
        return [float(x) for x in embeddings]

    raise ValueError(f"Unexpected Ollama embedding response: {body!r}")


def vector_literal(values: list[float]) -> str:
    """Convert a Python embedding list into pgvector literal syntax."""

    return "[" + ",".join(str(float(v)) for v in values) + "]"


def search_vector(
    conn,
    query: str,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search active child chunks using pgvector cosine similarity."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    query_embedding = embed_query(query)
    query_vector = vector_literal(query_embedding)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                c.id,
                c.document_id,
                d.filename,
                c.chunk_index,
                c.content,
                c.page,
                c.section_path,
                c.chunk_type,
                c.parent_chunk_id,
                1 - (c.embedding <=> %s::vector) AS similarity
            FROM document_chunks AS c
            JOIN documents AS d
                ON d.id = c.document_id
            WHERE
                d.status = 'active'
                AND c.chunk_level = 'child'
                AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, query_vector, top_k),
        )

        rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "document_id": row[1],
            "filename": row[2],
            "chunk_index": row[3],
            "content": row[4],
            "page": row[5],
            "section_path": row[6],
            "chunk_type": row[7],
            "parent_chunk_id": row[8],
            "similarity": float(row[9]),
        }
        for row in rows
    ]

def fetch_parent_context(
    conn,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fetch the parent chunk associated with each retrieved child."""

    parent_ids = list({
        result["parent_chunk_id"]
        for result in results
        if result.get("parent_chunk_id") is not None
    })

    if not parent_ids:
        return results

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                content,
                page,
                section_path,
                chunk_type
            FROM document_chunks
            WHERE id = ANY(%s)
            """,
            (parent_ids,),
        )

        parents = {
            row[0]: {
                "id": row[0],
                "content": row[1],
                "page": row[2],
                "section_path": row[3],
                "chunk_type": row[4],
            }
            for row in cur.fetchall()
        }

    enriched_results = []

    for result in results:
        item = dict(result)
        item["parent"] = parents.get(result["parent_chunk_id"])
        enriched_results.append(item)

    return enriched_results