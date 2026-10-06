"""Semantic retrieval from pgvector."""

import json
from typing import Any

import psycopg
import requests

from .config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    TOP_K,
)


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    body = response.json()
    embeddings = body.get("embeddings") or body.get("embedding")
    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
        return [float(x) for x in embeddings[0]]
    if isinstance(embeddings, list):
        return [float(x) for x in embeddings]
    raise ValueError("Unexpected embedding response")


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def retrieve(question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    embedding = _vector_literal(get_embedding(question))
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    dc.id,
                    dc.chunk_index,
                    dc.content,
                    dc.source,
                    dc.page,
                    dc.section_path,
                    dc.chunk_type,
                    dc.element_ids,
                    dc.evidence,
                    1 - (dc.embedding <=> %s::vector) AS similarity,
                    d.filename,
                    d.version
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE d.status = 'active'
                  AND dc.embedding IS NOT NULL
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for row in rows:
        results.append(
            {
                "id": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "source": row[3],
                "page": row[4],
                "section_path": row[5],
                "chunk_type": row[6],
                "element_ids": row[7],
                "evidence": row[8],
                "similarity": float(row[9]),
                "filename": row[10],
                "version": row[11],
            }
        )
    return results
