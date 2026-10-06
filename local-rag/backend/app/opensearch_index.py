"""OpenSearch indexing for RAG child chunks."""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch
from typing import Any

from .ingest import get_connection


INDEX_NAME = "rag-chunks"

def load_child_chunks(conn) -> list[dict[str, Any]]:
    """Load active child chunks from PostgreSQL."""

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
                c.chunk_type,
                c.chunk_level,
                c.parent_chunk_id,
                c.section_path
            FROM document_chunks AS c
            JOIN documents AS d
                ON d.id = c.document_id
            WHERE
                d.status = 'active'
                AND c.chunk_level = 'child'
            ORDER BY c.id
            """
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
            "chunk_type": row[6],
            "chunk_level": row[7],
            "parent_chunk_id": row[8],
            "section_path": row[9],
        }
        for row in rows
    ]

def get_opensearch_client() -> OpenSearch:
    return OpenSearch(
        hosts=[
            {
                "host": "opensearch",
                "port": 9200,
            }
        ],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )


def create_index(client: OpenSearch) -> None:
    """Create the BM25 index if it does not already exist."""

    if client.indices.exists(index=INDEX_NAME):
        return

    client.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "document_id": {"type": "long"},
                    "chunk_id": {"type": "long"},
                    "chunk_index": {"type": "integer"},
                    "filename": {"type": "keyword"},
                    "content": {"type": "text"},
                    "page": {"type": "integer"},
                    "chunk_type": {"type": "keyword"},
                    "chunk_level": {"type": "keyword"},
                    "parent_chunk_id": {"type": "long"},
                    "section_path": {"type": "text"},
                }
            }
        },
    )


def index_chunks(
    client: OpenSearch,
    chunks: list[dict[str, Any]],
) -> None:
    """Index child chunks for BM25 search."""

    for chunk in chunks:
        document = {
            "document_id": chunk["document_id"],
            "chunk_id": chunk["id"],
            "chunk_index": chunk["chunk_index"],
            "filename": chunk["filename"],
            "content": chunk["content"],
            "page": chunk["page"],
            "chunk_type": chunk["chunk_type"],
            "chunk_level": chunk["chunk_level"],
            "parent_chunk_id": chunk["parent_chunk_id"],
            "section_path": chunk["section_path"],
        }

        client.index(
            index=INDEX_NAME,
            id=str(chunk["id"]),
            body=document,
            refresh=True,
        )

if __name__ == "__main__":
    client = get_opensearch_client()

    create_index(client)

    conn = get_connection()

    try:
        chunks = load_child_chunks(conn)

        print(f"[OPENSEARCH] loading {len(chunks)} child chunks")

        index_chunks(client, chunks)

        print(
            f"[OPENSEARCH] indexed={len(chunks)} "
            f"count={client.count(index=INDEX_NAME)['count']}"
        )

    finally:
        conn.close()

    print(f"[OPENSEARCH] index ready: {INDEX_NAME}")