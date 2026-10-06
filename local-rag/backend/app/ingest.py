"""Database persistence for extracted/enriched chunks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from .utils import log
import os
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
)
from .utils import sha256_file, sha256_text


def get_connection():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                source_uri TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                last_modified TIMESTAMPTZ,
                file_size BIGINT,
                content_hash CHAR(64) NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                page INTEGER,
                embedding vector(768),
                UNIQUE(document_id, chunk_index)
            )
            """
        )
        # Idempotent additions for databases created with the previous schema.
        cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_type TEXT NOT NULL DEFAULT 'text'")
        cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS section_path TEXT")
        cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS element_ids JSONB NOT NULL DEFAULT '[]'::jsonb")
        cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS evidence JSONB NOT NULL DEFAULT '{}'::jsonb")
        cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_hash CHAR(64)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_document ON document_chunks(document_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_page ON document_chunks(document_id, page)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_hash ON document_chunks(document_id, chunk_hash)")

def delete_missing_documents(conn, pdf_files: list[str]) -> int:
    """
    Permanently remove database documents whose source PDF
    no longer exists in the documents directory.

    document_chunks are removed automatically through
    ON DELETE CASCADE.
    """
    current_filenames = {
        os.path.basename(path)
        for path in pdf_files
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, filename
            FROM documents
            """
        )
        existing_documents = cur.fetchall()

    deleted_count = 0

    for document_id, filename in existing_documents:
        if filename not in current_filenames:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM documents
                    WHERE id = %s
                    """,
                    (document_id,),
                )

            print(f"[DELETE] removed document: {filename}")
            deleted_count += 1

    return deleted_count

def _embedding(text: str) -> list[float]:
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
    raise ValueError(f"Unexpected Ollama embedding response: {body!r}")


def _vector_literal(values):
    if isinstance(values, str):
        values = values.strip()

        if values.startswith("[") and values.endswith("]"):
            return values

        raise ValueError(
            f"Invalid vector string format: {values[:50]}"
        )

    return "[" + ",".join(str(float(v)) for v in values) + "]"


def upsert_document_and_chunks(
    conn,
    pdf_path: str,
    chunked: dict[str, list[dict[str, Any]]],
    *,
    force: bool = False,
) -> dict[str, Any]:

    import json
    import os

    filename = os.path.basename(pdf_path)
    source_uri = f"file://{filename}"
    file_size = os.path.getsize(pdf_path)
    last_modified = datetime.fromtimestamp(
        os.path.getmtime(pdf_path),
        tz=timezone.utc,
    )
    content_hash = sha256_file(pdf_path)

    parents = chunked["parents"]
    children = chunked["children"]
    existing_child_embeddings: dict[str, Any] = {}
    # ------------------------------------------------------------------
    # Document registry
    # ------------------------------------------------------------------

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, version, content_hash, status
            FROM documents
            WHERE source_uri = %s
            """,
            (source_uri,),
        )
        existing = cur.fetchone()

    if existing and existing[2] == content_hash and not force:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM document_chunks
                WHERE document_id = %s
                """,
                (existing[0],),
            )
            count = cur.fetchone()[0]

        return {
            "action": "SKIPPED_UNCHANGED",
            "document_id": existing[0],
            "version": existing[1],
            "chunks_inserted": 0,
            "embeddings_generated": 0,
            "existing_chunks": count,
        }

    if existing:
        document_id, version, _, _ = existing
        with conn.cursor() as cur:
            cur.execute(
                    """
                    SELECT
                        chunk_hash,
                        embedding
                    FROM document_chunks
                    WHERE document_id = %s
                    AND chunk_level = 'child'
                    """,
                    (document_id,),
                )

            existing_child_embeddings = {
                    row[0]: row[1]
                    for row in cur.fetchall()
                    if row[0]
                }
        new_version = (
            version + 1
            if existing[2] != content_hash
            else version
        )

        action = "UPDATED"

        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_chunks WHERE document_id = %s",
                (document_id,),
            )

            cur.execute(
                """
                UPDATE documents
                SET
                    version = %s,
                    last_modified = %s,
                    file_size = %s,
                    content_hash = %s,
                    status = 'processing',
                    ingested_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (
                    new_version,
                    last_modified,
                    file_size,
                    content_hash,
                    document_id,
                ),
            )

    else:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (
                        source_uri,
                        filename,
                        version,
                        last_modified,
                        file_size,
                        content_hash,
                        status
                    )
                VALUES (%s, %s, 1, %s, %s, %s, 'processing')
                RETURNING id
                """,
                (
                    source_uri,
                    filename,
                    last_modified,
                    file_size,
                    content_hash,
                ),
            )

            document_id = cur.fetchone()[0]

        new_version = 1
        action = "CREATED"

    # ------------------------------------------------------------------
    # Insert parent chunks first
    # ------------------------------------------------------------------

    parent_db_ids: dict[str, int] = {}

    for parent in parents:

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks
                    (
                        document_id,
                        chunk_index,
                        chunk_type,
                        chunk_level,
                        parent_chunk_id,
                        content,
                        source,
                        page,
                        section_path,
                        element_ids,
                        evidence,
                        embedding,
                        chunk_hash
                    )
                VALUES
                    (
                        %s,
                        %s,
                        %s,
                        'parent',
                        NULL,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::jsonb,
                        %s::jsonb,
                        NULL,
                        %s
                    )
                RETURNING id
                """,
                (
                    document_id,
                    parent["chunk_index"],
                    parent.get("chunk_type", "text"),
                    parent["content"],
                    filename,
                    parent.get("page"),
                    " > ".join(parent.get("section_path") or []) or None,
                    json.dumps(parent.get("element_ids", [])),
                    json.dumps(parent.get("evidence", {})),
                    sha256_text(parent["content"]),
                ),
            )

            parent_db_id = cur.fetchone()[0]

        parent_db_ids[parent["parent_id"]] = parent_db_id

    # ------------------------------------------------------------------
    # Insert child chunks + embeddings
    # ------------------------------------------------------------------

    embeddings_generated = 0
    embeddings_reused = 0
    print(f"[HASH] existing_child_hashes={len(existing_child_embeddings)}")

    for child in children:

        parent_id = child["parent_id"]

        if parent_id not in parent_db_ids:
            raise ValueError(
                f"Parent '{parent_id}' not found for child "
                f"'{child['child_id']}'"
            )

        parent_db_id = parent_db_ids[parent_id]

        chunk_hash = sha256_text(child["content"])

        if chunk_hash in existing_child_embeddings:
            vector = _vector_literal(existing_child_embeddings[chunk_hash])
            embeddings_reused += 1
        else:
            embedding = _embedding(child["content"])
            embeddings_generated += 1
            vector = _vector_literal(embedding)

        section_path = (
            " > ".join(child.get("section_path") or [])
            or None
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks
                    (
                        document_id,
                        chunk_index,
                        chunk_type,
                        chunk_level,
                        parent_chunk_id,
                        content,
                        source,
                        page,
                        section_path,
                        element_ids,
                        evidence,
                        embedding,
                        chunk_hash
                    )
                VALUES
                    (
                        %s,
                        %s,
                        %s,
                        'child',
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::jsonb,
                        %s::jsonb,
                        %s::vector,
                        %s
                    )
                """,
                (
                    document_id,
                    child["chunk_index"],
                    child.get("chunk_type", "text"),
                    parent_db_id,
                    child["content"],
                    filename,
                    child.get("page"),
                    section_path,
                    json.dumps(child.get("element_ids", [])),
                    json.dumps(child.get("evidence", {})),
                    vector,
                    chunk_hash,
                ),
            )

    # ------------------------------------------------------------------
    # Activate document only after all inserts succeed
    # ------------------------------------------------------------------

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE documents
            SET
                status = 'active',
                ingested_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            (document_id,),
        )
    print(
    f"[HASH] embeddings_reused={embeddings_reused} "
    f"embeddings_generated={embeddings_generated}"
    )
    return {
        "action": action,
        "document_id": document_id,
        "version": new_version,
        "parents_inserted": len(parents),
        "children_inserted": len(children),
        "chunks_inserted": len(parents) + len(children),
        "embeddings_generated": embeddings_generated,
        "embeddings_reused": embeddings_reused
    }