CREATE EXTENSION IF NOT EXISTS vector;

-- -----------------------------------------------------------------------------
-- Document registry
-- -----------------------------------------------------------------------------

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
);

-- -----------------------------------------------------------------------------
-- RAG chunks
--
-- Parent chunks provide broader context and are not embedded.
-- Child chunks are retrieval units and carry embeddings.
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,

    document_id BIGINT NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    chunk_index INTEGER NOT NULL,

    chunk_type TEXT NOT NULL DEFAULT 'text',

    chunk_level TEXT NOT NULL DEFAULT 'child',

    parent_chunk_id BIGINT NULL
        REFERENCES document_chunks(id)
        ON DELETE CASCADE,

    content TEXT NOT NULL,

    source TEXT,

    page INTEGER,

    section_path TEXT,

    element_ids JSONB NOT NULL DEFAULT '[]'::jsonb,

    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,

    embedding vector(768),

    chunk_hash CHAR(64),

    UNIQUE(document_id, chunk_index)
);

-- -----------------------------------------------------------------------------
-- Indexes
-- -----------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_document_chunks_document
    ON document_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_page
    ON document_chunks(document_id, page);

CREATE INDEX IF NOT EXISTS idx_document_chunks_parent
    ON document_chunks(parent_chunk_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_hash
    ON document_chunks(document_id, chunk_hash);



CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WHERE chunk_level = 'child'
  AND embedding IS NOT NULL;