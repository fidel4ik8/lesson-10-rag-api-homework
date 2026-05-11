-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- RAG document chunks
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    source       TEXT NOT NULL,
    chunk_index  INT  NOT NULL,
    embedding    vector(384) NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Semantic cache (separate table; same DB to avoid extra service)
CREATE TABLE IF NOT EXISTS cache_entries (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query         TEXT NOT NULL,
    response      TEXT NOT NULL,
    embedding     vector(384) NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INT  NOT NULL DEFAULT 0,
    output_tokens INT  NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    expires_at    TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS cache_embedding_idx
    ON cache_entries USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS cache_expires_idx
    ON cache_entries (expires_at);

-- Cost tracking
CREATE TABLE IF NOT EXISTS cost_records (
    request_id      UUID PRIMARY KEY,
    api_key         TEXT NOT NULL,
    model           TEXT NOT NULL,
    input_tokens    INT  NOT NULL DEFAULT 0,
    output_tokens   INT  NOT NULL DEFAULT 0,
    cost_usd        NUMERIC(12, 8) NOT NULL DEFAULT 0,
    latency_ms      INT  NOT NULL DEFAULT 0,
    ttft_ms         INT,
    cache_hit       BOOLEAN NOT NULL DEFAULT FALSE,
    fallback_used   BOOLEAN NOT NULL DEFAULT FALSE,
    output_filtered BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS cost_records_api_key_created_idx
    ON cost_records (api_key, created_at DESC);

CREATE INDEX IF NOT EXISTS cost_records_created_idx
    ON cost_records (created_at DESC);
