CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    author TEXT,
    translator TEXT,
    editor TEXT,
    publication_year INTEGER,
    edition TEXT,
    genre TEXT,
    language TEXT NOT NULL DEFAULT 'en',
    license TEXT NOT NULL,
    uri TEXT,
    version TEXT NOT NULL DEFAULT '1',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_hash TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS chunks_document_index_idx ON chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx ON chunks USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS chunks_text_fts_idx ON chunks USING gin (to_tsvector('english', text));
CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx ON documents USING gin (metadata jsonb_path_ops);
