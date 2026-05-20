CREATE TABLE IF NOT EXISTS notes (
    note_id        UUID PRIMARY KEY,
    original_input TEXT NOT NULL,
    rewritten_note TEXT NOT NULL,
    inferred_work  TEXT NOT NULL,
    source_id      TEXT,
    work_metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding      vector(1536) NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    trace_id       TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'saved',
    status_reason  TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS note_chunks (
    note_id  UUID NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    rank     INTEGER NOT NULL,
    label    TEXT NOT NULL,
    PRIMARY KEY (note_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS notes_embedding_hnsw_idx
    ON notes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS notes_rewritten_note_fts_idx
    ON notes USING gin (to_tsvector('english', rewritten_note));
CREATE INDEX IF NOT EXISTS notes_rewritten_note_trgm_idx
    ON notes USING gin (rewritten_note gin_trgm_ops);
CREATE INDEX IF NOT EXISTS notes_inferred_work_lower_idx
    ON notes(lower(inferred_work));
CREATE INDEX IF NOT EXISTS notes_model_idx
    ON notes(model);
CREATE INDEX IF NOT EXISTS notes_prompt_version_idx
    ON notes(prompt_version);
CREATE INDEX IF NOT EXISTS notes_inferred_work_idx
    ON notes(inferred_work);
CREATE INDEX IF NOT EXISTS notes_source_id_idx
    ON notes(source_id);
CREATE INDEX IF NOT EXISTS notes_work_metadata_gin_idx
    ON notes USING gin (work_metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS note_chunks_chunk_id_idx
    ON note_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS note_chunks_note_rank_idx
    ON note_chunks(note_id, rank);
