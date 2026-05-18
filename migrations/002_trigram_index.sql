CREATE INDEX IF NOT EXISTS chunks_text_trgm_idx
    ON chunks USING gin (text gin_trgm_ops);
