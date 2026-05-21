CREATE TABLE IF NOT EXISTS pending_note_actions (
    action_id   UUID PRIMARY KEY,
    operation   TEXT NOT NULL,
    payload     JSONB NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pending_note_actions_expires_at_idx
    ON pending_note_actions(expires_at);
CREATE INDEX IF NOT EXISTS pending_note_actions_consumed_at_idx
    ON pending_note_actions(consumed_at);
