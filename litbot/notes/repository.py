import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from litbot.config import Settings, get_settings
from litbot.models import RetrievedChunk
from litbot.notes.grounding import GroundedNote

PENDING_ACTION_TTL = timedelta(minutes=10)


class NoteRepository:
    """Persistence boundary for notes and their supporting chunks."""

    def __init__(self, conn: Connection, settings: Settings | None = None) -> None:
        self.conn = conn
        self.settings = settings or get_settings()

    def insert(self, *, note_id: str, grounded: GroundedNote, trace_id: str) -> None:
        source_id = _shared_source_id(grounded.selected_chunks)
        work_metadata = _work_metadata(grounded.inferred_work, grounded.selected_chunks)
        with self.conn.transaction():
            self.conn.execute(
                """
                INSERT INTO notes
                    (note_id, original_input, rewritten_note, inferred_work, source_id,
                     work_metadata, embedding, model, prompt_version, trace_id, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, 'saved')
                """,
                [
                    note_id,
                    grounded.original_input,
                    grounded.rewritten_note,
                    grounded.inferred_work,
                    source_id,
                    Jsonb(work_metadata),
                    _vector_literal(grounded.embedding),
                    self.settings.llm_model,
                    self.settings.note_prompt_version,
                    trace_id,
                ],
            )
            self._insert_note_chunks(note_id, grounded.selected_chunks)

    def update(self, *, note_id: str, grounded: GroundedNote, trace_id: str) -> None:
        source_id = _shared_source_id(grounded.selected_chunks)
        work_metadata = _work_metadata(grounded.inferred_work, grounded.selected_chunks)
        self.conn.execute(
            """
            UPDATE notes
            SET original_input = %s,
                rewritten_note = %s,
                inferred_work = %s,
                source_id = %s,
                work_metadata = %s,
                embedding = %s::vector,
                model = %s,
                prompt_version = %s,
                trace_id = %s,
                updated_at = now()
            WHERE note_id = %s
            """,
            [
                grounded.original_input,
                grounded.rewritten_note,
                grounded.inferred_work,
                source_id,
                Jsonb(work_metadata),
                _vector_literal(grounded.embedding),
                self.settings.llm_model,
                self.settings.note_prompt_version,
                trace_id,
                note_id,
            ],
        )
        self.conn.execute("DELETE FROM note_chunks WHERE note_id = %s", [note_id])
        self._insert_note_chunks(note_id, grounded.selected_chunks)

    def fetch(self, note_id: str) -> dict[str, Any] | None:
        if not _canonical_uuid(note_id):
            return None
        row = self.conn.execute(
            """
            SELECT note_id, original_input, rewritten_note, inferred_work, source_id, created_at
            FROM notes
            WHERE note_id = %s
            """,
            [note_id],
        ).fetchone()
        return dict(row) if row is not None else None

    def list_all_rows(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT note_id, original_input, rewritten_note, inferred_work, source_id, created_at
            FROM notes
            ORDER BY created_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def existing_ids(self, note_ids: list[str]) -> list[str]:
        if not note_ids:
            return []
        note_ids = [note_id for note_id in note_ids if _canonical_uuid(note_id)]
        if not note_ids:
            return []
        rows = self.conn.execute(
            """
            SELECT note_id
            FROM notes
            WHERE note_id = ANY(%s)
            """,
            [note_ids],
        ).fetchall()
        return [str(dict(row)["note_id"]) for row in rows]

    def delete_many(self, note_ids: list[str]) -> None:
        self.conn.execute("DELETE FROM notes WHERE note_id = ANY(%s)", [note_ids])

    def _insert_note_chunks(self, note_id: str, chunks: list[RetrievedChunk]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO note_chunks (note_id, chunk_id, rank, label)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (note_id, chunk.chunk_id, rank, chunk.label)
                    for rank, chunk in enumerate(chunks, start=1)
                ],
            )


@dataclass
class PendingNoteAction:
    action_id: str
    operation: str
    payload: dict[str, Any]
    expires_at: datetime | None
    consumed_at: datetime | None


class PendingNoteActionRepository:
    """Persistence boundary for pending note mutations."""

    def __init__(self, conn: Connection) -> None:
        self.conn = conn

    def create(self, operation: str, payload: dict[str, Any]) -> str:
        action_id = str(uuid.uuid4())
        expires_at = datetime.now(UTC) + PENDING_ACTION_TTL
        self.conn.execute(
            """
            INSERT INTO pending_note_actions (action_id, operation, payload, expires_at)
            VALUES (%s, %s, %s, %s)
            """,
            [action_id, operation, Jsonb(payload), expires_at],
        )
        return action_id

    def fetch_for_update(self, action_id: str) -> PendingNoteAction | None:
        row = self.conn.execute(
            """
            SELECT action_id, operation, payload, expires_at, consumed_at
            FROM pending_note_actions
            WHERE action_id = %s
            FOR UPDATE
            """,
            [action_id],
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        expires_at = data.get("expires_at")
        consumed_at = data.get("consumed_at")
        return PendingNoteAction(
            action_id=str(data["action_id"]),
            operation=str(data["operation"]),
            payload=json_payload(data["payload"]),
            expires_at=expires_at if isinstance(expires_at, datetime) else None,
            consumed_at=consumed_at if isinstance(consumed_at, datetime) else None,
        )

    def consume(self, action_id: str) -> None:
        self.conn.execute(
            """
            UPDATE pending_note_actions
            SET consumed_at = now(), updated_at = now()
            WHERE action_id = %s
            """,
            [action_id],
        )

    def status_error(self, action: PendingNoteAction) -> str | None:
        if action.consumed_at is not None:
            return "That pending note action has already been used."
        if action.expires_at is not None and action.expires_at <= datetime.now(UTC):
            return "That pending note action has expired."
        return None


def note_summary(row: dict[str, Any]) -> dict[str, str]:
    return {
        "note_id": str(row["note_id"]),
        "inferred_work": str(row["inferred_work"]),
        "rewritten_note": str(row["rewritten_note"]),
    }


def json_payload(payload: object) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def target_note_ids(payload: dict[str, Any]) -> list[str]:
    value = payload.get("target_note_ids")
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    single = payload.get("target_note_id")
    return [str(single)] if single else []


def valid_note_ids(note_ids: list[str]) -> list[str]:
    return [note_id for note_id in note_ids if _canonical_uuid(note_id)]


def intent_for_operation(operation: str) -> str | None:
    if operation == "edit":
        return "note_edit"
    if operation == "delete":
        return "note_delete"
    if operation == "delete_all":
        return "note_delete_all"
    return None


def response_operation(operation: str) -> str | None:
    if operation in {"edit", "delete", "delete_all"}:
        return operation
    return None


def _canonical_uuid(value: str) -> str | None:
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


def _shared_source_id(chunks: list[RetrievedChunk]) -> str | None:
    source_ids = {chunk.source_id for chunk in chunks}
    if len(source_ids) == 1:
        return chunks[0].source_id
    return None


def _work_metadata(inferred_work: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
    return {
        "work": inferred_work,
        "sources": [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
    }


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"
