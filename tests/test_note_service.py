from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest

from litbot.config import Settings
from litbot.models import NoteProcessingPayload, RetrievedChunk
from litbot.notes import service as note_module
from litbot.notes.service import DEFAULT_REJECTION_REASON, NoteService


def test_note_processing_payload_has_closed_citation_map_schema() -> None:
    schema = NoteProcessingPayload.model_json_schema()
    citation_map = schema["properties"]["citation_map"]
    ref = citation_map["items"]["$ref"]
    ref_name = ref.removeprefix("#/$defs/")

    assert schema["additionalProperties"] is False
    assert schema["$defs"][ref_name]["additionalProperties"] is False
    assert set(schema["$defs"][ref_name]["properties"]) == {"claim", "sources"}


def test_note_service_saves_grounded_literary_note(monkeypatch) -> None:
    conn = FakeConnection()
    monkeypatch.setattr(note_module, "embed_query", lambda text, settings: [0.1, 0.2])
    service = NoteService(
        conn,
        settings=Settings(llm_model="test-model", note_prompt_version="note-v1"),
        retrieval_service=FakeRetrievalService([_chunk()]),
        model=FakeModel(
            NoteProcessingPayload(
                should_save=True,
                rewritten_note="Hamlet opens with watchful uncertainty.",
                inferred_work="Hamlet",
                selected_chunk_ids=["hamlet:1"],
                citation_map=[{"claim": "watchful uncertainty", "sources": ["S1"]}],
            )
        ),
    )

    response = service.process(
        original_input="note: Hamlet starts uncertain",
        note_text="Hamlet starts uncertain",
        trace_id="trace-1",
    )

    assert response.note_status == "saved"
    assert response.note == "Hamlet opens with watchful uncertainty."
    assert response.original_note == "note: Hamlet starts uncertain"
    assert response.note_work == "Hamlet"
    assert response.note_id is not None
    assert response.note_chunk_ids == ["hamlet:1"]
    assert len(conn.note_rows) == 1
    assert len(conn.note_chunk_rows) == 1


def test_note_service_rejects_when_no_chunks_are_retrieved(monkeypatch) -> None:
    conn = FakeConnection()
    monkeypatch.setattr(note_module, "embed_query", lambda text, settings: [0.1])
    service = NoteService(
        conn,
        settings=Settings(),
        retrieval_service=FakeRetrievalService([]),
        model=FakeModel(NoteProcessingPayload(should_save=True)),
    )

    response = service.process(
        original_input="save this",
        note_text="save this",
        trace_id="trace-1",
    )

    assert response.note_status == "not_saved"
    assert response.note_rejection_reason == "No relevant chunks were retrieved."
    assert conn.note_rows == []


def test_note_service_rejects_non_literary_note_without_insert(monkeypatch) -> None:
    conn = FakeConnection()
    monkeypatch.setattr(note_module, "embed_query", lambda text, settings: [0.1])
    service = NoteService(
        conn,
        settings=Settings(),
        retrieval_service=FakeRetrievalService([_chunk()]),
        model=FakeModel(
            NoteProcessingPayload(
                should_save=False,
                rejection_reason="The note is not about a corpus work.",
            )
        ),
    )

    response = service.process(
        original_input="remember to buy milk",
        note_text="remember to buy milk",
        trace_id="trace-1",
    )

    assert response.note_status == "not_saved"
    assert response.note_rejection_reason == "The note is not about a corpus work."
    assert conn.note_rows == []


def test_note_service_rejects_selected_chunk_ids_outside_retrieved_results(monkeypatch) -> None:
    conn = FakeConnection()
    monkeypatch.setattr(note_module, "embed_query", lambda text, settings: [0.1])
    service = NoteService(
        conn,
        settings=Settings(),
        retrieval_service=FakeRetrievalService([_chunk()]),
        model=FakeModel(
            NoteProcessingPayload(
                should_save=True,
                rewritten_note="Grounded note.",
                inferred_work="Hamlet",
                selected_chunk_ids=["outside"],
            )
        ),
    )

    response = service.process(
        original_input="save this",
        note_text="save this",
        trace_id="trace-1",
    )

    assert response.note_status == "not_saved"
    assert "outside" in response.note_rejection_reason
    assert conn.note_rows == []


def test_note_service_uses_default_reason_for_blank_rejection(monkeypatch) -> None:
    conn = FakeConnection()
    monkeypatch.setattr(note_module, "embed_query", lambda text, settings: [0.1])
    service = NoteService(
        conn,
        settings=Settings(),
        retrieval_service=FakeRetrievalService([_chunk()]),
        model=FakeModel(NoteProcessingPayload(should_save=False, rejection_reason="  ")),
    )

    response = service.process(
        original_input="save this",
        note_text="save this",
        trace_id="trace-1",
    )

    assert response.note_status == "not_saved"
    assert response.note_rejection_reason == DEFAULT_REJECTION_REASON


def test_note_insert_rolls_back_if_chunk_link_insert_fails(monkeypatch) -> None:
    conn = FakeConnection(fail_chunk_insert=True)
    monkeypatch.setattr(note_module, "embed_query", lambda text, settings: [0.1])
    service = NoteService(
        conn,
        settings=Settings(),
        retrieval_service=FakeRetrievalService([_chunk()]),
        model=FakeModel(
            NoteProcessingPayload(
                should_save=True,
                rewritten_note="Grounded note.",
                inferred_work="Hamlet",
                selected_chunk_ids=["hamlet:1"],
            )
        ),
    )

    with pytest.raises(RuntimeError, match="chunk insert failed"):
        service.process(
            original_input="save this",
            note_text="save this",
            trace_id="trace-1",
        )

    assert conn.note_rows == []
    assert conn.note_chunk_rows == []


def test_edit_preview_rejects_ungrounded_note_without_pending_action(monkeypatch) -> None:
    conn = FakeMutationConnection(note_rows=[_note_row()])
    monkeypatch.setattr(note_module, "embed_query", lambda text, settings: [0.1])
    service = NoteService(
        conn,
        settings=Settings(),
        retrieval_service=FakeRetrievalService([]),
        model=FakeModel(NoteProcessingPayload(should_save=True)),
    )

    response = service.preview_edit(
        original_input="Edit this to mention fog.",
        note_text="Mention fog.",
        target_reference="note-1",
        note_context=None,
        filters=None,
        top_k=None,
        trace_id="trace-1",
        intent_confidence=0.9,
    )

    assert response.note_operation_status == "rejected"
    assert response.pending_note_action_id is None
    assert conn.pending_actions == {}


def test_confirm_delete_consumes_action_and_double_confirm_is_rejected(monkeypatch) -> None:
    conn = FakeMutationConnection(note_rows=[_note_row()])
    monkeypatch.setattr(note_module, "embed_query", lambda text, settings: [0.1])
    service = NoteService(conn, settings=Settings(), retrieval_service=FakeRetrievalService([]))

    preview = service.preview_delete(
        target_reference="note-1",
        note_context=None,
        trace_id="trace-1",
        intent_confidence=0.9,
    )
    first = service.confirm_pending_action(preview.pending_note_action_id, trace_id="trace-2")
    second = service.confirm_pending_action(preview.pending_note_action_id, trace_id="trace-3")

    assert first.note_operation_status == "completed"
    assert second.note_operation_status == "rejected"
    assert conn.note_rows == []
    assert conn.pending_actions[preview.pending_note_action_id]["consumed_at"] is not None


def test_confirm_after_expiration_rejects_without_mutation(monkeypatch) -> None:
    conn = FakeMutationConnection(note_rows=[_note_row()])
    conn.pending_actions["action-1"] = {
        "action_id": "action-1",
        "operation": "delete",
        "payload": {"target_note_ids": ["note-1"]},
        "expires_at": datetime.now(UTC) - timedelta(minutes=1),
        "consumed_at": None,
    }
    monkeypatch.setattr(note_module, "embed_query", lambda text, settings: [0.1])
    service = NoteService(conn, settings=Settings(), retrieval_service=FakeRetrievalService([]))

    response = service.confirm_pending_action("action-1", trace_id="trace-1")

    assert response.note_operation_status == "rejected"
    assert len(conn.note_rows) == 1
    assert conn.pending_actions["action-1"]["consumed_at"] is None


def test_confirm_delete_consumes_action_when_target_disappeared(monkeypatch) -> None:
    conn = FakeMutationConnection(note_rows=[_note_row()])
    monkeypatch.setattr(note_module, "embed_query", lambda text, settings: [0.1])
    service = NoteService(conn, settings=Settings(), retrieval_service=FakeRetrievalService([]))

    preview = service.preview_delete(
        target_reference="note-1",
        note_context=None,
        trace_id="trace-1",
        intent_confidence=0.9,
    )
    conn.note_rows = []
    response = service.confirm_pending_action(preview.pending_note_action_id, trace_id="trace-2")

    assert response.note_operation_status == "not_found"
    assert conn.pending_actions[preview.pending_note_action_id]["consumed_at"] is not None


def test_delete_all_with_zero_notes_returns_completed_without_pending_action() -> None:
    service = NoteService(
        FakeMutationConnection(note_rows=[]),
        settings=Settings(),
        retrieval_service=FakeRetrievalService([]),
    )

    response = service.preview_delete_all(trace_id="trace-1", intent_confidence=0.9)

    assert response.note_operation_status == "completed"
    assert response.pending_note_action_id is None
    assert response.target_note_ids == []


def test_confirm_edit_updates_grounded_work_and_keeps_note_id(monkeypatch) -> None:
    conn = FakeMutationConnection(note_rows=[_note_row()])
    monkeypatch.setattr(note_module, "embed_query", lambda text, settings: [0.4, 0.5])
    service = NoteService(
        conn,
        settings=Settings(llm_model="test-model", note_prompt_version="note-v2"),
        retrieval_service=FakeRetrievalService([_chunk("middlemarch:1", "Middlemarch")]),
        model=FakeModel(
            NoteProcessingPayload(
                should_save=True,
                rewritten_note="Middlemarch opens with retrospective judgment.",
                inferred_work="Middlemarch",
                selected_chunk_ids=["middlemarch:1"],
            )
        ),
    )

    preview = service.preview_edit(
        original_input="Edit note-1: Middlemarch opens retrospectively.",
        note_text="Middlemarch opens retrospectively.",
        target_reference="note-1",
        note_context=None,
        filters=None,
        top_k=None,
        trace_id="trace-1",
        intent_confidence=0.9,
    )
    response = service.confirm_pending_action(preview.pending_note_action_id, trace_id="trace-2")

    assert response.note_operation_status == "completed"
    assert conn.note_rows[0]["note_id"] == "note-1"
    assert conn.note_rows[0]["inferred_work"] == "Middlemarch"
    assert conn.note_rows[0]["rewritten_note"] == "Middlemarch opens with retrospective judgment."
    assert conn.note_chunk_rows == [("note-1", "middlemarch:1", 1, "S1")]


def _chunk(chunk_id: str = "hamlet:1", work: str = "Hamlet") -> RetrievedChunk:
    return RetrievedChunk(
        label="S1",
        chunk_id=chunk_id,
        source_id=work.lower(),
        text="Barnardo asks who is there.",
        metadata={"work": work, "title": work, "author": "William Shakespeare"},
        combined_score=1.0,
        reason="test",
    )


def _note_row() -> dict:
    return {
        "note_id": "note-1",
        "original_input": "Save this.",
        "rewritten_note": "Hamlet begins with watchful uncertainty.",
        "inferred_work": "Hamlet",
        "source_id": "hamlet",
        "created_at": datetime.now(UTC),
    }


class FakeRetrievalService:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, question, filters=None, top_k=None) -> list[RetrievedChunk]:
        return self.chunks


class FakeModel:
    def __init__(self, payload: NoteProcessingPayload) -> None:
        self.payload = payload

    def invoke(self, _prompt):
        return self.payload


class FakeConnection:
    def __init__(self, fail_chunk_insert: bool = False) -> None:
        self.fail_chunk_insert = fail_chunk_insert
        self.note_rows = []
        self.note_chunk_rows = []

    @contextmanager
    def transaction(self):
        note_snapshot = list(self.note_rows)
        chunk_snapshot = list(self.note_chunk_rows)
        try:
            yield
        except Exception:
            self.note_rows = note_snapshot
            self.note_chunk_rows = chunk_snapshot
            raise

    def execute(self, query, params):
        if "INSERT INTO notes" in query:
            self.note_rows.append(params)
        return self

    @contextmanager
    def cursor(self):
        yield self

    def executemany(self, query, rows):
        if self.fail_chunk_insert:
            raise RuntimeError("chunk insert failed")
        if "INSERT INTO note_chunks" in query:
            self.note_chunk_rows.extend(rows)


class FakeRows:
    def __init__(self, rows=None, row=None) -> None:
        self.rows = rows or []
        self.row = row

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.row


class FakeMutationConnection:
    def __init__(self, note_rows=None) -> None:
        self.note_rows = list(note_rows or [])
        self.note_chunk_rows = []
        self.pending_actions = {}

    @contextmanager
    def transaction(self):
        note_snapshot = [dict(row) for row in self.note_rows]
        chunk_snapshot = list(self.note_chunk_rows)
        pending_snapshot = {key: dict(value) for key, value in self.pending_actions.items()}
        try:
            yield
        except Exception:
            self.note_rows = note_snapshot
            self.note_chunk_rows = chunk_snapshot
            self.pending_actions = pending_snapshot
            raise

    def execute(self, query, params=None):
        params = params or []
        if "SELECT note_id, original_input, rewritten_note" in query and "WHERE note_id = %s" in query:
            note_id = params[0]
            return FakeRows(row=next((row for row in self.note_rows if row["note_id"] == note_id), None))
        if "SELECT note_id, original_input, rewritten_note" in query and "ORDER BY created_at DESC" in query:
            return FakeRows(rows=list(self.note_rows))
        if "SELECT note_id" in query and "WHERE note_id = ANY" in query:
            target_ids = set(params[0])
            return FakeRows(
                rows=[{"note_id": row["note_id"]} for row in self.note_rows if row["note_id"] in target_ids]
            )
        if "INSERT INTO pending_note_actions" in query:
            action_id, operation, payload, expires_at = params
            self.pending_actions[action_id] = {
                "action_id": action_id,
                "operation": operation,
                "payload": _unwrap_jsonb(payload),
                "expires_at": expires_at,
                "consumed_at": None,
            }
            return FakeRows()
        if "FROM pending_note_actions" in query and "FOR UPDATE" in query:
            return FakeRows(row=self.pending_actions.get(params[0]))
        if "UPDATE pending_note_actions" in query:
            self.pending_actions[params[0]]["consumed_at"] = datetime.now(UTC)
            return FakeRows()
        if "DELETE FROM notes WHERE note_id = ANY" in query:
            target_ids = set(params[0])
            self.note_rows = [row for row in self.note_rows if row["note_id"] not in target_ids]
            return FakeRows()
        if "UPDATE notes" in query:
            note_id = params[-1]
            row = next(row for row in self.note_rows if row["note_id"] == note_id)
            row.update(
                {
                    "original_input": params[0],
                    "rewritten_note": params[1],
                    "inferred_work": params[2],
                    "source_id": params[3],
                    "work_metadata": _unwrap_jsonb(params[4]),
                    "embedding": params[5],
                    "model": params[6],
                    "prompt_version": params[7],
                    "trace_id": params[8],
                }
            )
            return FakeRows()
        if "DELETE FROM note_chunks" in query:
            note_id = params[0]
            self.note_chunk_rows = [row for row in self.note_chunk_rows if row[0] != note_id]
            return FakeRows()
        return FakeRows()

    @contextmanager
    def cursor(self):
        yield self

    def executemany(self, query, rows):
        if "INSERT INTO note_chunks" in query:
            self.note_chunk_rows.extend(rows)


def _unwrap_jsonb(value):
    return getattr(value, "obj", value)
