from contextlib import contextmanager

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


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        label="S1",
        chunk_id="hamlet:1",
        source_id="hamlet",
        text="Barnardo asks who is there.",
        metadata={"work": "Hamlet", "title": "Hamlet", "author": "William Shakespeare"},
        combined_score=1.0,
        reason="test",
    )


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
