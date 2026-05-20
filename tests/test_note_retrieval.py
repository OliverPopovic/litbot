from datetime import UTC, datetime

from litbot.config import Settings
from litbot.models import RetrievedNote
from litbot.notes.retrieval import NoteRelevanceService, NoteRetrievalService


def test_list_all_with_zero_notes_returns_clean_empty_result() -> None:
    service = NoteRetrievalService(FakeConnection(note_rows=[]), Settings())

    result = service.list_all()

    assert result.notes == []
    assert result.has_more is False


def test_list_all_handles_missing_linked_chunks_gracefully() -> None:
    service = NoteRetrievalService(
        FakeConnection(
            note_rows=[
                {
                    "note_id": "note-1",
                    "original_input": "Save this.",
                    "rewritten_note": "Hamlet begins with uncertainty.",
                    "inferred_work": "Hamlet",
                    "source_id": "hamlet",
                    "created_at": datetime.now(UTC),
                }
            ],
            chunk_rows=[
                {
                    "note_id": "note-1",
                    "rank": 1,
                    "label": "S1",
                    "chunk_id": None,
                    "source_id": None,
                    "text": None,
                    "metadata": None,
                }
            ],
        ),
        Settings(),
    )

    result = service.list_all()

    assert len(result.notes) == 1
    assert result.notes[0].supporting_chunks == []


def test_list_all_caps_preview_and_sets_has_more() -> None:
    now = datetime.now(UTC)
    service = NoteRetrievalService(
        FakeConnection(
            note_rows=[
                {
                    "note_id": f"note-{index}",
                    "original_input": "Save this.",
                    "rewritten_note": f"Note {index}.",
                    "inferred_work": "Hamlet",
                    "source_id": "hamlet",
                    "created_at": now,
                }
                for index in range(3)
            ]
        ),
        Settings(note_query_top_k=2),
    )

    result = service.list_all()

    assert [note.label for note in result.notes] == ["N1", "N2"]
    assert result.has_more is True


def test_similarity_gate_requires_one_raw_score_threshold() -> None:
    settings = Settings(note_min_vector_score=0.35, note_min_trigram_score=0.18)
    service = NoteRetrievalService(FakeConnection(), settings)

    assert service.passes_similarity_gate({"vector_score": 0.34, "trigram_score": 0.17}) is False
    assert service.passes_similarity_gate({"vector_score": 0.35}) is True
    assert service.passes_similarity_gate({"lexical_score": 0.01}) is True
    assert service.passes_similarity_gate({"trigram_score": 0.18}) is True


def test_relevance_filter_malformed_output_fails_closed() -> None:
    note = RetrievedNote(
        label="N1",
        note_id="note-1",
        rewritten_note="Hamlet begins with uncertainty.",
        original_input="Save this.",
        inferred_work="Hamlet",
        created_at=datetime.now(UTC),
        combined_score=1.0,
        reason="test",
    )
    service = NoteRelevanceService(Settings(), model=FakeMalformedModel())

    assert service.filter("What happens in Hamlet?", [note]) == []


class FakeMalformedModel:
    def invoke(self, _prompt):
        return {"unexpected": ["note-1"]}


class FakeRows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, note_rows=None, chunk_rows=None) -> None:
        self.note_rows = note_rows or []
        self.chunk_rows = chunk_rows or []

    def execute(self, query, params=None):
        if "FROM note_chunks" in query:
            return FakeRows(self.chunk_rows)
        limit = params[-1] if params else len(self.note_rows)
        return FakeRows(self.note_rows[:limit])
