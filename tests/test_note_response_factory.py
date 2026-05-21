from datetime import UTC, datetime

from litbot.config import Settings
from litbot.models import RetrievedChunk, RetrievedNote
from litbot.notes.grounding import GroundedNote
from litbot.notes.responses import NoteResponseFactory
from litbot.notes.retrieval import NoteRetrievalResult


def test_saved_response_full_json_contract() -> None:
    chunk = _chunk()
    response = NoteResponseFactory(Settings(note_prompt_version="note-v1")).saved(
        note_id="note-1",
        grounded=GroundedNote(
            original_input="Save this.",
            note_text="Hamlet starts uncertain.",
            rewritten_note="Hamlet opens with watchful uncertainty.",
            inferred_work="Hamlet",
            selected_chunks=[chunk],
            retrieved_chunks=[chunk],
            citation_map=[{"claim": "watchful uncertainty", "sources": ["S1"]}],
            embedding=[0.1, 0.2],
        ),
        original_input="Save this.",
        trace_id="trace-1",
        intent_confidence=0.91,
    )

    _assert_full_json_contract(
        response,
        {
            "answer": "Saved note for Hamlet:\nHamlet opens with watchful uncertainty.",
            "citations": [
                {
                    "label": "S1",
                    "source_id": "hamlet",
                    "chunk_id": "hamlet:1",
                    "reference": "William Shakespeare, Hamlet",
                }
            ],
            "retrieved_chunks": [_chunk_json()],
            "prompt_version": "note-v1",
            "trace_id": "trace-1",
            "citation_map": [{"claim": "watchful uncertainty", "sources": ["S1"]}],
            "unsupported": [],
            "intent": "note",
            "intent_confidence": 0.91,
            "retrieved_notes": [],
            "note_query_status": None,
            "note_query_has_more": None,
            "note_status": "saved",
            "note_id": "note-1",
            "note": "Hamlet opens with watchful uncertainty.",
            "original_note": "Save this.",
            "note_work": "Hamlet",
            "note_chunk_ids": ["hamlet:1"],
            "note_rejection_reason": None,
            "note_operation": None,
            "note_operation_status": None,
            "pending_note_action_id": None,
            "target_note_ids": [],
        },
    )


def test_not_saved_response_full_json_contract() -> None:
    response = NoteResponseFactory(Settings(note_prompt_version="note-v1")).not_saved(
        original_input="Save this.",
        note_text="Save this.",
        chunks=[],
        trace_id="trace-1",
        reason="No relevant chunks were retrieved.",
        intent_confidence=0.7,
    )

    _assert_full_json_contract(
        response,
        {
            "answer": "I did not save that note: No relevant chunks were retrieved.",
            "citations": [],
            "retrieved_chunks": [],
            "prompt_version": "note-v1",
            "trace_id": "trace-1",
            "citation_map": [],
            "unsupported": ["No relevant chunks were retrieved."],
            "intent": "note",
            "intent_confidence": 0.7,
            "retrieved_notes": [],
            "note_query_status": None,
            "note_query_has_more": None,
            "note_status": "not_saved",
            "note_id": None,
            "note": "Save this.",
            "original_note": "Save this.",
            "note_work": None,
            "note_chunk_ids": [],
            "note_rejection_reason": "No relevant chunks were retrieved.",
            "note_operation": None,
            "note_operation_status": None,
            "pending_note_action_id": None,
            "target_note_ids": [],
        },
    )


def test_note_query_response_full_json_contract() -> None:
    note = RetrievedNote(
        label="N1",
        note_id="note-1",
        rewritten_note="Hamlet begins in uncertainty.",
        original_input="Save this.",
        inferred_work="Hamlet",
        matched_work="Hamlet",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        supporting_chunks=[_chunk()],
        combined_score=1.0,
        reason="test",
    )
    response = NoteResponseFactory(Settings(note_prompt_version="note-v1")).note_query(
        result=NoteRetrievalResult(notes=[note], has_more=False, match_strategy="test"),
        trace_id="trace-1",
        intent_confidence=0.9,
    )

    _assert_full_json_contract(
        response,
        {
            "answer": "I found these saved notes:\n- [N1] Hamlet: Hamlet begins in uncertainty.",
            "citations": [],
            "retrieved_chunks": [_chunk_json()],
            "prompt_version": "note-v1",
            "trace_id": "trace-1",
            "citation_map": [],
            "unsupported": [],
            "intent": "note_query",
            "intent_confidence": 0.9,
            "retrieved_notes": [
                {
                    "label": "N1",
                    "note_id": "note-1",
                    "rewritten_note": "Hamlet begins in uncertainty.",
                    "original_input": "Save this.",
                    "inferred_work": "Hamlet",
                    "matched_work": "Hamlet",
                    "source_id": None,
                    "created_at": "2026-01-01T00:00:00Z",
                    "supporting_chunks": [_chunk_json()],
                    "combined_score": 1.0,
                    "reason": "test",
                    "vector_score": None,
                    "lexical_score": None,
                    "trigram_score": None,
                }
            ],
            "note_query_status": "found",
            "note_query_has_more": False,
            "note_status": None,
            "note_id": None,
            "note": None,
            "original_note": None,
            "note_work": None,
            "note_chunk_ids": None,
            "note_rejection_reason": None,
            "note_operation": None,
            "note_operation_status": None,
            "pending_note_action_id": None,
            "target_note_ids": [],
        },
    )


def test_operation_response_full_json_contract() -> None:
    response = NoteResponseFactory(Settings(note_prompt_version="note-v1")).operation(
        answer="Please confirm deleting this note.",
        trace_id="trace-1",
        intent="note_delete",
        intent_confidence=0.9,
        operation="delete",
        status="pending_confirmation",
        target_note_ids=["note-1"],
        pending_note_action_id="action-1",
    )

    _assert_full_json_contract(
        response,
        {
            "answer": "Please confirm deleting this note.",
            "citations": [],
            "retrieved_chunks": [],
            "prompt_version": "note-v1",
            "trace_id": "trace-1",
            "citation_map": [],
            "unsupported": [],
            "intent": "note_delete",
            "intent_confidence": 0.9,
            "retrieved_notes": [],
            "note_query_status": None,
            "note_query_has_more": None,
            "note_status": None,
            "note_id": None,
            "note": None,
            "original_note": None,
            "note_work": None,
            "note_chunk_ids": None,
            "note_rejection_reason": None,
            "note_operation": "delete",
            "note_operation_status": "pending_confirmation",
            "pending_note_action_id": "action-1",
            "target_note_ids": ["note-1"],
        },
    )


def _assert_full_json_contract(response, expected_without_created_at: dict) -> None:
    actual = response.model_dump(mode="json")
    assert set(actual) == {*expected_without_created_at, "created_at"}
    assert actual["created_at"].endswith("Z")
    actual_without_created_at = {key: value for key, value in actual.items() if key != "created_at"}
    assert actual_without_created_at == expected_without_created_at


def _chunk_json() -> dict:
    return {
        "label": "S1",
        "chunk_id": "hamlet:1",
        "source_id": "hamlet",
        "text": "Barnardo asks who is there.",
        "metadata": {"work": "Hamlet", "title": "Hamlet", "author": "William Shakespeare"},
        "combined_score": 1.0,
        "reason": "test",
        "vector_score": None,
        "lexical_score": None,
        "trigram_score": None,
    }


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
