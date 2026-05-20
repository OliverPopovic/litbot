from datetime import UTC, datetime

from litbot.generation.service import GenerationPayload, GenerationService
from litbot.models import RetrievedChunk, RetrievedNote


class FakeStructuredModel:
    def invoke(self, _prompt):
        return GenerationPayload(
            answer="Victor is horrified by the creature [S1].",
            citation_map=[{"claim": "Victor is horrified", "sources": ["S1"]}],
            unsupported=[],
        )


class FakeNoteRefModel:
    def invoke(self, _prompt):
        return GenerationPayload(
            answer="Victor is horrified by the creature [S1]. My note says this too [N1].",
            citation_map=[{"claim": "Victor is horrified", "sources": ["S1"]}],
            unsupported=[],
        )


def test_generation_service_maps_structured_output_to_chat_response() -> None:
    chunk = RetrievedChunk(
        label="S1",
        chunk_id="chunk-1",
        source_id="source-1",
        text="Victor felt horror.",
        metadata={"author": "Mary Shelley", "title": "Frankenstein", "chapter": 5},
        combined_score=0.8,
        reason="test",
    )

    response = GenerationService(model=FakeStructuredModel()).answer(
        "How does Victor react?",
        [chunk],
        trace_id="trace-1",
    )

    assert response.answer == "Victor is horrified by the creature [S1]."
    assert response.trace_id == "trace-1"
    assert response.citations[0].reference == "Mary Shelley, Frankenstein, ch. 5"
    assert response.retrieved_chunks == [chunk]


def test_generation_service_strips_note_labels_from_primary_answer_and_appends_notes() -> None:
    chunk = RetrievedChunk(
        label="S1",
        chunk_id="chunk-1",
        source_id="source-1",
        text="Victor felt horror.",
        metadata={"author": "Mary Shelley", "title": "Frankenstein", "chapter": 5},
        combined_score=0.8,
        reason="test",
    )
    note = RetrievedNote(
        label="N1",
        note_id="note-1",
        rewritten_note="Victor's reaction emphasizes horror.",
        original_input="Save this.",
        inferred_work="Frankenstein",
        created_at=datetime.now(UTC),
        combined_score=1.0,
        reason="test",
    )

    response = GenerationService(model=FakeNoteRefModel()).answer(
        "How does Victor react?",
        [chunk],
        notes=[note],
        trace_id="trace-1",
    )

    primary_answer = response.answer.split("Relevant notes:")[0]
    assert "[N1]" not in primary_answer
    assert "[N1] Frankenstein: Victor's reaction emphasizes horror." in response.answer
    assert "Removed note references" in response.unsupported[0]
