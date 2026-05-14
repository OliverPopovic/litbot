from litbot.generation.service import GenerationPayload, GenerationService
from litbot.models import RetrievedChunk


class FakeStructuredModel:
    def invoke(self, _prompt):
        return GenerationPayload(
            answer="Victor is horrified by the creature [S1].",
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
