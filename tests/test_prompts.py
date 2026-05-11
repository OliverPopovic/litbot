import json

from litbot.generation.prompts import build_messages
from litbot.models import RetrievedChunk


def test_build_messages_includes_structured_sources() -> None:
    chunk = RetrievedChunk(
        label="S1",
        chunk_id="chunk-1",
        source_id="source-1",
        text="A passage.",
        metadata={"title": "Work"},
        combined_score=0.8,
        reason="test",
    )

    messages = build_messages("What happens?", [chunk])
    payload = json.loads(messages[-1]["content"])

    assert messages[0]["role"] == "system"
    assert payload["question"] == "What happens?"
    assert payload["retrieved_sources"][0]["label"] == "S1"
    assert payload["retrieved_sources"][0]["chunk_text"] == "A passage."
