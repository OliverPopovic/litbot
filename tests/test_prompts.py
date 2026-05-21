import json

from litbot.generation.prompts import build_prompt_value, build_user_payload
from litbot.models import RetrievedChunk


def test_build_user_payload_includes_structured_sources() -> None:
    chunk = RetrievedChunk(
        label="S1",
        chunk_id="chunk-1",
        source_id="source-1",
        text="A passage.",
        metadata={
            "work": "Work",
            "title": "Work",
            "author": "Author",
            "uri": "https://example.test/work",
            "license": "Public domain",
        },
        combined_score=0.8,
        reason="test",
    )

    payload = json.loads(build_user_payload("What happens?", [chunk]))

    assert payload["question"] == "What happens?"
    assert payload["retrieved_sources"][0]["label"] == "S1"
    assert payload["retrieved_sources"][0]["chunk_text"] == "A passage."
    assert payload["retrieved_sources"][0]["metadata"] == {
        "work": "Work",
        "title": "Work",
        "author": "Author",
        "source_id": "source-1",
    }


def test_build_prompt_value_uses_langchain_template() -> None:
    chunk = RetrievedChunk(
        label="S1",
        chunk_id="chunk-1",
        source_id="source-1",
        text="A passage.",
        metadata={"title": "Work"},
        combined_score=0.8,
        reason="test",
    )

    prompt_value = build_prompt_value("What happens?", [chunk])
    messages = prompt_value.to_messages()

    assert messages[0].type == "system"
    assert "literary research assistant" in messages[0].content
    assert "Use concise, student-friendly prose" in messages[1].content
    assert messages[2].type == "human"
    assert '"label": "S1"' in messages[2].content
