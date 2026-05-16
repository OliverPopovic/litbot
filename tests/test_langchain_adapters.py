from litbot.langchain import document_to_retrieved_chunk


def test_document_to_retrieved_chunk_round_trips_scores_and_text() -> None:
    row = {
        "chunk_id": "chunk-1",
        "source_id": "source-1",
        "text": "Evidence text.",
        "metadata": {"title": "Work"},
    }

    chunk = document_to_retrieved_chunk(
        row,
        label="S1",
        vector_score=0.9,
        lexical_score=0.2,
        combined_score=0.725,
        reason="hybrid vector + lexical match",
    )

    assert chunk.label == "S1"
    assert chunk.chunk_id == "chunk-1"
    assert chunk.source_id == "source-1"
    assert chunk.text == "Evidence text."
    assert chunk.combined_score == 0.725
    assert chunk.reason == "hybrid vector + lexical match"
