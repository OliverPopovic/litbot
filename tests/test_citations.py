import pytest

from litbot.generation.citations import (
    format_reference,
    validate_and_format_citations,
    validate_and_format_labels,
)
from litbot.models import RetrievedChunk


def make_chunk(label: str = "S1") -> RetrievedChunk:
    return RetrievedChunk(
        label=label,
        chunk_id="chunk-1",
        source_id="source-1",
        text="Evidence text",
        metadata={"author": "Mary Shelley", "title": "Frankenstein", "chapter": 5},
        combined_score=0.9,
        reason="test",
    )


def test_validate_and_format_citations_from_answer() -> None:
    citations = validate_and_format_citations("Victor is horrified [S1].", [make_chunk()])

    assert len(citations) == 1
    assert citations[0].reference == "Mary Shelley, Frankenstein, ch. 5"


def test_validate_and_format_citations_rejects_unretrieved_label() -> None:
    with pytest.raises(ValueError, match="Invalid citation labels"):
        validate_and_format_citations("Unsupported citation [S2].", [make_chunk()])


def test_validate_and_format_labels_from_citation_map_sources() -> None:
    citations = validate_and_format_labels({"S1"}, [make_chunk()])

    assert len(citations) == 1
    assert citations[0].label == "S1"


def test_format_reference_falls_back_to_source_id() -> None:
    chunk = make_chunk()
    chunk.metadata = {}
    assert format_reference(chunk) == "source-1"
