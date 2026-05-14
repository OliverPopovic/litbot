from langchain_core.documents import Document

from litbot.ingestion.chunking import chunk_document
from litbot.langchain import (
    chunk_to_document,
    document_to_retrieved_chunk,
    langchain_connection_string,
)
from litbot.models import DocumentMetadata


def test_chunk_to_document_preserves_citation_metadata() -> None:
    metadata = DocumentMetadata(
        source_id="test-work",
        title="Test Work",
        author="A. Writer",
        publication_year=1818,
        genre="novel",
        language="en",
        license="Public domain",
        uri="https://example.test/work",
        metadata={"work": "Test Work", "chapter": 1},
    )
    chunk = chunk_document("First paragraph.", metadata)[0]

    document = chunk_to_document(chunk)

    assert document.id == chunk.chunk_id
    assert document.page_content == chunk.text
    assert document.metadata["chunk_id"] == chunk.chunk_id
    assert document.metadata["source_id"] == "test-work"
    assert document.metadata["work"] == "Test Work"
    assert document.metadata["chapter"] == 1
    assert document.metadata["token_count"] == chunk.token_count
    assert document.metadata["genre"] == "novel"


def test_document_to_retrieved_chunk_round_trips_scores_and_text() -> None:
    document = Document(
        id="chunk-1",
        page_content="Evidence text.",
        metadata={"chunk_id": "chunk-1", "source_id": "source-1", "title": "Work"},
    )

    chunk = document_to_retrieved_chunk(
        document,
        label="S1",
        vector_score=0.9,
        lexical_score=0.2,
        combined_score=0.725,
    )

    assert chunk.label == "S1"
    assert chunk.chunk_id == "chunk-1"
    assert chunk.source_id == "source-1"
    assert chunk.text == "Evidence text."
    assert chunk.combined_score == 0.725


def test_langchain_connection_string_uses_psycopg_driver() -> None:
    assert (
        langchain_connection_string("postgresql://user:pass@localhost/db")
        == "postgresql+psycopg://user:pass@localhost/db"
    )
