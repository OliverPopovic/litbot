from litbot.ingestion.chunking import chunk_document, estimate_tokens
from litbot.models import DocumentMetadata


def test_estimate_tokens_counts_words_and_punctuation() -> None:
    assert estimate_tokens("Hello, world!") == 4


def test_chunk_document_creates_stable_metadata_and_ids() -> None:
    metadata = DocumentMetadata(
        source_id="test-work",
        title="Test Work",
        author="A. Writer",
        publication_year=1818,
        license="Public domain",
        genre="novel",
        language="en",
        uri="https://example.test/work",
        metadata={"work": "Test Work", "chapter": 1},
    )
    text = "First paragraph has evidence.\n\nSecond paragraph continues the idea."

    chunks = chunk_document(text, metadata, target_tokens=20, overlap_tokens=5)

    assert len(chunks) == 1
    assert chunks[0].chunk_id.startswith("test-work:00000:")
    assert chunks[0].metadata["title"] == "Test Work"
    assert chunks[0].metadata["chapter"] == 1


def test_chunk_document_preserves_poetry_line_structure() -> None:
    metadata = DocumentMetadata(
        source_id="test-poem",
        title="Test Poem",
        author="A. Poet",
        publication_year=1900,
        license="Public domain",
        genre="poetry",
        language="en",
        uri="https://example.test/poem",
        metadata={"work": "Test Poem"},
    )
    text = "\n".join(f"Line {index}" for index in range(1, 11))

    chunks = chunk_document(text, metadata, target_tokens=100, overlap_tokens=0)

    assert len(chunks) == 1
    assert "Line 1\nLine 2" in chunks[0].text
    assert chunks[0].metadata["work"] == "Test Poem"
