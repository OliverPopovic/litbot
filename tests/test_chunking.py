from litbot.ingestion.chunking import chunk_document, estimate_tokens
from litbot.models import DocumentMetadata


def test_estimate_tokens_counts_words_and_punctuation() -> None:
    assert estimate_tokens("Hello, world!") == 4


def test_chunk_document_creates_stable_metadata_and_ids() -> None:
    metadata = DocumentMetadata(
        source_id="test-work",
        title="Test Work",
        author="A. Writer",
        license="Public domain",
        genre="novel",
        metadata={"chapter": 1},
    )
    text = "First paragraph has evidence.\n\nSecond paragraph continues the idea."

    chunks = chunk_document(text, metadata, target_tokens=20, overlap_tokens=5)

    assert len(chunks) == 1
    assert chunks[0].chunk_id.startswith("test-work:00000:")
    assert chunks[0].metadata["title"] == "Test Work"
    assert chunks[0].metadata["chapter"] == 1
