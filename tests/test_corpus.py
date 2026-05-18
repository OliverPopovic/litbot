from litbot.corpus import CorpusWork, metadata_for_work, strip_gutenberg_boilerplate


def test_strip_gutenberg_boilerplate_removes_header_and_footer() -> None:
    text = """
Header line
*** START OF THE PROJECT GUTENBERG EBOOK TEST WORK ***

Chapter 1
Body text.

*** END OF THE PROJECT GUTENBERG EBOOK TEST WORK ***
Footer line
"""

    assert strip_gutenberg_boilerplate(text) == "Chapter 1\nBody text.\n"


def test_metadata_for_work_generates_ingestion_sidecar_shape() -> None:
    work = CorpusWork(
        source_id="test-work",
        filename="test-work.txt",
        title="Test Work",
        author="Test Author",
        translator="Test Translator",
        publication_year=1900,
        genre="novel",
        uri="https://example.test/book",
        url="https://example.test/book.txt",
        work="Test Work",
    )

    metadata = metadata_for_work(work)

    assert metadata["source_id"] == "test-work"
    assert metadata["translator"] == "Test Translator"
    assert metadata["license"] == "Public domain"
    assert metadata["metadata"] == {"work": "Test Work"}
