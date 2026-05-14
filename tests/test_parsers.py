import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from litbot.ingestion.parsers import parse_document
from litbot.models import DocumentMetadata


def test_document_metadata_requires_work_metadata() -> None:
    with pytest.raises(ValidationError, match="metadata.work"):
        DocumentMetadata(
            source_id="source",
            title="Source",
            author="Author",
            publication_year=1818,
            genre="novel",
            language="en",
            license="Public domain",
            uri="https://example.test/source",
            version="1",
            metadata={},
        )


def test_parse_document_requires_corpus_metadata_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("A passage.", encoding="utf-8")
    source.with_suffix(".txt.json").write_text(
        json.dumps(
            {
                "source_id": "source",
                "title": "Source",
                "license": "Public domain",
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Metadata sidecar .*source\.txt\.json.*metadata.work"):
        parse_document(source)


@pytest.mark.parametrize(
    "path",
    [
        Path("corpus/public_domain/pride-prejudice-1813-ch1.txt"),
        Path("corpus/public_domain/hamlet-act1-sc1.txt"),
        Path("corpus/public_domain/moby-dick-1851-ch1.txt"),
    ],
)
def test_public_domain_corpus_examples_parse(path: Path) -> None:
    parsed = parse_document(path)

    assert parsed.text
    assert parsed.metadata.license == "Public domain"
    assert parsed.metadata.metadata["work"]
