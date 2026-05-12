import json
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup

from litbot.models import DocumentMetadata, ParsedDocument


def parse_document(path: Path, metadata_path: Path | None = None) -> ParsedDocument:
    """Parse a TXT/MD/HTML/PDF document with sidecar JSON metadata."""

    metadata = _load_metadata(path, metadata_path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
    elif suffix in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()
        text = soup.get_text("\n")
    elif suffix == ".pdf":
        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"\n\n[page {index}]\n{page_text}")
        text = "\n".join(pages)
    else:
        raise ValueError(f"Unsupported document type: {path.suffix}")
    return ParsedDocument(metadata=metadata, text=_normalize_text(text))


def _load_metadata(path: Path, metadata_path: Path | None) -> DocumentMetadata:
    candidate = metadata_path or path.with_suffix(path.suffix + ".json")
    if not candidate.exists():
        raise FileNotFoundError(f"Metadata sidecar not found: {candidate}")
    data = json.loads(candidate.read_text(encoding="utf-8"))
    _validate_metadata(data, candidate)
    return DocumentMetadata(**data)


def _validate_metadata(data: dict[str, object], path: Path) -> None:
    required = {
        "source_id",
        "title",
        "author",
        "publication_year",
        "genre",
        "language",
        "license",
        "uri",
        "version",
    }
    missing = sorted(field for field in required if not data.get(field))
    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("work"):
        missing.append("metadata.work")
    if missing:
        raise ValueError(f"Metadata sidecar {path} is missing required fields: {missing}")


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized: list[str] = []
    blank_seen = False
    for line in lines:
        if not line.strip():
            if not blank_seen:
                normalized.append("")
            blank_seen = True
        else:
            normalized.append(line)
            blank_seen = False
    return "\n".join(normalized).strip()
