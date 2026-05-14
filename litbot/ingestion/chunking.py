import hashlib
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from litbot.models import DocumentMetadata, TextChunk

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PARAGRAPH_RE = re.compile(r"\n\s*\n")
POETRY_LINES_PER_UNIT = 8


def estimate_tokens(text: str) -> int:
    """Fast token estimate suitable for chunk sizing and validation."""

    return len(TOKEN_RE.findall(text))


def chunk_document(
    text: str,
    metadata: DocumentMetadata,
    target_tokens: int = 550,
    overlap_tokens: int = 80,
) -> list[TextChunk]:
    """Split text into structure-aware chunks while preserving paragraph and poetry breaks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_tokens,
        chunk_overlap=overlap_tokens,
        length_function=estimate_tokens,
        separators=["\n\n", "\n", " ", ""],
    )
    prepared_text = _prepare_text_for_splitting(text, metadata.genre)
    documents = splitter.create_documents([prepared_text], metadatas=[_chunk_metadata(metadata)])
    return [
        _make_chunk(document.page_content, index, metadata, dict(document.metadata))
        for index, document in enumerate(documents)
        if document.page_content.strip()
    ]


def _prepare_text_for_splitting(text: str, genre: str | None) -> str:
    if not genre or genre.lower() not in {"poetry", "poem"}:
        return text

    # LangChain handles chunk size and overlap; this small pre-pass gives poems stanza-like
    # boundaries so line structure remains visible in retrieved evidence.
    stanza_units: list[str] = []
    raw_units = [unit.strip() for unit in PARAGRAPH_RE.split(text) if unit.strip()]
    for unit in raw_units:
        lines = [line for line in unit.splitlines() if line.strip()]
        for start in range(0, len(lines), POETRY_LINES_PER_UNIT):
            stanza_units.append("\n".join(lines[start : start + POETRY_LINES_PER_UNIT]))
    return "\n\n".join(stanza_units)


def _chunk_metadata(metadata: DocumentMetadata) -> dict[str, object]:
    chunk_metadata = dict(metadata.metadata)
    chunk_metadata.update(
        metadata.model_dump(
            exclude={
                "metadata",
            }
        )
    )
    return chunk_metadata


def _make_chunk(
    text: str,
    chunk_index: int,
    metadata: DocumentMetadata,
    chunk_metadata: dict[str, object],
) -> TextChunk:
    text = text.strip()
    chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunk_id = f"{metadata.source_id}:{chunk_index:05d}:{chunk_hash[:10]}"
    return TextChunk(
        chunk_id=chunk_id,
        source_id=metadata.source_id,
        chunk_index=chunk_index,
        text=text,
        token_count=estimate_tokens(text),
        chunk_hash=chunk_hash,
        metadata=chunk_metadata,
    )
