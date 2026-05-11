import hashlib
import re

from litbot.models import DocumentMetadata, TextChunk

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PARAGRAPH_RE = re.compile(r"\n\s*\n")


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

    units = _split_units(text, metadata.genre)
    chunks: list[TextChunk] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > target_tokens:
            chunks.append(_make_chunk(chunks, current, metadata))
            current = _overlap_units(current, overlap_tokens)
            current_tokens = estimate_tokens("\n\n".join(current))
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(_make_chunk(chunks, current, metadata))
    return chunks


def _split_units(text: str, genre: str | None) -> list[str]:
    raw_units = [unit.strip() for unit in PARAGRAPH_RE.split(text) if unit.strip()]
    if genre and genre.lower() in {"poetry", "poem"}:
        stanza_units: list[str] = []
        for unit in raw_units:
            lines = [line for line in unit.splitlines() if line.strip()]
            for start in range(0, len(lines), 8):
                stanza_units.append("\n".join(lines[start : start + 8]))
        return stanza_units
    return raw_units


def _overlap_units(units: list[str], overlap_tokens: int) -> list[str]:
    selected: list[str] = []
    total = 0
    for unit in reversed(units):
        total += estimate_tokens(unit)
        selected.insert(0, unit)
        if total >= overlap_tokens:
            break
    return selected


def _make_chunk(
    existing: list[TextChunk],
    units: list[str],
    metadata: DocumentMetadata,
) -> TextChunk:
    text = "\n\n".join(units).strip()
    chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunk_index = len(existing)
    chunk_id = f"{metadata.source_id}:{chunk_index:05d}:{chunk_hash[:10]}"
    chunk_metadata = dict(metadata.metadata)
    chunk_metadata.update(
        {"title": metadata.title, "author": metadata.author, "license": metadata.license}
    )
    return TextChunk(
        chunk_id=chunk_id,
        source_id=metadata.source_id,
        chunk_index=chunk_index,
        text=text,
        token_count=estimate_tokens(text),
        chunk_hash=chunk_hash,
        metadata=chunk_metadata,
    )
