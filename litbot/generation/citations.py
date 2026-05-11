import re

from litbot.models import Citation, RetrievedChunk

CITATION_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def validate_and_format_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Validate generated citation labels and format metadata-backed references."""

    labels_in_answer: set[str] = set()
    for match in CITATION_RE.findall(answer):
        labels_in_answer.update(label.strip() for label in match.split(","))

    return validate_and_format_labels(labels_in_answer, chunks)


def validate_and_format_labels(labels: set[str], chunks: list[RetrievedChunk]) -> list[Citation]:
    """Validate citation labels and format metadata-backed references."""

    by_label = {chunk.label: chunk for chunk in chunks}
    invalid = labels - set(by_label)
    if invalid:
        raise ValueError(f"Invalid citation labels: {sorted(invalid)}")

    citations: list[Citation] = []
    for label in sorted(labels, key=lambda value: int(value[1:])):
        chunk = by_label[label]
        citations.append(
            Citation(
                label=label,
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
                reference=format_reference(chunk),
            )
        )
    return citations


def format_reference(chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata
    parts = []
    author = metadata.get("author")
    title = metadata.get("title")
    if author:
        parts.append(str(author))
    if title:
        parts.append(str(title))
    for key, label in [
        ("chapter", "ch."),
        ("act", "act"),
        ("scene", "sc."),
        ("page_start", "p."),
        ("line_start", "line"),
    ]:
        if metadata.get(key) is not None:
            parts.append(f"{label} {metadata[key]}")
    return ", ".join(parts) if parts else chunk.source_id
