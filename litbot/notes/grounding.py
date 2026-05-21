import json
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from litbot.config import Settings, get_settings
from litbot.langchain import embed_query, make_chat_model
from litbot.models import NoteProcessingPayload, RetrievedChunk
from litbot.retrieval.service import RetrievalService

logger = structlog.get_logger(__name__)

DEFAULT_REJECTION_REASON = "The note could not be grounded in the corpus."

NOTE_SYSTEM_PROMPT = """
You rewrite literary reading notes using only the provided corpus evidence. The rewritten note
must be concise, factual, and brief. Do not invent facts, quotations, authors, titles, scenes, or
interpretations that the sources do not support. If the note cannot be grounded in the retrieved
sources, set should_save to false.
""".strip()

NOTE_DEVELOPER_PROMPT = """
Return valid structured data with should_save, rewritten_note, inferred_work,
selected_chunk_ids, citation_map, and rejection_reason. Select only chunk_id values present in the
retrieved_sources payload. If no work filter was supplied, infer the literary work from source
metadata and evidence. A saved note must name one corpus work and at least one supporting chunk.
""".strip()

NOTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", NOTE_SYSTEM_PROMPT),
        ("system", NOTE_DEVELOPER_PROMPT),
        ("human", "{user_payload}"),
    ]
)


@dataclass
class GroundedNote:
    original_input: str
    note_text: str
    rewritten_note: str = ""
    inferred_work: str = ""
    selected_chunks: list[RetrievedChunk] | None = None
    retrieved_chunks: list[RetrievedChunk] | None = None
    citation_map: list[dict[str, Any]] | None = None
    embedding: list[float] | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        self.selected_chunks = self.selected_chunks or []
        self.retrieved_chunks = self.retrieved_chunks or []
        self.citation_map = self.citation_map or []
        self.embedding = self.embedding or []

    def to_payload(self) -> dict[str, Any]:
        return {
            "original_input": self.original_input,
            "note_text": self.note_text,
            "rewritten_note": self.rewritten_note,
            "inferred_work": self.inferred_work,
            "selected_chunks": [chunk.model_dump(mode="json") for chunk in self.selected_chunks],
            "citation_map": self.citation_map,
            "embedding": self.embedding,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GroundedNote":
        return cls(
            original_input=str(payload.get("original_input") or ""),
            note_text=str(payload.get("note_text") or ""),
            rewritten_note=str(payload.get("rewritten_note") or ""),
            inferred_work=str(payload.get("inferred_work") or ""),
            selected_chunks=[
                RetrievedChunk.model_validate(chunk)
                for chunk in payload.get("selected_chunks", [])
                if isinstance(chunk, dict)
            ],
            citation_map=[
                item for item in payload.get("citation_map", []) if isinstance(item, dict)
            ],
            embedding=[float(value) for value in payload.get("embedding", [])],
        )


class NoteGroundingService:
    """Retrieve, rewrite, validate, and embed a reading note."""

    def __init__(
        self,
        conn: Any,
        settings: Settings | None = None,
        retrieval_service: RetrievalService | None = None,
        model: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or RetrievalService(conn, self.settings)
        self.model = model or make_chat_model(self.settings).with_structured_output(
            NoteProcessingPayload
        )

    def prepare(
        self,
        *,
        original_input: str,
        note_text: str,
        filters: dict[str, Any],
        top_k: int | None,
        trace_id: str,
    ) -> GroundedNote:
        chunks = self.retrieval_service.retrieve(note_text, filters=filters, top_k=top_k)
        logger.info("note_retrieval_completed", trace_id=trace_id, chunk_count=len(chunks))
        if not chunks:
            return GroundedNote(
                original_input=original_input,
                note_text=note_text,
                retrieved_chunks=[],
                rejection_reason="No relevant chunks were retrieved.",
            )

        payload = self._rewrite_note(note_text, original_input, filters, chunks)
        rewritten_note = payload.rewritten_note.strip()
        selected_chunks, invalid_chunk_ids = _selected_chunks(payload.selected_chunk_ids, chunks)
        inferred_work = _infer_work(payload, filters, selected_chunks)
        rejection_reason = _clean_text(payload.rejection_reason)
        if invalid_chunk_ids:
            rejection_reason = (
                "The note rewrite selected chunks that were not retrieved: "
                + ", ".join(invalid_chunk_ids)
            )
        elif not payload.should_save:
            rejection_reason = rejection_reason or DEFAULT_REJECTION_REASON
        elif not rewritten_note:
            rejection_reason = rejection_reason or "The rewritten note was blank."
        elif not selected_chunks:
            rejection_reason = rejection_reason or DEFAULT_REJECTION_REASON
        elif not inferred_work:
            rejection_reason = rejection_reason or "The note was not tied to a corpus work."

        if rejection_reason:
            return GroundedNote(
                original_input=original_input,
                note_text=note_text,
                retrieved_chunks=chunks,
                rejection_reason=rejection_reason,
            )

        return GroundedNote(
            original_input=original_input,
            note_text=note_text,
            rewritten_note=rewritten_note,
            inferred_work=str(inferred_work),
            selected_chunks=selected_chunks,
            retrieved_chunks=chunks,
            citation_map=[item.model_dump() for item in payload.citation_map],
            embedding=embed_query(rewritten_note, self.settings),
        )

    def _rewrite_note(
        self,
        note_text: str,
        original_input: str,
        filters: dict[str, Any],
        chunks: list[RetrievedChunk],
    ) -> NoteProcessingPayload:
        user_payload = {
            "note_text": note_text,
            "original_input": original_input,
            "filters": filters,
            "retrieved_sources": [
                {
                    "label": chunk.label,
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "metadata": chunk.metadata,
                    "chunk_text": chunk.text,
                }
                for chunk in chunks
            ],
        }
        try:
            payload = self.model.invoke(
                NOTE_PROMPT.invoke(
                    {"user_payload": json.dumps(user_payload, ensure_ascii=False)}
                )
            )
            return _note_payload_from_model(payload)
        except Exception as exc:
            logger.warning("note_rewrite_failed", error=str(exc))
            return NoteProcessingPayload(
                should_save=False,
                rejection_reason=DEFAULT_REJECTION_REASON,
            )


def _note_payload_from_model(payload: object) -> NoteProcessingPayload:
    if isinstance(payload, NoteProcessingPayload):
        return payload
    if isinstance(payload, BaseModel):
        payload = payload.model_dump()
    if not isinstance(payload, dict):
        return NoteProcessingPayload(
            should_save=False,
            rejection_reason=DEFAULT_REJECTION_REASON,
        )
    try:
        return NoteProcessingPayload.model_validate(payload)
    except ValidationError:
        return NoteProcessingPayload(
            should_save=False,
            rejection_reason=DEFAULT_REJECTION_REASON,
        )


def _selected_chunks(
    selected_chunk_ids: list[str],
    chunks: list[RetrievedChunk],
) -> tuple[list[RetrievedChunk], list[str]]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    selected: list[RetrievedChunk] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for chunk_id in selected_chunk_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = by_id.get(chunk_id)
        if chunk is None:
            invalid.append(chunk_id)
        else:
            selected.append(chunk)
    return selected, invalid


def _infer_work(
    payload: NoteProcessingPayload,
    filters: dict[str, Any],
    chunks: list[RetrievedChunk],
) -> str | None:
    for candidate in [payload.inferred_work, filters.get("work")]:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    for chunk in chunks:
        work = chunk.metadata.get("work") or chunk.metadata.get("title")
        if isinstance(work, str) and work.strip():
            return work.strip()
    return None


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
