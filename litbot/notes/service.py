import json
import uuid
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from psycopg import Connection
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ValidationError

from litbot.config import Settings, get_settings
from litbot.generation.citations import validate_and_format_labels
from litbot.langchain import embed_query, make_chat_model
from litbot.models import ChatResponse, NoteProcessingPayload, RetrievedChunk
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


class NoteService:
    """Ground, rewrite, embed, and transactionally store global reading notes."""

    def __init__(
        self,
        conn: Connection,
        settings: Settings | None = None,
        retrieval_service: RetrievalService | None = None,
        model: Any | None = None,
    ) -> None:
        self.conn = conn
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or RetrievalService(conn, self.settings)
        self.model = model or make_chat_model(self.settings).with_structured_output(
            NoteProcessingPayload
        )

    def process(
        self,
        *,
        original_input: str,
        note_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        trace_id: str,
        intent_confidence: float | None = None,
    ) -> ChatResponse:
        filters = dict(filters or {})
        chunks = self.retrieval_service.retrieve(note_text, filters=filters, top_k=top_k)
        logger.info("note_retrieval_completed", trace_id=trace_id, chunk_count=len(chunks))
        if not chunks:
            return self._not_saved_response(
                original_input=original_input,
                note_text=note_text,
                chunks=[],
                trace_id=trace_id,
                reason="No relevant chunks were retrieved.",
                intent_confidence=intent_confidence,
            )

        payload = self._rewrite_note(note_text, original_input, filters, chunks)
        rewritten_note = payload.rewritten_note.strip()
        selected_chunks, invalid_chunk_ids = _selected_chunks(payload.selected_chunk_ids, chunks)
        inferred_work = _infer_work(payload, filters, selected_chunks)

        rejection_reason = _clean_reason(payload.rejection_reason)
        if invalid_chunk_ids:
            return self._not_saved_response(
                original_input=original_input,
                note_text=note_text,
                chunks=chunks,
                trace_id=trace_id,
                reason=(
                    "The note rewrite selected chunks that were not retrieved: "
                    + ", ".join(invalid_chunk_ids)
                ),
                intent_confidence=intent_confidence,
            )
        if not payload.should_save:
            return self._not_saved_response(
                original_input=original_input,
                note_text=note_text,
                chunks=chunks,
                trace_id=trace_id,
                reason=rejection_reason or DEFAULT_REJECTION_REASON,
                intent_confidence=intent_confidence,
            )
        if not rewritten_note:
            return self._not_saved_response(
                original_input=original_input,
                note_text=note_text,
                chunks=chunks,
                trace_id=trace_id,
                reason=rejection_reason or "The rewritten note was blank.",
                intent_confidence=intent_confidence,
            )
        if not selected_chunks:
            return self._not_saved_response(
                original_input=original_input,
                note_text=note_text,
                chunks=chunks,
                trace_id=trace_id,
                reason=rejection_reason or DEFAULT_REJECTION_REASON,
                intent_confidence=intent_confidence,
            )
        if not inferred_work:
            return self._not_saved_response(
                original_input=original_input,
                note_text=note_text,
                chunks=chunks,
                trace_id=trace_id,
                reason=rejection_reason or "The note was not tied to a corpus work.",
                intent_confidence=intent_confidence,
            )

        note_id = str(uuid.uuid4())
        embedding = embed_query(rewritten_note, self.settings)
        self._insert_note(
            note_id=note_id,
            original_input=original_input,
            rewritten_note=rewritten_note,
            inferred_work=inferred_work,
            selected_chunks=selected_chunks,
            embedding=embedding,
            trace_id=trace_id,
        )
        selected_chunk_ids = [chunk.chunk_id for chunk in selected_chunks]
        logger.info(
            "note_saved",
            trace_id=trace_id,
            note_id=note_id,
            inferred_work=inferred_work,
            selected_chunks=selected_chunk_ids,
        )
        return ChatResponse(
            answer=f"Saved note for {inferred_work}:\n{rewritten_note}",
            citations=validate_and_format_labels(
                {chunk.label for chunk in selected_chunks},
                chunks,
            ),
            retrieved_chunks=chunks,
            prompt_version=self.settings.note_prompt_version,
            trace_id=trace_id,
            citation_map=[item.model_dump() for item in payload.citation_map],
            intent="note",
            intent_confidence=intent_confidence,
            note_status="saved",
            note_id=note_id,
            note=rewritten_note,
            original_note=original_input,
            note_work=inferred_work,
            note_chunk_ids=selected_chunk_ids,
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

    def _insert_note(
        self,
        *,
        note_id: str,
        original_input: str,
        rewritten_note: str,
        inferred_work: str,
        selected_chunks: list[RetrievedChunk],
        embedding: list[float],
        trace_id: str,
    ) -> None:
        source_id = _shared_source_id(selected_chunks)
        work_metadata = _work_metadata(inferred_work, selected_chunks)
        with self.conn.transaction():
            self.conn.execute(
                """
                INSERT INTO notes
                    (note_id, original_input, rewritten_note, inferred_work, source_id,
                     work_metadata, embedding, model, prompt_version, trace_id, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, 'saved')
                """,
                [
                    note_id,
                    original_input,
                    rewritten_note,
                    inferred_work,
                    source_id,
                    Jsonb(work_metadata),
                    _vector_literal(embedding),
                    self.settings.llm_model,
                    self.settings.note_prompt_version,
                    trace_id,
                ],
            )
            with self.conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO note_chunks (note_id, chunk_id, rank, label)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (note_id, chunk.chunk_id, rank, chunk.label)
                        for rank, chunk in enumerate(selected_chunks, start=1)
                    ],
                )

    def _not_saved_response(
        self,
        *,
        original_input: str,
        note_text: str,
        chunks: list[RetrievedChunk],
        trace_id: str,
        reason: str,
        intent_confidence: float | None,
    ) -> ChatResponse:
        logger.info(
            "note_not_saved",
            trace_id=trace_id,
            rejection_reason=reason,
            chunk_count=len(chunks),
        )
        return ChatResponse(
            answer=f"I did not save that note: {reason}",
            citations=[],
            retrieved_chunks=chunks,
            prompt_version=self.settings.note_prompt_version,
            trace_id=trace_id,
            unsupported=[reason],
            intent="note",
            intent_confidence=intent_confidence,
            note_status="not_saved",
            note=note_text,
            original_note=original_input,
            note_chunk_ids=[],
            note_rejection_reason=reason,
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


def _clean_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    stripped = reason.strip()
    return stripped or None


def _shared_source_id(chunks: list[RetrievedChunk]) -> str | None:
    source_ids = {chunk.source_id for chunk in chunks}
    if len(source_ids) == 1:
        return chunks[0].source_id
    return None


def _work_metadata(inferred_work: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
    return {
        "work": inferred_work,
        "sources": [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
    }


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"
