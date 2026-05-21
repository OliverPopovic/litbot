import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from psycopg import Connection
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ValidationError

from litbot.config import Settings, get_settings
from litbot.generation.citations import validate_and_format_labels
from litbot.langchain import embed_query, make_chat_model
from litbot.models import ChatResponse, NoteContext, NoteProcessingPayload, RetrievedChunk
from litbot.retrieval.service import RetrievalService

logger = structlog.get_logger(__name__)

DEFAULT_REJECTION_REASON = "The note could not be grounded in the corpus."
PENDING_ACTION_TTL = timedelta(minutes=10)

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
        grounded = self._prepare_grounded_note(
            original_input=original_input,
            note_text=note_text,
            filters=filters,
            top_k=top_k,
            trace_id=trace_id,
        )
        if grounded.rejection_reason:
            return self._not_saved_response(
                original_input=original_input,
                note_text=note_text,
                chunks=grounded.retrieved_chunks,
                trace_id=trace_id,
                reason=grounded.rejection_reason,
                intent_confidence=intent_confidence,
            )

        note_id = str(uuid.uuid4())
        self._insert_note(
            note_id=note_id,
            original_input=grounded.original_input,
            rewritten_note=grounded.rewritten_note,
            inferred_work=grounded.inferred_work,
            selected_chunks=grounded.selected_chunks,
            embedding=grounded.embedding,
            trace_id=trace_id,
        )
        selected_chunk_ids = [chunk.chunk_id for chunk in grounded.selected_chunks]
        logger.info(
            "note_saved",
            trace_id=trace_id,
            note_id=note_id,
            inferred_work=grounded.inferred_work,
            selected_chunks=selected_chunk_ids,
        )
        return ChatResponse(
            answer=f"Saved note for {grounded.inferred_work}:\n{grounded.rewritten_note}",
            citations=validate_and_format_labels(
                {chunk.label for chunk in grounded.selected_chunks},
                grounded.retrieved_chunks,
            ),
            retrieved_chunks=grounded.retrieved_chunks,
            prompt_version=self.settings.note_prompt_version,
            trace_id=trace_id,
            citation_map=grounded.citation_map,
            intent="note",
            intent_confidence=intent_confidence,
            note_status="saved",
            note_id=note_id,
            note=grounded.rewritten_note,
            original_note=original_input,
            note_work=grounded.inferred_work,
            note_chunk_ids=selected_chunk_ids,
        )

    def preview_edit(
        self,
        *,
        original_input: str,
        note_text: str | None,
        target_reference: str | None,
        note_context: NoteContext | None,
        filters: dict[str, Any] | None,
        top_k: int | None,
        trace_id: str,
        intent_confidence: float | None,
    ) -> ChatResponse:
        target_id, error_status, reason = self._resolve_target(target_reference, note_context)
        if error_status:
            return self._operation_response(
                answer=reason,
                trace_id=trace_id,
                intent="note_edit",
                intent_confidence=intent_confidence,
                operation="edit",
                status=error_status,
                target_note_ids=[],
                unsupported=[reason],
            )
        target = self._fetch_note(str(target_id))
        if target is None:
            reason = "I could not find that saved note."
            return self._operation_response(
                answer=reason,
                trace_id=trace_id,
                intent="note_edit",
                intent_confidence=intent_confidence,
                operation="edit",
                status="not_found",
                target_note_ids=[str(target_id)],
                unsupported=[reason],
            )
        if not note_text or not note_text.strip():
            reason = "I need the replacement note text before I can prepare an edit."
            return self._operation_response(
                answer=reason,
                trace_id=trace_id,
                intent="note_edit",
                intent_confidence=intent_confidence,
                operation="edit",
                status="rejected",
                target_note_ids=[str(target_id)],
                unsupported=[reason],
            )

        grounded = self._prepare_grounded_note(
            original_input=original_input,
            note_text=note_text,
            filters=dict(filters or {}),
            top_k=top_k,
            trace_id=trace_id,
        )
        if grounded.rejection_reason:
            return self._operation_response(
                answer=f"I could not prepare that edit: {grounded.rejection_reason}",
                trace_id=trace_id,
                intent="note_edit",
                intent_confidence=intent_confidence,
                operation="edit",
                status="rejected",
                target_note_ids=[str(target_id)],
                retrieved_chunks=grounded.retrieved_chunks,
                unsupported=[grounded.rejection_reason],
            )

        action_id = self._create_pending_action(
            "edit",
            {
                "target_note_id": str(target_id),
                "original_input": original_input,
                "note_text": note_text,
                "previous_note": target["rewritten_note"],
                "grounded": grounded.to_payload(),
            },
        )
        answer = (
            "Please confirm this note edit.\n"
            f"Current note: {target['rewritten_note']}\n"
            f"Proposed note: {grounded.rewritten_note}"
        )
        return self._operation_response(
            answer=answer,
            trace_id=trace_id,
            intent="note_edit",
            intent_confidence=intent_confidence,
            operation="edit",
            status="pending_confirmation",
            target_note_ids=[str(target_id)],
            pending_note_action_id=action_id,
            retrieved_chunks=grounded.retrieved_chunks,
            note=grounded.rewritten_note,
            note_work=grounded.inferred_work,
            note_chunk_ids=[chunk.chunk_id for chunk in grounded.selected_chunks],
        )

    def preview_delete(
        self,
        *,
        target_reference: str | None,
        note_context: NoteContext | None,
        trace_id: str,
        intent_confidence: float | None,
    ) -> ChatResponse:
        target_id, error_status, reason = self._resolve_target(target_reference, note_context)
        if error_status:
            return self._operation_response(
                answer=reason,
                trace_id=trace_id,
                intent="note_delete",
                intent_confidence=intent_confidence,
                operation="delete",
                status=error_status,
                unsupported=[reason],
            )
        target = self._fetch_note(str(target_id))
        if target is None:
            reason = "I could not find that saved note."
            return self._operation_response(
                answer=reason,
                trace_id=trace_id,
                intent="note_delete",
                intent_confidence=intent_confidence,
                operation="delete",
                status="not_found",
                target_note_ids=[str(target_id)],
                unsupported=[reason],
            )
        action_id = self._create_pending_action(
            "delete",
            {
                "target_note_ids": [str(target_id)],
                "notes": [_note_summary(target)],
            },
        )
        answer = (
            "Please confirm deleting this note.\n"
            f"[{target['inferred_work']}] {target['rewritten_note']}"
        )
        return self._operation_response(
            answer=answer,
            trace_id=trace_id,
            intent="note_delete",
            intent_confidence=intent_confidence,
            operation="delete",
            status="pending_confirmation",
            target_note_ids=[str(target_id)],
            pending_note_action_id=action_id,
        )

    def preview_delete_all(
        self,
        *,
        trace_id: str,
        intent_confidence: float | None,
    ) -> ChatResponse:
        rows = self.conn.execute(
            """
            SELECT note_id, original_input, rewritten_note, inferred_work, source_id, created_at
            FROM notes
            ORDER BY created_at DESC
            """
        ).fetchall()
        notes = [dict(row) for row in rows]
        note_ids = [str(row["note_id"]) for row in notes]
        if not note_ids:
            return self._operation_response(
                answer="There are no saved notes to delete.",
                trace_id=trace_id,
                intent="note_delete_all",
                intent_confidence=intent_confidence,
                operation="delete_all",
                status="completed",
                target_note_ids=[],
            )
        action_id = self._create_pending_action(
            "delete_all",
            {
                "target_note_ids": note_ids,
                "notes": [_note_summary(note) for note in notes],
            },
        )
        answer = f"Please confirm deleting all saved notes. This will delete {len(note_ids)} notes."
        return self._operation_response(
            answer=answer,
            trace_id=trace_id,
            intent="note_delete_all",
            intent_confidence=intent_confidence,
            operation="delete_all",
            status="pending_confirmation",
            target_note_ids=note_ids,
            pending_note_action_id=action_id,
        )

    def confirm_pending_action(
        self,
        action_id: str | None,
        *,
        trace_id: str,
    ) -> ChatResponse:
        if not action_id:
            return self._operation_response(
                answer="No pending note action was supplied.",
                trace_id=trace_id,
                intent=None,
                intent_confidence=None,
                operation=None,
                status="rejected",
                unsupported=["No pending note action was supplied."],
            )
        with self.conn.transaction():
            row = self._fetch_pending_action_for_update(action_id)
            if row is None:
                return self._operation_response(
                    answer="I could not find that pending note action.",
                    trace_id=trace_id,
                    intent=None,
                    intent_confidence=None,
                    operation=None,
                    status="rejected",
                    unsupported=["Pending note action was not found."],
                )
            operation = str(row["operation"])
            payload = _json_payload(row["payload"])
            status_error = self._pending_status_error(row)
            if status_error:
                return self._operation_response(
                    answer=status_error,
                    trace_id=trace_id,
                    intent=_intent_for_operation(operation),
                    intent_confidence=None,
                    operation=_response_operation(operation),
                    status="rejected",
                    pending_note_action_id=action_id,
                    target_note_ids=_target_note_ids(payload),
                    unsupported=[status_error],
                )

            if operation == "edit":
                response = self._confirm_edit(action_id, payload, trace_id)
            elif operation in {"delete", "delete_all"}:
                response = self._confirm_delete(action_id, operation, payload, trace_id)
            else:
                reason = "The pending note action has an unsupported operation."
                self._consume_pending_action(action_id)
                response = self._operation_response(
                    answer=reason,
                    trace_id=trace_id,
                    intent=None,
                    intent_confidence=None,
                    operation=None,
                    status="rejected",
                    pending_note_action_id=action_id,
                    target_note_ids=_target_note_ids(payload),
                    unsupported=[reason],
                )
            return response

    def cancel_pending_action(
        self,
        action_id: str | None,
        *,
        trace_id: str,
    ) -> ChatResponse:
        if not action_id:
            return self._operation_response(
                answer="No pending note action was supplied.",
                trace_id=trace_id,
                intent=None,
                intent_confidence=None,
                operation=None,
                status="rejected",
                unsupported=["No pending note action was supplied."],
            )
        with self.conn.transaction():
            row = self._fetch_pending_action_for_update(action_id)
            if row is None:
                return self._operation_response(
                    answer="I could not find that pending note action.",
                    trace_id=trace_id,
                    intent=None,
                    intent_confidence=None,
                    operation=None,
                    status="rejected",
                    unsupported=["Pending note action was not found."],
                )
            operation = str(row["operation"])
            payload = _json_payload(row["payload"])
            status_error = self._pending_status_error(row)
            if status_error:
                return self._operation_response(
                    answer=status_error,
                    trace_id=trace_id,
                    intent=_intent_for_operation(operation),
                    intent_confidence=None,
                    operation=_response_operation(operation),
                    status="rejected",
                    pending_note_action_id=action_id,
                    target_note_ids=_target_note_ids(payload),
                    unsupported=[status_error],
                )
            self._consume_pending_action(action_id)
            return self._operation_response(
                answer="Cancelled the pending note action.",
                trace_id=trace_id,
                intent=_intent_for_operation(operation),
                intent_confidence=None,
                operation=_response_operation(operation),
                status="cancelled",
                pending_note_action_id=action_id,
                target_note_ids=_target_note_ids(payload),
            )

    def _prepare_grounded_note(
        self,
        *,
        original_input: str,
        note_text: str,
        filters: dict[str, Any],
        top_k: int | None,
        trace_id: str,
    ) -> "GroundedNote":
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
        rejection_reason = _clean_reason(payload.rejection_reason)
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

    def _update_note(
        self,
        *,
        note_id: str,
        grounded: "GroundedNote",
        trace_id: str,
    ) -> None:
        source_id = _shared_source_id(grounded.selected_chunks)
        work_metadata = _work_metadata(grounded.inferred_work, grounded.selected_chunks)
        self.conn.execute(
            """
            UPDATE notes
            SET original_input = %s,
                rewritten_note = %s,
                inferred_work = %s,
                source_id = %s,
                work_metadata = %s,
                embedding = %s::vector,
                model = %s,
                prompt_version = %s,
                trace_id = %s,
                updated_at = now()
            WHERE note_id = %s
            """,
            [
                grounded.original_input,
                grounded.rewritten_note,
                grounded.inferred_work,
                source_id,
                Jsonb(work_metadata),
                _vector_literal(grounded.embedding),
                self.settings.llm_model,
                self.settings.note_prompt_version,
                trace_id,
                note_id,
            ],
        )
        self.conn.execute("DELETE FROM note_chunks WHERE note_id = %s", [note_id])
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO note_chunks (note_id, chunk_id, rank, label)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (note_id, chunk.chunk_id, rank, chunk.label)
                    for rank, chunk in enumerate(grounded.selected_chunks, start=1)
                ],
            )

    def _confirm_edit(self, action_id: str, payload: dict[str, Any], trace_id: str) -> ChatResponse:
        target_id = str(payload.get("target_note_id") or "")
        target = self._fetch_note(target_id)
        if target is None:
            reason = "I could not find the note to edit."
            self._consume_pending_action(action_id)
            return self._operation_response(
                answer=reason,
                trace_id=trace_id,
                intent="note_edit",
                intent_confidence=None,
                operation="edit",
                status="not_found",
                pending_note_action_id=action_id,
                target_note_ids=[target_id] if target_id else [],
                unsupported=[reason],
            )
        grounded = GroundedNote.from_payload(_json_payload(payload.get("grounded")))
        self._update_note(note_id=target_id, grounded=grounded, trace_id=trace_id)
        self._consume_pending_action(action_id)
        return self._operation_response(
            answer=f"Updated note for {grounded.inferred_work}:\n{grounded.rewritten_note}",
            trace_id=trace_id,
            intent="note_edit",
            intent_confidence=None,
            operation="edit",
            status="completed",
            pending_note_action_id=action_id,
            target_note_ids=[target_id],
            note=grounded.rewritten_note,
            note_work=grounded.inferred_work,
            note_chunk_ids=[chunk.chunk_id for chunk in grounded.selected_chunks],
        )

    def _confirm_delete(
        self,
        action_id: str,
        operation: str,
        payload: dict[str, Any],
        trace_id: str,
    ) -> ChatResponse:
        target_ids = _target_note_ids(payload)
        if not target_ids:
            self._consume_pending_action(action_id)
            return self._operation_response(
                answer="There were no saved notes to delete.",
                trace_id=trace_id,
                intent=_intent_for_operation(operation),
                intent_confidence=None,
                operation=_response_operation(operation),
                status="completed",
                pending_note_action_id=action_id,
                target_note_ids=[],
            )
        existing_ids = self._existing_note_ids(target_ids)
        if not existing_ids:
            reason = "I could not find the note or notes to delete."
            self._consume_pending_action(action_id)
            return self._operation_response(
                answer=reason,
                trace_id=trace_id,
                intent=_intent_for_operation(operation),
                intent_confidence=None,
                operation=_response_operation(operation),
                status="not_found",
                pending_note_action_id=action_id,
                target_note_ids=target_ids,
                unsupported=[reason],
            )
        self.conn.execute("DELETE FROM notes WHERE note_id = ANY(%s)", [existing_ids])
        self._consume_pending_action(action_id)
        note_count = len(existing_ids)
        noun = "note" if note_count == 1 else "notes"
        return self._operation_response(
            answer=f"Deleted {note_count} saved {noun}.",
            trace_id=trace_id,
            intent=_intent_for_operation(operation),
            intent_confidence=None,
            operation=_response_operation(operation),
            status="completed",
            pending_note_action_id=action_id,
            target_note_ids=target_ids,
        )

    def _fetch_note(self, note_id: str) -> dict[str, Any] | None:
        if not note_id:
            return None
        row = self.conn.execute(
            """
            SELECT note_id, original_input, rewritten_note, inferred_work, source_id, created_at
            FROM notes
            WHERE note_id = %s
            """,
            [note_id],
        ).fetchone()
        return dict(row) if row is not None else None

    def _existing_note_ids(self, note_ids: list[str]) -> list[str]:
        if not note_ids:
            return []
        rows = self.conn.execute(
            """
            SELECT note_id
            FROM notes
            WHERE note_id = ANY(%s)
            """,
            [note_ids],
        ).fetchall()
        return [str(dict(row)["note_id"]) for row in rows]

    def _resolve_target(
        self,
        target_reference: str | None,
        note_context: NoteContext | None,
    ) -> tuple[str | None, str | None, str | None]:
        target_reference = _clean_reason(target_reference)
        if target_reference:
            label_index = _note_label_index(target_reference)
            if label_index is not None:
                note_ids = note_context.retrieved_note_ids if note_context else []
                if 0 <= label_index < len(note_ids):
                    return note_ids[label_index], None, None
                return None, "ambiguous", "I could not match that note label to the current notes."
            return target_reference, None, None
        if note_context is None or not note_context.active_note_id:
            return None, "ambiguous", "Please specify which saved note to use."
        return note_context.active_note_id, None, None

    def _create_pending_action(self, operation: str, payload: dict[str, Any]) -> str:
        action_id = str(uuid.uuid4())
        expires_at = datetime.now(UTC) + PENDING_ACTION_TTL
        self.conn.execute(
            """
            INSERT INTO pending_note_actions (action_id, operation, payload, expires_at)
            VALUES (%s, %s, %s, %s)
            """,
            [action_id, operation, Jsonb(payload), expires_at],
        )
        return action_id

    def _fetch_pending_action_for_update(self, action_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT action_id, operation, payload, expires_at, consumed_at
            FROM pending_note_actions
            WHERE action_id = %s
            FOR UPDATE
            """,
            [action_id],
        ).fetchone()
        return dict(row) if row is not None else None

    def _pending_status_error(self, row: dict[str, Any]) -> str | None:
        if row.get("consumed_at") is not None:
            return "That pending note action has already been used."
        expires_at = row.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at <= datetime.now(UTC):
            return "That pending note action has expired."
        return None

    def _consume_pending_action(self, action_id: str) -> None:
        self.conn.execute(
            """
            UPDATE pending_note_actions
            SET consumed_at = now(), updated_at = now()
            WHERE action_id = %s
            """,
            [action_id],
        )

    def _operation_response(
        self,
        *,
        answer: str,
        trace_id: str,
        intent: str | None,
        intent_confidence: float | None,
        operation: str | None,
        status: str,
        target_note_ids: list[str] | None = None,
        pending_note_action_id: str | None = None,
        retrieved_chunks: list[RetrievedChunk] | None = None,
        unsupported: list[str] | None = None,
        note: str | None = None,
        note_work: str | None = None,
        note_chunk_ids: list[str] | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            answer=answer,
            citations=[],
            retrieved_chunks=retrieved_chunks or [],
            prompt_version=self.settings.note_prompt_version,
            trace_id=trace_id,
            unsupported=unsupported or [],
            intent=intent,  # type: ignore[arg-type]
            intent_confidence=intent_confidence,
            note_operation=operation,  # type: ignore[arg-type]
            note_operation_status=status,  # type: ignore[arg-type]
            pending_note_action_id=pending_note_action_id,
            target_note_ids=target_note_ids or [],
            note=note,
            note_work=note_work,
            note_chunk_ids=note_chunk_ids,
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


def _note_label_index(value: str) -> int | None:
    candidate = value.strip()
    if len(candidate) < 2 or candidate[0].lower() != "n":
        return None
    suffix = candidate[1:]
    if not suffix.isdigit():
        return None
    return int(suffix) - 1


def _note_summary(row: dict[str, Any]) -> dict[str, str]:
    return {
        "note_id": str(row["note_id"]),
        "inferred_work": str(row["inferred_work"]),
        "rewritten_note": str(row["rewritten_note"]),
    }


def _json_payload(payload: object) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _target_note_ids(payload: dict[str, Any]) -> list[str]:
    value = payload.get("target_note_ids")
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    single = payload.get("target_note_id")
    return [str(single)] if single else []


def _intent_for_operation(operation: str) -> str | None:
    if operation == "edit":
        return "note_edit"
    if operation == "delete":
        return "note_delete"
    if operation == "delete_all":
        return "note_delete_all"
    return None


def _response_operation(operation: str) -> str | None:
    if operation in {"edit", "delete", "delete_all"}:
        return operation
    return None
