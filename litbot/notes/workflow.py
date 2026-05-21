import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from psycopg import Connection

from litbot.config import Settings, get_settings
from litbot.models import ChatResponse, NoteContext
from litbot.notes.grounding import GroundedNote, NoteGroundingService
from litbot.notes.repository import (
    NoteRepository,
    PendingNoteActionRepository,
    intent_for_operation,
    json_payload,
    note_summary,
    response_operation,
    target_note_ids,
    valid_note_ids,
)
from litbot.notes.responses import NoteResponseFactory
from litbot.notes.retrieval import NoteRetrievalService
from litbot.notes.target import NoteTargetResolver
from litbot.retrieval.service import RetrievalService

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SaveNoteCommand:
    original_input: str
    note_text: str
    filters: dict[str, Any] = field(default_factory=dict)
    top_k: int | None = None
    trace_id: str = ""
    intent_confidence: float | None = None


@dataclass(frozen=True)
class QueryNotesCommand:
    query: str
    filters: dict[str, Any] = field(default_factory=dict)
    top_k: int | None = None
    exact_work: str | None = None
    mode: str | None = None
    trace_id: str = ""
    intent_confidence: float | None = None


@dataclass(frozen=True)
class PreviewEditNoteCommand:
    original_input: str
    note_text: str | None
    target_reference: str | None
    note_context: NoteContext | None
    filters: dict[str, Any] | None
    top_k: int | None
    trace_id: str
    intent_confidence: float | None


@dataclass(frozen=True)
class PreviewDeleteNoteCommand:
    target_reference: str | None
    note_context: NoteContext | None
    trace_id: str
    intent_confidence: float | None


@dataclass(frozen=True)
class PreviewDeleteAllNotesCommand:
    trace_id: str
    intent_confidence: float | None


@dataclass(frozen=True)
class ConfirmPendingNoteActionCommand:
    action_id: str | None
    trace_id: str


@dataclass(frozen=True)
class CancelPendingNoteActionCommand:
    action_id: str | None
    trace_id: str


class NoteWorkflow:
    """Application workflow layer for all note operations."""

    def __init__(
        self,
        conn: Connection,
        settings: Settings | None = None,
        retrieval_service: RetrievalService | None = None,
        grounding_service: NoteGroundingService | None = None,
        note_repository: NoteRepository | None = None,
        pending_repository: PendingNoteActionRepository | None = None,
        note_retrieval_service: NoteRetrievalService | None = None,
        target_resolver: NoteTargetResolver | None = None,
        response_factory: NoteResponseFactory | None = None,
    ) -> None:
        self.conn = conn
        self.settings = settings or get_settings()
        self.grounding = grounding_service or NoteGroundingService(
            conn,
            self.settings,
            retrieval_service=retrieval_service,
        )
        self.notes = note_repository or NoteRepository(conn, self.settings)
        self.pending = pending_repository or PendingNoteActionRepository(conn)
        self.note_retrieval = note_retrieval_service or NoteRetrievalService(conn, self.settings)
        self.targets = target_resolver or NoteTargetResolver()
        self.responses = response_factory or NoteResponseFactory(self.settings)

    def save(self, command: SaveNoteCommand) -> ChatResponse:
        _log_started("note", command.trace_id)
        filters = dict(command.filters or {})
        grounded = self.grounding.prepare(
            original_input=command.original_input,
            note_text=command.note_text,
            filters=filters,
            top_k=command.top_k,
            trace_id=command.trace_id,
        )
        if grounded.rejection_reason:
            response = self.responses.not_saved(
                original_input=command.original_input,
                note_text=command.note_text,
                chunks=grounded.retrieved_chunks,
                trace_id=command.trace_id,
                reason=grounded.rejection_reason,
                intent_confidence=command.intent_confidence,
            )
            _log_completed(response)
            return response

        note_id = str(uuid.uuid4())
        self.notes.insert(note_id=note_id, grounded=grounded, trace_id=command.trace_id)
        selected_chunk_ids = [chunk.chunk_id for chunk in grounded.selected_chunks]
        logger.info(
            "note_saved",
            trace_id=command.trace_id,
            note_id=note_id,
            inferred_work=grounded.inferred_work,
            selected_chunks=selected_chunk_ids,
        )
        response = self.responses.saved(
            note_id=note_id,
            grounded=grounded,
            original_input=command.original_input,
            trace_id=command.trace_id,
            intent_confidence=command.intent_confidence,
        )
        _log_completed(response)
        return response

    def query(self, command: QueryNotesCommand) -> ChatResponse:
        _log_started("note_query", command.trace_id)
        exact_work = command.exact_work or command.filters.get("work")
        if command.mode == "list_all":
            result = self.note_retrieval.list_all(
                filters=command.filters,
                top_k=command.top_k,
                exact_work=exact_work if isinstance(exact_work, str) else None,
            )
        else:
            result = self.note_retrieval.search(
                command.query,
                filters=command.filters,
                top_k=command.top_k,
                exact_work=exact_work if isinstance(exact_work, str) else None,
            )
        logger.info(
            "chat_note_query_routed",
            trace_id=command.trace_id,
            note_count=len(result.notes),
            has_more=result.has_more,
            match_strategy=result.match_strategy,
            confidence=command.intent_confidence,
        )
        response = self.responses.note_query(
            result=result,
            trace_id=command.trace_id,
            intent_confidence=command.intent_confidence,
        )
        _log_completed(response)
        return response

    def preview_edit(self, command: PreviewEditNoteCommand) -> ChatResponse:
        _log_started("note_edit", command.trace_id, operation="edit")
        target = self.targets.resolve(command.target_reference, command.note_context)
        if target.failed:
            response = self._target_failed_response(
                trace_id=command.trace_id,
                intent="note_edit",
                intent_confidence=command.intent_confidence,
                operation="edit",
                status=str(target.status),
                reason=str(target.reason),
                target_note_ids=[target.note_id] if target.note_id else None,
            )
            _log_completed(response)
            return response
        current_note = self.notes.fetch(str(target.note_id))
        if current_note is None:
            reason = "I could not find that saved note."
            response = self.responses.operation(
                answer=reason,
                trace_id=command.trace_id,
                intent="note_edit",
                intent_confidence=command.intent_confidence,
                operation="edit",
                status="not_found",
                target_note_ids=[str(target.note_id)],
                unsupported=[reason],
            )
            _log_completed(response)
            return response
        if not command.note_text or not command.note_text.strip():
            reason = "I need the replacement note text before I can prepare an edit."
            response = self.responses.operation(
                answer=reason,
                trace_id=command.trace_id,
                intent="note_edit",
                intent_confidence=command.intent_confidence,
                operation="edit",
                status="rejected",
                target_note_ids=[str(target.note_id)],
                unsupported=[reason],
            )
            _log_completed(response)
            return response

        grounded = self.grounding.prepare(
            original_input=command.original_input,
            note_text=command.note_text,
            filters=dict(command.filters or {}),
            top_k=command.top_k,
            trace_id=command.trace_id,
        )
        if grounded.rejection_reason:
            response = self.responses.operation(
                answer=f"I could not prepare that edit: {grounded.rejection_reason}",
                trace_id=command.trace_id,
                intent="note_edit",
                intent_confidence=command.intent_confidence,
                operation="edit",
                status="rejected",
                target_note_ids=[str(target.note_id)],
                retrieved_chunks=grounded.retrieved_chunks,
                unsupported=[grounded.rejection_reason],
            )
            _log_completed(response)
            return response

        action_id = self.pending.create(
            "edit",
            {
                "target_note_id": str(target.note_id),
                "original_input": command.original_input,
                "note_text": command.note_text,
                "previous_note": current_note["rewritten_note"],
                "grounded": grounded.to_payload(),
            },
        )
        logger.info(
            "pending_note_action_created",
            trace_id=command.trace_id,
            operation="edit",
            pending_note_action_id=action_id,
            target_count=1,
        )
        answer = (
            "Please confirm this note edit.\n"
            f"Current note: {current_note['rewritten_note']}\n"
            f"Proposed note: {grounded.rewritten_note}"
        )
        response = self.responses.operation(
            answer=answer,
            trace_id=command.trace_id,
            intent="note_edit",
            intent_confidence=command.intent_confidence,
            operation="edit",
            status="pending_confirmation",
            target_note_ids=[str(target.note_id)],
            pending_note_action_id=action_id,
            retrieved_chunks=grounded.retrieved_chunks,
            note=grounded.rewritten_note,
            note_work=grounded.inferred_work,
            note_chunk_ids=[chunk.chunk_id for chunk in grounded.selected_chunks],
        )
        _log_completed(response)
        return response

    def preview_delete(self, command: PreviewDeleteNoteCommand) -> ChatResponse:
        _log_started("note_delete", command.trace_id, operation="delete")
        target = self.targets.resolve(command.target_reference, command.note_context)
        if target.failed:
            response = self._target_failed_response(
                trace_id=command.trace_id,
                intent="note_delete",
                intent_confidence=command.intent_confidence,
                operation="delete",
                status=str(target.status),
                reason=str(target.reason),
                target_note_ids=[target.note_id] if target.note_id else None,
            )
            _log_completed(response)
            return response
        current_note = self.notes.fetch(str(target.note_id))
        if current_note is None:
            reason = "I could not find that saved note."
            response = self.responses.operation(
                answer=reason,
                trace_id=command.trace_id,
                intent="note_delete",
                intent_confidence=command.intent_confidence,
                operation="delete",
                status="not_found",
                target_note_ids=[str(target.note_id)],
                unsupported=[reason],
            )
            _log_completed(response)
            return response
        action_id = self.pending.create(
            "delete",
            {
                "target_note_ids": [str(target.note_id)],
                "notes": [note_summary(current_note)],
            },
        )
        logger.info(
            "pending_note_action_created",
            trace_id=command.trace_id,
            operation="delete",
            pending_note_action_id=action_id,
            target_count=1,
        )
        answer = (
            "Please confirm deleting this note.\n"
            f"[{current_note['inferred_work']}] {current_note['rewritten_note']}"
        )
        response = self.responses.operation(
            answer=answer,
            trace_id=command.trace_id,
            intent="note_delete",
            intent_confidence=command.intent_confidence,
            operation="delete",
            status="pending_confirmation",
            target_note_ids=[str(target.note_id)],
            pending_note_action_id=action_id,
        )
        _log_completed(response)
        return response

    def preview_delete_all(self, command: PreviewDeleteAllNotesCommand) -> ChatResponse:
        _log_started("note_delete_all", command.trace_id, operation="delete_all")
        notes = self.notes.list_all_rows()
        note_ids = [str(row["note_id"]) for row in notes]
        if not note_ids:
            response = self.responses.operation(
                answer="There are no saved notes to delete.",
                trace_id=command.trace_id,
                intent="note_delete_all",
                intent_confidence=command.intent_confidence,
                operation="delete_all",
                status="completed",
                target_note_ids=[],
            )
            _log_completed(response)
            return response
        action_id = self.pending.create(
            "delete_all",
            {
                "target_note_ids": note_ids,
                "notes": [note_summary(note) for note in notes],
            },
        )
        logger.info(
            "pending_note_action_created",
            trace_id=command.trace_id,
            operation="delete_all",
            pending_note_action_id=action_id,
            target_count=len(note_ids),
        )
        answer = (
            "Please confirm deleting all saved notes. "
            f"This will delete {len(note_ids)} notes."
        )
        response = self.responses.operation(
            answer=answer,
            trace_id=command.trace_id,
            intent="note_delete_all",
            intent_confidence=command.intent_confidence,
            operation="delete_all",
            status="pending_confirmation",
            target_note_ids=note_ids,
            pending_note_action_id=action_id,
        )
        _log_completed(response)
        return response

    def confirm(self, command: ConfirmPendingNoteActionCommand) -> ChatResponse:
        _log_started("pending_note_action", command.trace_id, operation="confirm")
        if not command.action_id:
            response = self._missing_action_response(command.trace_id)
            _log_completed(response)
            return response
        with self.conn.transaction():
            action = self.pending.fetch_for_update(command.action_id)
            if action is None:
                return self.responses.operation(
                    answer="I could not find that pending note action.",
                    trace_id=command.trace_id,
                    intent=None,
                    intent_confidence=None,
                    operation=None,
                    status="rejected",
                    unsupported=["Pending note action was not found."],
                )
            status_error = self.pending.status_error(action)
            if status_error:
                response = self.responses.operation(
                    answer=status_error,
                    trace_id=command.trace_id,
                    intent=intent_for_operation(action.operation),
                    intent_confidence=None,
                    operation=response_operation(action.operation),
                    status="rejected",
                    pending_note_action_id=command.action_id,
                    target_note_ids=target_note_ids(action.payload),
                    unsupported=[status_error],
                )
                _log_completed(response)
                return response
            if action.operation == "edit":
                response = self._confirm_edit(command.action_id, action.payload, command.trace_id)
            elif action.operation in {"delete", "delete_all"}:
                response = self._confirm_delete(
                    command.action_id,
                    action.operation,
                    action.payload,
                    command.trace_id,
                )
            else:
                reason = "The pending note action has an unsupported operation."
                self.pending.consume(command.action_id)
                _log_consumed(command.trace_id, action.operation, command.action_id)
                response = self.responses.operation(
                    answer=reason,
                    trace_id=command.trace_id,
                    intent=None,
                    intent_confidence=None,
                    operation=None,
                    status="rejected",
                    pending_note_action_id=command.action_id,
                    target_note_ids=target_note_ids(action.payload),
                    unsupported=[reason],
                )
            _log_completed(response)
            return response

    def cancel(self, command: CancelPendingNoteActionCommand) -> ChatResponse:
        _log_started("pending_note_action", command.trace_id, operation="cancel")
        if not command.action_id:
            response = self._missing_action_response(command.trace_id)
            _log_completed(response)
            return response
        with self.conn.transaction():
            action = self.pending.fetch_for_update(command.action_id)
            if action is None:
                return self.responses.operation(
                    answer="I could not find that pending note action.",
                    trace_id=command.trace_id,
                    intent=None,
                    intent_confidence=None,
                    operation=None,
                    status="rejected",
                    unsupported=["Pending note action was not found."],
                )
            status_error = self.pending.status_error(action)
            if status_error:
                response = self.responses.operation(
                    answer=status_error,
                    trace_id=command.trace_id,
                    intent=intent_for_operation(action.operation),
                    intent_confidence=None,
                    operation=response_operation(action.operation),
                    status="rejected",
                    pending_note_action_id=command.action_id,
                    target_note_ids=target_note_ids(action.payload),
                    unsupported=[status_error],
                )
                _log_completed(response)
                return response
            self.pending.consume(command.action_id)
            _log_consumed(command.trace_id, action.operation, command.action_id)
            response = self.responses.operation(
                answer="Cancelled the pending note action.",
                trace_id=command.trace_id,
                intent=intent_for_operation(action.operation),
                intent_confidence=None,
                operation=response_operation(action.operation),
                status="cancelled",
                pending_note_action_id=command.action_id,
                target_note_ids=target_note_ids(action.payload),
            )
            _log_completed(response)
            return response

    def _confirm_edit(
        self,
        action_id: str,
        payload: dict[str, Any],
        trace_id: str,
    ) -> ChatResponse:
        target_id = str(payload.get("target_note_id") or "")
        if not valid_note_ids([target_id]):
            reason = "I could not find the note to edit."
            self.pending.consume(action_id)
            _log_consumed(trace_id, "edit", action_id)
            return self.responses.operation(
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
        target = self.notes.fetch(target_id)
        if target is None:
            reason = "I could not find the note to edit."
            self.pending.consume(action_id)
            _log_consumed(trace_id, "edit", action_id)
            return self.responses.operation(
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
        grounded = GroundedNote.from_payload(json_payload(payload.get("grounded")))
        self.notes.update(note_id=target_id, grounded=grounded, trace_id=trace_id)
        self.pending.consume(action_id)
        _log_consumed(trace_id, "edit", action_id)
        return self.responses.operation(
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
        target_ids = target_note_ids(payload)
        valid_target_ids = valid_note_ids(target_ids)
        if not target_ids:
            self.pending.consume(action_id)
            _log_consumed(trace_id, operation, action_id)
            return self.responses.operation(
                answer="There were no saved notes to delete.",
                trace_id=trace_id,
                intent=intent_for_operation(operation),
                intent_confidence=None,
                operation=response_operation(operation),
                status="completed",
                pending_note_action_id=action_id,
                target_note_ids=[],
            )
        existing_ids = self.notes.existing_ids(valid_target_ids)
        if not existing_ids:
            reason = "I could not find the note or notes to delete."
            self.pending.consume(action_id)
            _log_consumed(trace_id, operation, action_id)
            return self.responses.operation(
                answer=reason,
                trace_id=trace_id,
                intent=intent_for_operation(operation),
                intent_confidence=None,
                operation=response_operation(operation),
                status="not_found",
                pending_note_action_id=action_id,
                target_note_ids=target_ids,
                unsupported=[reason],
            )
        self.notes.delete_many(existing_ids)
        self.pending.consume(action_id)
        _log_consumed(trace_id, operation, action_id)
        note_count = len(existing_ids)
        noun = "note" if note_count == 1 else "notes"
        return self.responses.operation(
            answer=f"Deleted {note_count} saved {noun}.",
            trace_id=trace_id,
            intent=intent_for_operation(operation),
            intent_confidence=None,
            operation=response_operation(operation),
            status="completed",
            pending_note_action_id=action_id,
            target_note_ids=target_ids,
        )

    def _target_failed_response(
        self,
        *,
        trace_id: str,
        intent: str,
        intent_confidence: float | None,
        operation: str,
        status: str,
        reason: str,
        target_note_ids: list[str] | None = None,
    ) -> ChatResponse:
        logger.info(
            "note_target_resolution_failed",
            trace_id=trace_id,
            intent=intent,
            operation=operation,
            status=status,
        )
        return self.responses.operation(
            answer=reason,
            trace_id=trace_id,
            intent=intent,
            intent_confidence=intent_confidence,
            operation=operation,
            status=status,
            target_note_ids=target_note_ids or [],
            unsupported=[reason],
        )

    def _missing_action_response(self, trace_id: str) -> ChatResponse:
        return self.responses.operation(
            answer="No pending note action was supplied.",
            trace_id=trace_id,
            intent=None,
            intent_confidence=None,
            operation=None,
            status="rejected",
            unsupported=["No pending note action was supplied."],
        )


def _log_started(intent: str, trace_id: str, *, operation: str | None = None) -> None:
    logger.info(
        "note_workflow_started",
        trace_id=trace_id,
        intent=intent,
        operation=operation,
    )


def _log_completed(response: ChatResponse) -> None:
    logger.info(
        "note_workflow_completed",
        trace_id=response.trace_id,
        intent=response.intent,
        operation=response.note_operation,
        status=response.note_operation_status or response.note_status or response.note_query_status,
        pending_note_action_id=response.pending_note_action_id,
        target_count=len(response.target_note_ids),
    )


def _log_consumed(trace_id: str, operation: str, action_id: str) -> None:
    logger.info(
        "pending_note_action_consumed",
        trace_id=trace_id,
        operation=operation,
        pending_note_action_id=action_id,
    )
