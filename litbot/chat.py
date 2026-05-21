import re
import uuid

import structlog
from psycopg import Connection

from litbot.config import Settings, get_settings
from litbot.generation.service import GenerationService
from litbot.intent import IntentService
from litbot.models import (
    ChatRequest,
    ChatResponse,
    IntentClassification,
)
from litbot.notes import (
    CancelPendingNoteActionCommand,
    ConfirmPendingNoteActionCommand,
    NoteRelevanceService,
    NoteRetrievalService,
    NoteWorkflow,
    PreviewDeleteAllNotesCommand,
    PreviewDeleteNoteCommand,
    PreviewEditNoteCommand,
    QueryNotesCommand,
    SaveNoteCommand,
)
from litbot.retrieval.service import RetrievalService

logger = structlog.get_logger(__name__)
NOTE_INTENTS = {"note", "note_query", "note_edit", "note_delete", "note_delete_all"}
NOTE_TARGET_RE = re.compile(
    r"\b(?:N\d+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|note-[\w-]+)\b",
    re.I,
)


class ChatOrchestrator:
    """Shared /chat and CLI ask workflow."""

    def __init__(
        self,
        conn: Connection,
        settings: Settings | None = None,
        intent_service: IntentService | None = None,
        retrieval_service: RetrievalService | None = None,
        generation_service: GenerationService | None = None,
        note_workflow: NoteWorkflow | None = None,
        note_retrieval_service: NoteRetrievalService | None = None,
        note_relevance_service: NoteRelevanceService | None = None,
    ) -> None:
        self.conn = conn
        self.settings = settings or get_settings()
        self.intent_service = intent_service or IntentService(self.settings)
        self.retrieval_service = retrieval_service or RetrievalService(conn, self.settings)
        self.generation_service = generation_service or GenerationService(self.settings)
        self.note_retrieval_service = note_retrieval_service or NoteRetrievalService(
            conn,
            self.settings,
        )
        self.note_relevance_service = note_relevance_service or NoteRelevanceService(self.settings)
        self.note_workflow = note_workflow or NoteWorkflow(
            conn,
            self.settings,
            retrieval_service=self.retrieval_service,
            note_retrieval_service=self.note_retrieval_service,
        )

    def handle(self, request: ChatRequest, trace_id: str | None = None) -> ChatResponse:
        trace_id = trace_id or str(uuid.uuid4())
        if request.confirm_note_action:
            return self.note_workflow.confirm(
                ConfirmPendingNoteActionCommand(request.pending_note_action_id, trace_id)
            )
        if request.cancel_note_action:
            return self.note_workflow.cancel(
                CancelPendingNoteActionCommand(request.pending_note_action_id, trace_id)
            )

        classification = self.intent_service.classify(request.question)
        if _should_route_as_question(classification, self.settings.intent_confidence_threshold):
            if classification.intent in NOTE_INTENTS:
                logger.info(
                    "intent_low_confidence_fallback",
                    trace_id=trace_id,
                    intent=classification.intent,
                    confidence=classification.confidence,
                    threshold=self.settings.intent_confidence_threshold,
                )
            return self._answer_question(request, trace_id, classification)

        if classification.intent == "note_query":
            return self.note_workflow.query(
                QueryNotesCommand(
                    query=(classification.extracted_note_query or request.question).strip(),
                    filters=request.filters,
                    top_k=request.top_k,
                    exact_work=classification.extracted_work,
                    mode=classification.note_query_mode,
                    trace_id=trace_id,
                    intent_confidence=classification.confidence,
                )
            )
        if classification.intent == "note_edit":
            return self.note_workflow.preview_edit(
                PreviewEditNoteCommand(
                    original_input=request.question,
                    note_text=classification.extracted_note_text,
                    target_reference=_target_reference(request.question, classification),
                    note_context=request.note_context,
                    filters=request.filters,
                    top_k=request.top_k,
                    trace_id=trace_id,
                    intent_confidence=classification.confidence,
                )
            )
        if classification.intent == "note_delete":
            return self.note_workflow.preview_delete(
                PreviewDeleteNoteCommand(
                    target_reference=_target_reference(request.question, classification),
                    note_context=request.note_context,
                    trace_id=trace_id,
                    intent_confidence=classification.confidence,
                )
            )
        if classification.intent == "note_delete_all":
            return self.note_workflow.preview_delete_all(
                PreviewDeleteAllNotesCommand(
                    trace_id=trace_id,
                    intent_confidence=classification.confidence,
                )
            )

        note_text = (classification.extracted_note_text or request.question).strip()
        return self.note_workflow.save(
            SaveNoteCommand(
                original_input=request.question,
                note_text=note_text,
                filters=request.filters,
                top_k=request.top_k,
                trace_id=trace_id,
                intent_confidence=classification.confidence,
            )
        )

    def _answer_question(
        self,
        request: ChatRequest,
        trace_id: str,
        classification: IntentClassification,
    ) -> ChatResponse:
        chunks = self.retrieval_service.retrieve(
            request.question,
            filters=request.filters,
            top_k=request.top_k,
        )
        note_candidates = self.note_retrieval_service.candidates_for_question(
            request.question,
            filters=request.filters,
        )
        relevant_notes = (
            self.note_relevance_service.filter(request.question, note_candidates)
            if note_candidates
            else []
        )
        logger.info(
            "chat_question_routed",
            trace_id=trace_id,
            chunk_count=len(chunks),
            note_candidate_count=len(note_candidates),
            relevant_note_count=len(relevant_notes),
            intent=classification.intent,
            confidence=classification.confidence,
        )
        response = self.generation_service.answer(
            request.question,
            chunks,
            notes=relevant_notes,
            trace_id=trace_id,
        )
        return response.model_copy(
            update={
                "intent": "question",
                "intent_confidence": classification.confidence,
            }
        )

def handle_chat_request(
    conn: Connection,
    settings: Settings,
    request: ChatRequest,
    trace_id: str | None = None,
) -> ChatResponse:
    return ChatOrchestrator(conn, settings).handle(request, trace_id=trace_id)


def _should_route_as_question(
    classification: IntentClassification,
    threshold: float,
) -> bool:
    return (
        classification.intent not in NOTE_INTENTS
        or classification.confidence < threshold
    )


def _target_reference(question: str, classification: IntentClassification) -> str | None:
    match = NOTE_TARGET_RE.search(question)
    if match:
        return match.group(0)
    if classification.extracted_note_target:
        return classification.extracted_note_target
    return None
