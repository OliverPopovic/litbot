import uuid

import structlog
from psycopg import Connection

from litbot.config import Settings, get_settings
from litbot.generation.service import GenerationService
from litbot.intent import IntentService
from litbot.models import ChatRequest, ChatResponse, IntentClassification
from litbot.notes import NoteService
from litbot.retrieval.service import RetrievalService

logger = structlog.get_logger(__name__)


class ChatOrchestrator:
    """Shared /chat and CLI ask workflow."""

    def __init__(
        self,
        conn: Connection,
        settings: Settings | None = None,
        intent_service: IntentService | None = None,
        retrieval_service: RetrievalService | None = None,
        generation_service: GenerationService | None = None,
        note_service: NoteService | None = None,
    ) -> None:
        self.conn = conn
        self.settings = settings or get_settings()
        self.intent_service = intent_service or IntentService(self.settings)
        self.retrieval_service = retrieval_service or RetrievalService(conn, self.settings)
        self.generation_service = generation_service or GenerationService(self.settings)
        self.note_service = note_service or NoteService(
            conn,
            self.settings,
            retrieval_service=self.retrieval_service,
        )

    def handle(self, request: ChatRequest, trace_id: str | None = None) -> ChatResponse:
        trace_id = trace_id or str(uuid.uuid4())
        classification = self.intent_service.classify(request.question)
        if _should_route_as_question(classification, self.settings.intent_confidence_threshold):
            if classification.intent == "note":
                logger.info(
                    "intent_low_confidence_fallback",
                    trace_id=trace_id,
                    confidence=classification.confidence,
                    threshold=self.settings.intent_confidence_threshold,
                )
            return self._answer_question(request, trace_id, classification)

        note_text = (classification.extracted_note_text or request.question).strip()
        return self.note_service.process(
            original_input=request.question,
            note_text=note_text,
            filters=request.filters,
            top_k=request.top_k,
            trace_id=trace_id,
            intent_confidence=classification.confidence,
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
        logger.info(
            "chat_question_routed",
            trace_id=trace_id,
            chunk_count=len(chunks),
            intent=classification.intent,
            confidence=classification.confidence,
        )
        response = self.generation_service.answer(request.question, chunks, trace_id=trace_id)
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
    return classification.intent != "note" or classification.confidence < threshold
