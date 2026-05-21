from datetime import UTC, datetime

from litbot.chat import ChatOrchestrator
from litbot.config import Settings
from litbot.models import (
    ChatRequest,
    ChatResponse,
    IntentClassification,
    RetrievedChunk,
    RetrievedNote,
)


def test_question_intent_routes_to_rag_answer_flow() -> None:
    retrieval = FakeRetrievalService()
    generation = FakeGenerationService()
    note = FakeNoteWorkflow()
    orchestrator = ChatOrchestrator(
        conn=object(),
        settings=Settings(intent_confidence_threshold=0.65),
        intent_service=FakeIntentService(IntentClassification(intent="question", confidence=0.92)),
        retrieval_service=retrieval,
        generation_service=generation,
        note_workflow=note,
        note_retrieval_service=FakeNoteRetrievalService(),
        note_relevance_service=FakeNoteRelevanceService(),
    )

    response = orchestrator.handle(ChatRequest(question="What happens?"), trace_id="trace-1")

    assert response.intent == "question"
    assert response.intent_confidence == 0.92
    assert retrieval.calls == ["What happens?"]
    assert generation.calls == [("What happens?", "trace-1")]
    assert note.save_calls == []


def test_note_intent_routes_to_note_workflow() -> None:
    note = FakeNoteWorkflow()
    orchestrator = ChatOrchestrator(
        conn=object(),
        settings=Settings(intent_confidence_threshold=0.65),
        intent_service=FakeIntentService(
            IntentClassification(
                intent="note",
                confidence=0.91,
                extracted_note_text="Hamlet opens with uncertainty.",
            )
        ),
        retrieval_service=FakeRetrievalService(),
        generation_service=FakeGenerationService(),
        note_workflow=note,
    )

    response = orchestrator.handle(ChatRequest(question="Save this: Hamlet opens..."))

    assert response.note_status == "saved"
    assert note.save_calls[0].note_text == "Hamlet opens with uncertainty."
    assert note.save_calls[0].intent_confidence == 0.91


def test_low_confidence_note_intent_falls_back_to_question_flow() -> None:
    retrieval = FakeRetrievalService()
    note = FakeNoteWorkflow()
    orchestrator = ChatOrchestrator(
        conn=object(),
        settings=Settings(intent_confidence_threshold=0.65),
        intent_service=FakeIntentService(IntentClassification(intent="note", confidence=0.4)),
        retrieval_service=retrieval,
        generation_service=FakeGenerationService(),
        note_workflow=note,
        note_retrieval_service=FakeNoteRetrievalService(),
        note_relevance_service=FakeNoteRelevanceService(),
    )

    response = orchestrator.handle(ChatRequest(question="Maybe save this?"))

    assert response.intent == "question"
    assert response.intent_confidence == 0.4
    assert retrieval.calls == ["Maybe save this?"]
    assert note.save_calls == []


def test_note_query_intent_routes_to_note_workflow() -> None:
    note = FakeNoteWorkflow()
    orchestrator = ChatOrchestrator(
        conn=object(),
        settings=Settings(intent_confidence_threshold=0.65),
        intent_service=FakeIntentService(
            IntentClassification(
                intent="note_query",
                confidence=0.9,
                extracted_note_query="Hamlet",
                extracted_work="Hamlet",
                note_query_mode="search",
            )
        ),
        retrieval_service=FakeRetrievalService(),
        generation_service=FakeGenerationService(),
        note_workflow=note,
        note_retrieval_service=FakeNoteRetrievalService(),
        note_relevance_service=FakeNoteRelevanceService(),
    )

    response = orchestrator.handle(ChatRequest(question="What note did I make for Hamlet?"))

    assert response.intent == "note_query"
    assert response.note_query_status == "found"
    assert response.retrieved_notes[0].rewritten_note == "Hamlet begins in uncertainty."
    assert note.query_calls[0].exact_work == "Hamlet"
    assert note.query_calls[0].query == "Hamlet"


def test_note_delete_intent_routes_to_preview_with_context() -> None:
    note = FakeNoteWorkflow()
    orchestrator = ChatOrchestrator(
        conn=object(),
        settings=Settings(intent_confidence_threshold=0.65),
        intent_service=FakeIntentService(
            IntentClassification(intent="note_delete", confidence=0.9)
        ),
        retrieval_service=FakeRetrievalService(),
        generation_service=FakeGenerationService(),
        note_workflow=note,
        note_retrieval_service=FakeNoteRetrievalService(),
        note_relevance_service=FakeNoteRelevanceService(),
    )

    response = orchestrator.handle(
        ChatRequest(
            question="Delete this.",
            note_context={"active_note_id": "note-1", "retrieved_note_ids": ["note-1"]},
        ),
        trace_id="trace-1",
    )

    assert response.note_operation == "delete"
    assert response.note_operation_status == "pending_confirmation"
    assert note.delete_calls[0].note_context.active_note_id == "note-1"


def test_note_delete_explicit_uuid_target_is_parsed_from_question() -> None:
    note = FakeNoteWorkflow()
    note_id = "123e4567-e89b-12d3-a456-426614174000"
    orchestrator = ChatOrchestrator(
        conn=object(),
        settings=Settings(intent_confidence_threshold=0.65),
        intent_service=FakeIntentService(
            IntentClassification(intent="note_delete", confidence=0.9)
        ),
        retrieval_service=FakeRetrievalService(),
        generation_service=FakeGenerationService(),
        note_workflow=note,
        note_retrieval_service=FakeNoteRetrievalService(),
        note_relevance_service=FakeNoteRelevanceService(),
    )

    orchestrator.handle(ChatRequest(question=f"Delete note {note_id}"), trace_id="trace-1")

    assert note.delete_calls[0].target_reference == note_id


def test_pending_confirmation_bypasses_intent_classifier() -> None:
    note = FakeNoteWorkflow()
    intent = FakeIntentService(IntentClassification(intent="question", confidence=0.1))
    orchestrator = ChatOrchestrator(
        conn=object(),
        settings=Settings(intent_confidence_threshold=0.65),
        intent_service=intent,
        retrieval_service=FakeRetrievalService(),
        generation_service=FakeGenerationService(),
        note_workflow=note,
        note_retrieval_service=FakeNoteRetrievalService(),
        note_relevance_service=FakeNoteRelevanceService(),
    )

    response = orchestrator.handle(
        ChatRequest(
            question="Confirm note action.",
            pending_note_action_id="action-1",
            confirm_note_action=True,
        ),
        trace_id="trace-1",
    )

    assert response.note_operation_status == "completed"
    assert [(call.action_id, call.trace_id) for call in note.confirm_calls] == [
        ("action-1", "trace-1")
    ]


def test_question_supplement_skips_relevance_filter_without_note_candidates() -> None:
    relevance = FakeNoteRelevanceService()
    orchestrator = ChatOrchestrator(
        conn=object(),
        settings=Settings(intent_confidence_threshold=0.65),
        intent_service=FakeIntentService(IntentClassification(intent="question", confidence=0.92)),
        retrieval_service=FakeRetrievalService(),
        generation_service=FakeGenerationService(),
        note_workflow=FakeNoteWorkflow(),
        note_retrieval_service=FakeNoteRetrievalService(notes=[]),
        note_relevance_service=relevance,
    )

    response = orchestrator.handle(ChatRequest(question="What happens?"), trace_id="trace-1")

    assert response.intent == "question"
    assert response.retrieved_notes == []
    assert relevance.calls == []


class FakeIntentService:
    def __init__(self, classification: IntentClassification) -> None:
        self.classification = classification
        self.calls = []

    def classify(self, _user_input: str) -> IntentClassification:
        self.calls.append(_user_input)
        return self.classification


class FakeRetrievalService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, question, filters=None, top_k=None) -> list[RetrievedChunk]:
        self.calls.append(question)
        return [
            RetrievedChunk(
                label="S1",
                chunk_id="chunk-1",
                source_id="source-1",
                text="Evidence.",
                metadata={"work": "Hamlet"},
                combined_score=1.0,
                reason="test",
            )
        ]


class FakeGenerationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def answer(self, question, chunks, notes=None, trace_id=None) -> ChatResponse:
        self.calls.append((question, trace_id))
        return ChatResponse(
            answer="Answer.",
            citations=[],
            retrieved_chunks=chunks,
            retrieved_notes=notes or [],
            prompt_version="test",
            trace_id=trace_id,
        )


class FakeNoteWorkflow:
    def __init__(self) -> None:
        self.save_calls = []
        self.query_calls = []
        self.delete_calls = []
        self.confirm_calls = []

    def save(self, command) -> ChatResponse:
        self.save_calls.append(command)
        return ChatResponse(
            answer="Saved note.",
            citations=[],
            retrieved_chunks=[],
            prompt_version="note-test",
            trace_id=command.trace_id,
            intent="note",
            intent_confidence=command.intent_confidence,
            note_status="saved",
            note="Saved note.",
            original_note=command.original_input,
            note_work="Hamlet",
            note_chunk_ids=["chunk-1"],
        )

    def query(self, command) -> ChatResponse:
        self.query_calls.append(command)
        note = RetrievedNote(
            label="N1",
            note_id="note-1",
            rewritten_note="Hamlet begins in uncertainty.",
            original_input="Save this.",
            inferred_work="Hamlet",
            matched_work="Hamlet",
            created_at=datetime.now(UTC),
            combined_score=1.0,
            reason="test",
        )
        return ChatResponse(
            answer="I found these saved notes:\n- [N1] Hamlet: Hamlet begins in uncertainty.",
            citations=[],
            retrieved_chunks=[],
            retrieved_notes=[note],
            prompt_version="note-test",
            trace_id=command.trace_id,
            intent="note_query",
            intent_confidence=command.intent_confidence,
            note_query_status="found",
            note_query_has_more=False,
        )

    def preview_delete(self, command) -> ChatResponse:
        self.delete_calls.append(command)
        return ChatResponse(
            answer="Please confirm deleting this note.",
            citations=[],
            retrieved_chunks=[],
            prompt_version="note-test",
            trace_id=command.trace_id,
            intent="note_delete",
            intent_confidence=command.intent_confidence,
            note_operation="delete",
            note_operation_status="pending_confirmation",
            pending_note_action_id="action-1",
            target_note_ids=["note-1"],
        )

    def confirm(self, command) -> ChatResponse:
        self.confirm_calls.append(command)
        return ChatResponse(
            answer="Deleted 1 saved note.",
            citations=[],
            retrieved_chunks=[],
            prompt_version="note-test",
            trace_id=command.trace_id,
            note_operation="delete",
            note_operation_status="completed",
            pending_note_action_id=command.action_id,
            target_note_ids=["note-1"],
        )

    def cancel(self, command) -> ChatResponse:
        return self.confirm(command)


class FakeNoteRetrievalResult:
    def __init__(self, notes: list[RetrievedNote], has_more: bool = False) -> None:
        self.notes = notes
        self.has_more = has_more
        self.match_strategy = "test"


class FakeNoteRetrievalService:
    def __init__(self, notes: list[RetrievedNote] | None = None) -> None:
        self.notes = notes or []
        self.search_calls: list[dict] = []
        self.list_calls: list[dict] = []

    def search(self, query, filters=None, top_k=None, exact_work=None):
        self.search_calls.append(
            {"query": query, "filters": filters, "top_k": top_k, "exact_work": exact_work}
        )
        return FakeNoteRetrievalResult(self.notes)

    def list_all(self, filters=None, top_k=None, exact_work=None):
        self.list_calls.append({"filters": filters, "top_k": top_k, "exact_work": exact_work})
        return FakeNoteRetrievalResult(self.notes)

    def candidates_for_question(self, question, filters=None):
        return self.notes


class FakeNoteRelevanceService:
    def __init__(self, relevant: list[RetrievedNote] | None = None) -> None:
        self.relevant = relevant or []
        self.calls: list[tuple[str, list[RetrievedNote]]] = []

    def filter(self, question, notes):
        self.calls.append((question, notes))
        return self.relevant
