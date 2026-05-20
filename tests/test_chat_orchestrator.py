from litbot.chat import ChatOrchestrator
from litbot.config import Settings
from litbot.models import ChatRequest, ChatResponse, IntentClassification, RetrievedChunk


def test_question_intent_routes_to_rag_answer_flow() -> None:
    retrieval = FakeRetrievalService()
    generation = FakeGenerationService()
    note = FakeNoteService()
    orchestrator = ChatOrchestrator(
        conn=object(),
        settings=Settings(intent_confidence_threshold=0.65),
        intent_service=FakeIntentService(IntentClassification(intent="question", confidence=0.92)),
        retrieval_service=retrieval,
        generation_service=generation,
        note_service=note,
    )

    response = orchestrator.handle(ChatRequest(question="What happens?"), trace_id="trace-1")

    assert response.intent == "question"
    assert response.intent_confidence == 0.92
    assert retrieval.calls == ["What happens?"]
    assert generation.calls == [("What happens?", "trace-1")]
    assert note.calls == []


def test_note_intent_routes_to_note_service() -> None:
    note = FakeNoteService()
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
        note_service=note,
    )

    response = orchestrator.handle(ChatRequest(question="Save this: Hamlet opens..."))

    assert response.note_status == "saved"
    assert note.calls[0]["note_text"] == "Hamlet opens with uncertainty."
    assert note.calls[0]["intent_confidence"] == 0.91


def test_low_confidence_note_intent_falls_back_to_question_flow() -> None:
    retrieval = FakeRetrievalService()
    note = FakeNoteService()
    orchestrator = ChatOrchestrator(
        conn=object(),
        settings=Settings(intent_confidence_threshold=0.65),
        intent_service=FakeIntentService(IntentClassification(intent="note", confidence=0.4)),
        retrieval_service=retrieval,
        generation_service=FakeGenerationService(),
        note_service=note,
    )

    response = orchestrator.handle(ChatRequest(question="Maybe save this?"))

    assert response.intent == "question"
    assert response.intent_confidence == 0.4
    assert retrieval.calls == ["Maybe save this?"]
    assert note.calls == []


class FakeIntentService:
    def __init__(self, classification: IntentClassification) -> None:
        self.classification = classification

    def classify(self, _user_input: str) -> IntentClassification:
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

    def answer(self, question, chunks, trace_id=None) -> ChatResponse:
        self.calls.append((question, trace_id))
        return ChatResponse(
            answer="Answer.",
            citations=[],
            retrieved_chunks=chunks,
            prompt_version="test",
            trace_id=trace_id,
        )


class FakeNoteService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def process(self, **kwargs) -> ChatResponse:
        self.calls.append(kwargs)
        return ChatResponse(
            answer="Saved note.",
            citations=[],
            retrieved_chunks=[],
            prompt_version="note-test",
            trace_id=kwargs["trace_id"],
            intent="note",
            intent_confidence=kwargs["intent_confidence"],
            note_status="saved",
            note="Saved note.",
            original_note=kwargs["original_input"],
            note_work="Hamlet",
            note_chunk_ids=["chunk-1"],
        )
