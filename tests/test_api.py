from fastapi.testclient import TestClient

from litbot.api.main import app
from litbot.models import ChatResponse


def test_health_endpoint_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_route_remains_registered() -> None:
    routes = {
        (route.path, tuple(sorted(route.methods or [])))
        for route in app.routes
        if hasattr(route, "methods")
    }

    assert ("/chat", ("POST",)) in routes


def test_chat_request_rejects_blank_question_before_runtime_services() -> None:
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 422


def test_chat_saved_note_response_serializes_note_fields(monkeypatch) -> None:
    response = ChatResponse(
        answer="Saved note for Hamlet:\nHamlet opens with watchful uncertainty.",
        citations=[],
        retrieved_chunks=[],
        prompt_version="note-test",
        trace_id="trace-1",
        intent="note",
        intent_confidence=0.9,
        note_status="saved",
        note_id="note-1",
        note="Hamlet opens with watchful uncertainty.",
        original_note="Save this note.",
        note_work="Hamlet",
        note_chunk_ids=["hamlet:00001"],
    )

    monkeypatch.setattr("litbot.api.main.get_settings", lambda: object())
    monkeypatch.setattr("litbot.api.main.get_connection", lambda settings: _fake_connection())
    monkeypatch.setattr(
        "litbot.api.main.handle_chat_request",
        lambda conn, settings, request, trace_id=None: response,
    )

    with TestClient(app) as client:
        result = client.post(
            "/chat",
            json={"question": "Save this note."},
            headers={"x-trace-id": "trace-1"},
        )

    payload = result.json()
    assert result.status_code == 200
    assert payload["note_status"] == "saved"
    assert payload["note"] == "Hamlet opens with watchful uncertainty."
    assert payload["original_note"] == "Save this note."
    assert payload["note_work"] == "Hamlet"
    assert payload["note_chunk_ids"] == ["hamlet:00001"]


class _FakeConnection:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_connection():
    return _FakeConnection()
