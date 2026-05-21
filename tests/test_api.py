from datetime import UTC, datetime

from fastapi.testclient import TestClient

from litbot.api.main import app
from litbot.models import ChatRequest, ChatResponse, RetrievedNote


def test_health_endpoint_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ui_root_returns_html() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="app"' in response.text
    assert "/ui/static/app.js" in response.text
    assert "/ui/static/styles.css" in response.text
    assert "Developer" in response.text


def test_ui_static_assets_are_served() -> None:
    with TestClient(app) as client:
        script = client.get("/ui/static/app.js")
        styles = client.get("/ui/static/styles.css")

    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert "fetch(\"/chat\"" in script.text
    assert styles.status_code == 200
    assert "text/css" in styles.headers["content-type"]
    assert "[data-theme=\"dark\"]" in styles.text


def test_chat_route_remains_registered() -> None:
    routes = {
        (route.path, tuple(sorted(route.methods or [])))
        for route in app.routes
        if hasattr(route, "methods")
    }

    assert ("/chat", ("POST",)) in routes


def test_chat_request_accepts_camel_case_ui_note_context() -> None:
    request = ChatRequest.model_validate(
        {
            "question": "Delete this",
            "noteContext": {
                "activeNoteId": "note-1",
                "retrievedNoteIds": ["note-1"],
            },
            "pendingNoteActionId": "action-1",
            "confirmNoteAction": True,
        }
    )

    assert request.note_context.active_note_id == "note-1"
    assert request.note_context.retrieved_note_ids == ["note-1"]
    assert request.pending_note_action_id == "action-1"
    assert request.confirm_note_action is True


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


def test_chat_note_query_response_serializes_retrieved_notes(monkeypatch) -> None:
    response = ChatResponse(
        answer="I found these saved notes:\n- [N1] Hamlet: Hamlet begins with uncertainty.",
        citations=[],
        retrieved_chunks=[],
        retrieved_notes=[
            RetrievedNote(
                label="N1",
                note_id="note-1",
                rewritten_note="Hamlet begins with uncertainty.",
                original_input="Save this.",
                inferred_work="Hamlet",
                matched_work="Hamlet",
                created_at=datetime.now(UTC),
                combined_score=1.0,
                reason="test",
            )
        ],
        prompt_version="note-test",
        trace_id="trace-1",
        intent="note_query",
        intent_confidence=0.9,
        note_query_status="found",
        note_query_has_more=False,
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
            json={"question": "What notes did I make for Hamlet?"},
            headers={"x-trace-id": "trace-1"},
        )

    payload = result.json()
    assert result.status_code == 200
    assert payload["intent"] == "note_query"
    assert payload["note_query_status"] == "found"
    assert payload["retrieved_notes"][0]["rewritten_note"] == "Hamlet begins with uncertainty."


class _FakeConnection:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_connection():
    return _FakeConnection()
