import json
import sys
from contextlib import contextmanager
from datetime import UTC, datetime

from typer.testing import CliRunner

from litbot import cli
from litbot.evaluation.notes import NoteCaseResult, NoteEvalResult
from litbot.evaluation.retrieval import RetrievalCase
from litbot.models import ChatResponse, Citation, RetrievedChunk, RetrievedNote


def test_eval_retrieval_cli_scores_with_mocked_retriever(monkeypatch) -> None:
    runner = CliRunner()
    case = RetrievalCase(
        question="Find Hamlet",
        expected_work="Hamlet",
        filters={"work": "Hamlet"},
        k=3,
        expected_source_id="hamlet",
    )

    monkeypatch.setattr(cli, "load_retrieval_cases", lambda path: [case])
    monkeypatch.setattr(cli, "get_settings", lambda: object())
    monkeypatch.setattr(cli, "get_connection", lambda settings: _fake_connection())
    monkeypatch.setattr(cli, "close_pool", lambda: None)
    monkeypatch.setattr(cli, "RetrievalService", FakeRetrievalService)
    log_streams = []
    log_renderers = []
    monkeypatch.setattr(
        cli,
        "configure_logging",
        lambda stream=sys.stdout, renderer="json": (
            log_streams.append(stream),
            log_renderers.append(renderer),
        ),
    )

    result = runner.invoke(cli.app, ["eval-retrieval", "cases.jsonl", "--json"])

    assert result.exit_code == 0
    assert getattr(log_streams[-1], "name", None) == "<stderr>"
    assert log_renderers[-1] == "console"
    assert '"total": 1' in result.output
    assert '"hit_at_1": 1' in result.output


def test_eval_notes_cli_scores_with_mocked_runner(monkeypatch) -> None:
    runner = CliRunner()
    calls = []
    note_result = NoteEvalResult(
        total=1,
        passed=1,
        failed=0,
        case_results=[NoteCaseResult(case_id="case-1")],
    )

    monkeypatch.setattr(cli, "load_note_cases", lambda path: ["case"])
    monkeypatch.setattr(cli, "get_settings", lambda: object())
    monkeypatch.setattr(cli, "get_connection", lambda settings: _fake_connection())
    monkeypatch.setattr(cli, "close_pool", lambda: None)
    monkeypatch.setattr(
        cli,
        "run_note_cases",
        lambda cases, conn, settings, live=False: calls.append(live) or note_result,
    )
    monkeypatch.setattr(
        cli,
        "configure_logging",
        lambda stream=sys.stderr, renderer="console": None,
    )

    result = runner.invoke(cli.app, ["eval-notes", "cases.jsonl", "--json"])

    assert result.exit_code == 0
    assert calls == [False]
    assert '"total": 1' in result.output
    assert '"failed": 0' in result.output


def test_eval_notes_live_mode_is_explicit_and_fails_nonzero(monkeypatch) -> None:
    runner = CliRunner()
    calls = []
    note_result = NoteEvalResult(
        total=1,
        passed=0,
        failed=1,
        case_results=[NoteCaseResult(case_id="case-1", failures=[])],
    )

    monkeypatch.setattr(cli, "load_note_cases", lambda path: ["case"])
    monkeypatch.setattr(cli, "get_settings", lambda: object())
    monkeypatch.setattr(cli, "get_connection", lambda settings: _fake_connection())
    monkeypatch.setattr(cli, "close_pool", lambda: None)
    monkeypatch.setattr(
        cli,
        "run_note_cases",
        lambda cases, conn, settings, live=False: calls.append(live) or note_result,
    )
    monkeypatch.setattr(
        cli,
        "configure_logging",
        lambda stream=sys.stderr, renderer="console": None,
    )

    result = runner.invoke(cli.app, ["eval-notes", "cases.jsonl", "--json", "--live"])

    assert result.exit_code == 1
    assert calls == [True]
    assert '"failed": 1' in result.output


def test_ask_cli_renders_readable_answer(monkeypatch) -> None:
    runner = CliRunner()
    response = ChatResponse(
        answer="Austen frames the party as a scene of expectation. [S1]",
        citations=[
            Citation(
                label="S1",
                source_id="pride",
                chunk_id="pride:00001",
                reference="Pride and Prejudice, chapter 1",
            )
        ],
        retrieved_chunks=[
            RetrievedChunk(
                label="S1",
                chunk_id="pride:00001",
                source_id="pride",
                text="A compact piece of supporting context.",
                metadata={"work": "Pride and Prejudice"},
                combined_score=0.95,
                reason="test",
            )
        ],
        prompt_version="test",
        trace_id="trace-1",
    )

    monkeypatch.setattr(cli, "get_settings", lambda: object())
    monkeypatch.setattr(cli, "get_connection", lambda settings: _fake_connection())
    monkeypatch.setattr(cli, "close_pool", lambda: None)
    monkeypatch.setattr(
        cli,
        "handle_chat_request",
        lambda conn, settings, request: response,
    )
    monkeypatch.setattr(
        cli,
        "configure_logging",
        lambda stream=sys.stderr, renderer="console": None,
    )

    result = runner.invoke(cli.app, ["ask", "What happens?"])

    assert result.exit_code == 0
    assert "Answer" in result.output
    assert "Citations" in result.output
    assert "Retrieved Context" in result.output
    assert "trace-1" in result.output
    assert '"answer"' not in result.output


def test_ask_cli_renders_saved_note(monkeypatch) -> None:
    runner = CliRunner()
    response = ChatResponse(
        answer="Saved note for Hamlet:\nHamlet opens with watchful uncertainty.",
        citations=[],
        retrieved_chunks=[
            RetrievedChunk(
                label="S1",
                chunk_id="hamlet:00001",
                source_id="hamlet",
                text="Who's there?",
                metadata={"work": "Hamlet"},
                combined_score=0.95,
                reason="test",
            )
        ],
        prompt_version="note-test",
        trace_id="trace-note",
        intent="note",
        intent_confidence=0.9,
        note_status="saved",
        note_id="note-1",
        note="Hamlet opens with watchful uncertainty.",
        original_note="Save this: Hamlet starts uncertain.",
        note_work="Hamlet",
        note_chunk_ids=["hamlet:00001"],
    )

    monkeypatch.setattr(cli, "get_settings", lambda: object())
    monkeypatch.setattr(cli, "get_connection", lambda settings: _fake_connection())
    monkeypatch.setattr(cli, "close_pool", lambda: None)
    monkeypatch.setattr(
        cli,
        "handle_chat_request",
        lambda conn, settings, request: response,
    )
    monkeypatch.setattr(
        cli,
        "configure_logging",
        lambda stream=sys.stderr, renderer="console": None,
    )

    result = runner.invoke(cli.app, ["ask", "Save this: Hamlet starts uncertain."])

    assert result.exit_code == 0
    assert "Note Saved" in result.output
    assert "Hamlet opens with watchful uncertainty." in result.output
    assert "Original Input" in result.output
    assert "hamlet:00001" in result.output


def test_ask_cli_renders_retrieved_notes(monkeypatch) -> None:
    runner = CliRunner()
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
                created_at=datetime.now(UTC),
                combined_score=1.0,
                reason="test",
            )
        ],
        prompt_version="note-test",
        trace_id="trace-note-query",
        intent="note_query",
        intent_confidence=0.9,
        note_query_status="found",
        note_query_has_more=False,
    )

    monkeypatch.setattr(cli, "get_settings", lambda: object())
    monkeypatch.setattr(cli, "get_connection", lambda settings: _fake_connection())
    monkeypatch.setattr(cli, "close_pool", lambda: None)
    monkeypatch.setattr(
        cli,
        "handle_chat_request",
        lambda conn, settings, request: response,
    )
    monkeypatch.setattr(
        cli,
        "configure_logging",
        lambda stream=sys.stderr, renderer="console": None,
    )

    result = runner.invoke(cli.app, ["ask", "What notes did I make for Hamlet?"])

    assert result.exit_code == 0
    assert "Notes Found" in result.output
    assert "Stored Notes" in result.output
    assert "Hamlet begins with uncertainty." in result.output


def test_cli_state_missing_or_malformed_returns_no_context(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv(cli.CLI_STATE_ENV, str(state_path))

    assert cli._load_cli_note_context() is None


def test_cli_state_writes_note_context_atomically(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setenv(cli.CLI_STATE_ENV, str(state_path))
    response = ChatResponse(
        answer="I found these saved notes.",
        citations=[],
        retrieved_chunks=[],
        retrieved_notes=[
            RetrievedNote(
                label="N1",
                note_id="note-1",
                rewritten_note="Hamlet begins with uncertainty.",
                original_input="Save this.",
                inferred_work="Hamlet",
                created_at=datetime.now(UTC),
                combined_score=1.0,
                reason="test",
            )
        ],
        prompt_version="note-test",
        trace_id="trace-note-query",
    )

    cli._save_cli_note_context(response)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["note_context"]["active_note_id"] == "note-1"
    assert payload["note_context"]["retrieved_note_ids"] == ["note-1"]
    assert not state_path.with_name("state.json.tmp").exists()


def test_cli_state_clears_context_after_delete_completion(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setenv(cli.CLI_STATE_ENV, str(state_path))
    response = ChatResponse(
        answer="Deleted 1 saved note.",
        citations=[],
        retrieved_chunks=[],
        prompt_version="note-test",
        trace_id="trace-1",
        note_operation="delete",
        note_operation_status="completed",
        target_note_ids=["note-1"],
    )

    cli._save_cli_note_context(response)

    assert cli._load_cli_note_context() is None


def test_cli_state_clears_context_after_empty_note_query(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setenv(cli.CLI_STATE_ENV, str(state_path))
    response = ChatResponse(
        answer="I did not find any saved notes matching that request.",
        citations=[],
        retrieved_chunks=[],
        prompt_version="note-test",
        trace_id="trace-1",
        intent="note_query",
        note_query_status="not_found",
        note_query_has_more=False,
    )

    cli._save_cli_note_context(response)

    assert cli._load_cli_note_context() is None


class FakeRetrievalService:
    def __init__(self, conn, settings) -> None:
        pass

    def retrieve(self, question, filters=None, top_k=None) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                label="S1",
                chunk_id="hamlet:00001",
                source_id="hamlet",
                text="Hamlet evidence.",
                metadata={"work": "Hamlet"},
                combined_score=1.0,
                reason="test",
            )
        ]


@contextmanager
def _fake_connection():
    yield object()
