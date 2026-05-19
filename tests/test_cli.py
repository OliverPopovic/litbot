import sys
from contextlib import contextmanager

from typer.testing import CliRunner

from litbot import cli
from litbot.evaluation.retrieval import RetrievalCase
from litbot.models import ChatResponse, Citation, RetrievedChunk


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
    monkeypatch.setattr(cli, "RetrievalService", FakeRetrievalService)
    monkeypatch.setattr(cli, "GenerationService", lambda settings: FakeGenerationService(response))
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


class FakeGenerationService:
    def __init__(self, response: ChatResponse) -> None:
        self.response = response

    def answer(self, question, chunks) -> ChatResponse:
        return self.response


@contextmanager
def _fake_connection():
    yield object()
