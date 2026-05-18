import sys
from contextlib import contextmanager

from typer.testing import CliRunner

from litbot import cli
from litbot.evaluation.retrieval import RetrievalCase
from litbot.models import RetrievedChunk


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
    monkeypatch.setattr(
        cli,
        "configure_logging",
        lambda stream=sys.stdout: log_streams.append(stream),
    )

    result = runner.invoke(cli.app, ["eval-retrieval", "cases.jsonl"])

    assert result.exit_code == 0
    assert getattr(log_streams[-1], "name", None) == "<stderr>"
    assert '"total": 1' in result.output
    assert '"hit_at_1": 1' in result.output


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
