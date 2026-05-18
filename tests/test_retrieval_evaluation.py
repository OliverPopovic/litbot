import json

import pytest

from litbot.evaluation.retrieval import (
    RetrievalCase,
    load_retrieval_cases,
    result_to_dict,
    score_retrieval,
)
from litbot.models import RetrievedChunk


def test_load_retrieval_cases_validates_exactly_one_matcher(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Where is the answer?",
                "expected_work": "Test Work",
                "expected_source_id": "source",
                "expected_chunk_text_contains": "answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one"):
        load_retrieval_cases(path)


def test_score_retrieval_reports_hit_rates_mrr_and_misses() -> None:
    cases = [
        RetrievalCase(
            question="Who speaks?",
            expected_work="Hamlet",
            filters={"work": "Hamlet"},
            k=3,
            expected_chunk_text_contains="To be",
        ),
        RetrievalCase(
            question="Who is missing?",
            expected_work="Moby-Dick",
            filters={},
            k=3,
            expected_source_id="moby-dick-1851",
        ),
    ]

    def retrieve(question: str, filters: dict, k: int) -> list[RetrievedChunk]:
        if question == "Who speaks?":
            return [
                _chunk("S1", "hamlet", "Hamlet", "Unrelated"),
                _chunk("S2", "hamlet", "Hamlet", "To be, or not to be"),
            ]
        return [_chunk("S1", "hamlet", "Hamlet", "Wrong work")]

    result = score_retrieval(cases, retrieve)

    assert result.total == 2
    assert result.hit_at_1 == 0
    assert result.hit_at_3 == 1
    assert result.hit_at_k == 1
    assert result.mean_reciprocal_rank == pytest.approx(0.25)
    assert result.misses[0].question == "Who is missing?"
    assert result_to_dict(result)["misses"][0]["retrieved"][0]["work"] == "Hamlet"


def test_score_retrieval_normalizes_whitespace_for_text_matches() -> None:
    case = RetrievalCase(
        question="What should Mr. Bennet do?",
        expected_work="Pride and Prejudice",
        filters={},
        k=1,
        expected_chunk_text_contains="you must visit him as soon as he comes",
    )

    result = score_retrieval(
        [case],
        lambda question, filters, k: [
            _chunk(
                "S1",
                "pride-prejudice-1813",
                "Pride and Prejudice",
                "you must visit him as\nsoon as he comes",
            )
        ],
    )

    assert result.hit_at_1 == 1
    assert result.misses == []


def _chunk(label: str, source_id: str, work: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        label=label,
        chunk_id=f"{source_id}:00001",
        source_id=source_id,
        text=text,
        metadata={"work": work},
        combined_score=1.0,
        reason="test",
    )
