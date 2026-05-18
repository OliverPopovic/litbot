import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litbot.models import RetrievedChunk


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    expected_work: str
    filters: dict[str, Any]
    k: int
    expected_source_id: str | None = None
    expected_chunk_text_contains: str | None = None
    expected_metadata_contains: dict[str, Any] | None = None


@dataclass(frozen=True)
class RetrievalMiss:
    question: str
    expected_work: str
    retrieved: list[dict[str, str]]


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    total: int
    hit_at_1: int
    hit_at_3: int
    hit_at_k: int
    mean_reciprocal_rank: float
    misses: list[RetrievalMiss]


RetrieveFn = Callable[[str, dict[str, Any], int], list[RetrievedChunk]]


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    cases: list[RetrievalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(retrieval_case_from_row(row, line_number=line_number))
    return cases


def retrieval_case_from_row(
    row: dict[str, Any],
    *,
    line_number: int | None = None,
) -> RetrievalCase:
    prefix = f"line {line_number}: " if line_number is not None else ""
    matchers = [
        row.get("expected_source_id"),
        row.get("expected_chunk_text_contains"),
        row.get("expected_metadata_contains"),
    ]
    if sum(value is not None for value in matchers) != 1:
        raise ValueError(
            f"{prefix}exactly one expected_source_id, expected_chunk_text_contains, "
            "or expected_metadata_contains is required"
        )
    question = str(row.get("question", "")).strip()
    expected_work = str(row.get("expected_work", "")).strip()
    if not question:
        raise ValueError(f"{prefix}question must not be blank")
    if not expected_work:
        raise ValueError(f"{prefix}expected_work must not be blank")
    k = int(row.get("k", 8))
    if k < 1:
        raise ValueError(f"{prefix}k must be at least 1")

    expected_metadata = row.get("expected_metadata_contains")
    if expected_metadata is not None and not isinstance(expected_metadata, dict):
        raise ValueError(f"{prefix}expected_metadata_contains must be an object")

    return RetrievalCase(
        question=question,
        expected_work=expected_work,
        filters=dict(row.get("filters") or {}),
        k=k,
        expected_source_id=row.get("expected_source_id"),
        expected_chunk_text_contains=row.get("expected_chunk_text_contains"),
        expected_metadata_contains=expected_metadata,
    )


def score_retrieval(cases: list[RetrievalCase], retrieve: RetrieveFn) -> RetrievalEvaluationResult:
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_k = 0
    reciprocal_rank_total = 0.0
    misses: list[RetrievalMiss] = []

    for case in cases:
        chunks = retrieve(case.question, case.filters, case.k)
        rank = _first_match_rank(case, chunks)
        if rank is None:
            misses.append(
                RetrievalMiss(
                    question=case.question,
                    expected_work=case.expected_work,
                    retrieved=[_retrieved_summary(chunk) for chunk in chunks],
                )
            )
            continue

        reciprocal_rank_total += 1 / rank
        if rank <= 1:
            hit_at_1 += 1
        if rank <= 3:
            hit_at_3 += 1
        if rank <= case.k:
            hit_at_k += 1

    total = len(cases)
    return RetrievalEvaluationResult(
        total=total,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_k=hit_at_k,
        mean_reciprocal_rank=reciprocal_rank_total / total if total else 0.0,
        misses=misses,
    )


def result_to_dict(result: RetrievalEvaluationResult) -> dict[str, Any]:
    return {
        "total": result.total,
        "hit_at_1": result.hit_at_1,
        "hit_at_3": result.hit_at_3,
        "hit_at_k": result.hit_at_k,
        "mean_reciprocal_rank": result.mean_reciprocal_rank,
        "misses": [
            {
                "question": miss.question,
                "expected_work": miss.expected_work,
                "retrieved": miss.retrieved,
            }
            for miss in result.misses
        ],
    }


def _first_match_rank(case: RetrievalCase, chunks: list[RetrievedChunk]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if _matches_case(case, chunk):
            return rank
    return None


def _matches_case(case: RetrievalCase, chunk: RetrievedChunk) -> bool:
    if chunk.metadata.get("work") != case.expected_work:
        return False
    if case.expected_source_id is not None:
        return chunk.source_id == case.expected_source_id
    if case.expected_chunk_text_contains is not None:
        expected = _normalize_match_text(case.expected_chunk_text_contains)
        actual = _normalize_match_text(chunk.text)
        return expected in actual
    if case.expected_metadata_contains is not None:
        return all(
            chunk.metadata.get(key) == value
            for key, value in case.expected_metadata_contains.items()
        )
    return False


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _retrieved_summary(chunk: RetrievedChunk) -> dict[str, str]:
    return {
        "label": chunk.label,
        "work": str(chunk.metadata.get("work", "")),
        "source_id": chunk.source_id,
        "chunk_id": chunk.chunk_id,
    }
