import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litbot.generation.citations import CITATION_RE


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    answered: int
    cited: int
    unsupported: int

    @property
    def citation_rate(self) -> float:
        return self.cited / self.total if self.total else 0.0


def load_golden_questions(path: Path) -> list[dict[str, Any]]:
    """Load JSONL golden questions for lightweight regression evaluation."""

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_answers(rows: list[dict[str, Any]]) -> EvaluationResult:
    """Score exported answers for basic answerability and citation presence."""

    answered = 0
    cited = 0
    unsupported = 0
    for row in rows:
        answer = str(row.get("answer", ""))
        if answer.strip():
            answered += 1
        if CITATION_RE.search(answer):
            cited += 1
        if row.get("unsupported"):
            unsupported += 1
    return EvaluationResult(
        total=len(rows),
        answered=answered,
        cited=cited,
        unsupported=unsupported,
    )
