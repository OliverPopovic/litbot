import json
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

from litbot.config import Settings
from litbot.evaluation.notes import (
    NoteEvalCase,
    NoteEvalFailure,
    NoteEvalResult,
    load_note_cases,
    result_to_dict,
    run_note_case,
    run_note_cases,
)


def test_load_note_cases_rejects_unknown_action(tmp_path) -> None:
    path = tmp_path / "note_cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "bad",
                "chunks": [_chunk_row()],
                "steps": [{"action": "dance", "expect": {"note_status": "saved"}}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported note eval action"):
        load_note_cases(path)


def test_load_note_cases_rejects_invalid_grounding_chunk(tmp_path) -> None:
    path = tmp_path / "note_cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "bad",
                "chunks": [_chunk_row()],
                "steps": [
                    {
                        "action": "save",
                        "input": "save this",
                        "grounding": {
                            "should_save": True,
                            "rewritten_note": "Saved.",
                            "inferred_work": "Hamlet",
                            "selected_chunk": "S999",
                        },
                        "expect": {"note_status": "saved"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selected_chunk"):
        load_note_cases(path)


def test_deterministic_save_then_query_sees_same_transaction_note(tmp_path) -> None:
    case = _case(
        "save_then_query",
        [
            {
                "action": "save",
                "input": "note: Hamlet opens with uncertainty.",
                "grounding": {
                    "should_save": True,
                    "rewritten_note": "Hamlet opens with uncertainty.",
                    "inferred_work": "Hamlet",
                    "selected_chunk": "S1",
                },
                "expect": {"note_status": "saved"},
            },
            {
                "action": "query",
                "query": "uncertainty",
                "expect": {
                    "note_query_status": "found",
                    "retrieved_note_count": 1,
                    "note_contains": "Hamlet opens with uncertainty",
                },
            },
        ],
    )
    conn = FakeEvalConnection()

    result = run_note_case(case, conn, Settings())

    assert result.passed
    assert conn.note_rows == []
    assert conn.pending_actions == {}


def test_deterministic_edit_confirm_then_query_sees_updated_note() -> None:
    case = _case(
        "edit_then_query",
        [
            {
                "action": "save",
                "input": "note: Hamlet opens with uncertainty.",
                "grounding": {
                    "should_save": True,
                    "rewritten_note": "Hamlet opens with uncertainty.",
                    "inferred_work": "Hamlet",
                    "selected_chunk": "S1",
                },
                "expect": {"note_status": "saved"},
            },
            {
                "action": "preview_edit",
                "input": "Edit this.",
                "note_text": "Hamlet opens with watchful uncertainty.",
                "target": "active",
                "grounding": {
                    "should_save": True,
                    "rewritten_note": "Hamlet opens with watchful uncertainty.",
                    "inferred_work": "Hamlet",
                    "selected_chunk": "S1",
                },
                "expect": {"note_operation_status": "pending_confirmation"},
            },
            {"action": "confirm_last", "expect": {"note_operation_status": "completed"}},
            {
                "action": "query",
                "query": "watchful uncertainty",
                "expect": {"note_query_status": "found", "note_contains": "watchful uncertainty"},
            },
        ],
    )

    result = run_note_case(case, FakeEvalConnection(), Settings())

    assert result.passed


def test_confirm_last_without_pending_action_fails_before_workflow() -> None:
    case = _case(
        "bad_confirm",
        [{"action": "confirm_last", "expect": {"note_operation_status": "completed"}}],
    )

    result = run_note_case(case, FakeEvalConnection(), Settings())

    assert not result.passed
    assert result.failures[0].reason == "confirm_last requires a previous pending action"


def test_score_and_result_to_dict_report_mismatches() -> None:
    result = NoteEvalResult(
        total=1,
        passed=0,
        failed=1,
        case_results=[
            NoteEvalFailureResult(
                case_id="case-1",
                failures=[NoteEvalFailure("case-1", 1, "save", "boom")],
            )
        ],
    )

    payload = result_to_dict(result)

    assert payload["total"] == 1
    assert payload["failed"] == 1
    assert payload["failures"][0]["errors"][0]["reason"] == "boom"


def test_note_golden_fixture_loads_and_passes() -> None:
    cases = load_note_cases(__import__("pathlib").Path("tests/fixtures/note_golden.jsonl"))

    result = run_note_cases(cases, FakeEvalConnection(), Settings())

    assert result.failed == 0
    assert result.passed == len(cases)


def _case(case_id: str, steps: list[dict]) -> NoteEvalCase:
    row = {
        "id": case_id,
        "description": case_id,
        "chunks": [_chunk_row()],
        "steps": steps,
    }
    return load_note_cases_from_rows([row])[0]


def load_note_cases_from_rows(rows: list[dict]) -> list[NoteEvalCase]:
    from litbot.evaluation.notes import note_case_from_row

    return [note_case_from_row(row) for row in rows]


def _chunk_row() -> dict:
    return {
        "label": "S1",
        "chunk_id": "hamlet:fixture:1",
        "source_id": "hamlet-fixture",
        "text": "Who's there? Barnardo begins Hamlet with uncertainty.",
        "metadata": {"work": "Hamlet", "title": "Hamlet", "author": "William Shakespeare"},
    }


class NoteEvalFailureResult:
    def __init__(self, case_id, failures) -> None:
        self.case_id = case_id
        self.failures = failures
        self.mismatches = []

    @property
    def passed(self) -> bool:
        return False


class FakeRows:
    def __init__(self, rows=None, row=None) -> None:
        self.rows = rows or []
        self.row = row

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.row


class FakeEvalConnection:
    def __init__(self) -> None:
        self.note_rows = []
        self.note_chunk_rows = []
        self.pending_actions = {}

    @contextmanager
    def transaction(self):
        note_snapshot = [dict(row) for row in self.note_rows]
        chunk_snapshot = list(self.note_chunk_rows)
        pending_snapshot = {key: dict(value) for key, value in self.pending_actions.items()}
        try:
            yield
        except Exception:
            self.note_rows = note_snapshot
            self.note_chunk_rows = chunk_snapshot
            self.pending_actions = pending_snapshot
            raise

    def execute(self, query, params=None):
        params = params or []
        if "INSERT INTO notes" in query:
            self.note_rows.append(
                {
                    "note_id": params[0],
                    "original_input": params[1],
                    "rewritten_note": params[2],
                    "inferred_work": params[3],
                    "source_id": params[4],
                    "created_at": datetime.now(UTC),
                }
            )
            return FakeRows()
        is_note_select = "SELECT note_id, original_input, rewritten_note" in query
        if is_note_select and "WHERE note_id = %s" in query:
            note_id = params[0]
            return FakeRows(
                row=next((row for row in self.note_rows if row["note_id"] == note_id), None)
            )
        if is_note_select:
            return FakeRows(rows=list(self.note_rows))
        if "UPDATE notes" in query:
            note_id = params[-1]
            row = next(row for row in self.note_rows if row["note_id"] == note_id)
            row.update(
                {
                    "original_input": params[0],
                    "rewritten_note": params[1],
                    "inferred_work": params[2],
                    "source_id": params[3],
                }
            )
            return FakeRows()
        if "SELECT note_id" in query and "WHERE note_id = ANY" in query:
            target_ids = set(params[0])
            return FakeRows(
                rows=[
                    {"note_id": row["note_id"]}
                    for row in self.note_rows
                    if row["note_id"] in target_ids
                ]
            )
        if "DELETE FROM notes WHERE note_id = ANY" in query:
            target_ids = set(params[0])
            self.note_rows = [row for row in self.note_rows if row["note_id"] not in target_ids]
            return FakeRows()
        if "DELETE FROM note_chunks" in query:
            note_id = params[0]
            self.note_chunk_rows = [row for row in self.note_chunk_rows if row[0] != note_id]
            return FakeRows()
        if "INSERT INTO pending_note_actions" in query:
            action_id, operation, payload, expires_at = params
            self.pending_actions[action_id] = {
                "action_id": action_id,
                "operation": operation,
                "payload": getattr(payload, "obj", payload),
                "expires_at": expires_at,
                "consumed_at": None,
            }
            return FakeRows()
        if "FROM pending_note_actions" in query and "FOR UPDATE" in query:
            return FakeRows(row=self.pending_actions.get(params[0]))
        if "UPDATE pending_note_actions" in query:
            self.pending_actions[params[0]]["consumed_at"] = datetime.now(UTC)
            return FakeRows()
        return FakeRows()

    @contextmanager
    def cursor(self):
        yield self

    def executemany(self, query, rows):
        if "INSERT INTO note_chunks" in query:
            self.note_chunk_rows.extend(rows)

    def force_pending_status(self, action_id: str, status: str) -> None:
        if status == "expired":
            self.pending_actions[action_id]["expires_at"] = datetime(2000, 1, 1, tzinfo=UTC)
        elif status == "consumed":
            self.pending_actions[action_id]["consumed_at"] = datetime.now(UTC)
