import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from litbot.config import Settings
from litbot.models import ChatResponse, NoteContext, RetrievedChunk, RetrievedNote
from litbot.notes.grounding import GroundedNote
from litbot.notes.repository import NoteRepository
from litbot.notes.retrieval import NoteRetrievalResult
from litbot.notes.workflow import (
    CancelPendingNoteActionCommand,
    ConfirmPendingNoteActionCommand,
    NoteWorkflow,
    PreviewDeleteAllNotesCommand,
    PreviewDeleteNoteCommand,
    PreviewEditNoteCommand,
    QueryNotesCommand,
    SaveNoteCommand,
)

NOTE_EVAL_ACTIONS = {
    "save",
    "query",
    "preview_edit",
    "preview_delete",
    "preview_delete_all",
    "confirm_last",
    "cancel_last",
}
NOTE_EVAL_EXPECTATIONS = {
    "intent",
    "note_status",
    "note_query_status",
    "note_operation",
    "note_operation_status",
    "answer_contains",
    "note_contains",
    "unsupported_contains",
    "saved_note_count",
    "target_note_count",
    "pending_action_present",
    "pending_action_consumed",
    "note_work",
    "note_chunk_count",
    "retrieved_note_count",
}
_DEFAULT_REJECTION = "The note could not be grounded in the corpus."


@dataclass(frozen=True)
class NoteEvalStep:
    action: str
    expect: dict[str, Any]
    input: str | None = None
    query: str | None = None
    note_text: str | None = None
    target: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    top_k: int | None = None
    exact_work: str | None = None
    mode: str | None = None
    grounding: dict[str, Any] | None = None
    pending_status: str | None = None


@dataclass(frozen=True)
class NoteEvalCase:
    id: str
    description: str
    chunks: list[RetrievedChunk]
    steps: list[NoteEvalStep]


@dataclass(frozen=True)
class NoteEvalMismatch:
    case_id: str
    step_index: int
    action: str
    field: str
    expected: Any
    actual: Any


@dataclass(frozen=True)
class NoteEvalFailure:
    case_id: str
    step_index: int
    action: str
    reason: str


@dataclass(frozen=True)
class NoteCaseResult:
    case_id: str
    mismatches: list[NoteEvalMismatch] = field(default_factory=list)
    failures: list[NoteEvalFailure] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches and not self.failures


@dataclass(frozen=True)
class NoteEvalResult:
    total: int
    passed: int
    failed: int
    case_results: list[NoteCaseResult]


def load_note_cases(path: Path) -> list[NoteEvalCase]:
    cases: list[NoteEvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(note_case_from_row(row, line_number=line_number))
    return cases


def note_case_from_row(row: dict[str, Any], *, line_number: int | None = None) -> NoteEvalCase:
    prefix = f"line {line_number}: " if line_number is not None else ""
    case_id = _required_text(row, "id", prefix)
    description = str(row.get("description") or "")
    chunks = [_chunk_from_row(chunk, prefix) for chunk in _required_list(row, "chunks", prefix)]
    if not chunks:
        raise ValueError(f"{prefix}chunks must not be empty")
    labels = {chunk.label for chunk in chunks}
    if len(labels) != len(chunks):
        raise ValueError(f"{prefix}chunk labels must be unique")
    steps = [_step_from_row(step, prefix, labels) for step in _required_list(row, "steps", prefix)]
    if not steps:
        raise ValueError(f"{prefix}steps must not be empty")
    return NoteEvalCase(id=case_id, description=description, chunks=chunks, steps=steps)


def score_note_cases(
    cases: list[NoteEvalCase],
    run_case,
) -> NoteEvalResult:
    case_results = [run_case(case) for case in cases]
    passed = sum(1 for result in case_results if result.passed)
    return NoteEvalResult(
        total=len(case_results),
        passed=passed,
        failed=len(case_results) - passed,
        case_results=case_results,
    )


def run_note_cases(
    cases: list[NoteEvalCase],
    conn,
    settings: Settings,
    *,
    live: bool = False,
) -> NoteEvalResult:
    return score_note_cases(
        cases,
        lambda case: run_note_case(case, conn, settings, live=live),
    )


def run_note_case(
    case: NoteEvalCase,
    conn,
    settings: Settings,
    *,
    live: bool = False,
) -> NoteCaseResult:
    runner = _NoteCaseRunner(case, conn, settings, live=live)
    result: NoteCaseResult | None = None
    try:
        with conn.transaction():
            result = runner.run()
            raise _RollbackNoteEval()
    except _RollbackNoteEval:
        return result or NoteCaseResult(case.id)
    except Exception as exc:
        return NoteCaseResult(
            case.id,
            failures=[
                NoteEvalFailure(
                    case_id=case.id,
                    step_index=0,
                    action="case",
                    reason=str(exc),
                )
            ],
        )


def result_to_dict(result: NoteEvalResult) -> dict[str, Any]:
    return {
        "total": result.total,
        "passed": result.passed,
        "failed": result.failed,
        "failures": [
            {
                "case_id": case_result.case_id,
                "mismatches": [
                    {
                        "step_index": mismatch.step_index,
                        "action": mismatch.action,
                        "field": mismatch.field,
                        "expected": mismatch.expected,
                        "actual": mismatch.actual,
                    }
                    for mismatch in case_result.mismatches
                ],
                "errors": [
                    {
                        "step_index": failure.step_index,
                        "action": failure.action,
                        "reason": failure.reason,
                    }
                    for failure in case_result.failures
                ],
            }
            for case_result in result.case_results
            if not case_result.passed
        ],
    }


class _NoteCaseRunner:
    def __init__(self, case: NoteEvalCase, conn, settings: Settings, *, live: bool) -> None:
        self.case = case
        self.conn = conn
        self.settings = settings
        self.state = _StepState()
        self.eval_note_ids: set[str] = set()
        chunks_by_label = {chunk.label: chunk for chunk in case.chunks}
        self.grounding = None if live else _DeterministicGroundingService(settings, chunks_by_label)
        self.workflow = NoteWorkflow(
            conn,
            settings,
            grounding_service=self.grounding,
            note_repository=_ScopedNoteRepository(conn, settings, self.eval_note_ids),
            note_retrieval_service=(
                None
                if live
                else _DeterministicNoteRetrievalService(conn, case.chunks, self.eval_note_ids)
            ),
        )

    def run(self) -> NoteCaseResult:
        mismatches: list[NoteEvalMismatch] = []
        failures: list[NoteEvalFailure] = []
        self._seed_fixture_chunks()
        for step_index, step in enumerate(self.case.steps, start=1):
            if self.grounding is not None:
                self.grounding.set_step(step)
            if step.pending_status:
                self._force_pending_status(step.pending_status)
            response, failure = self._run_step(step, step_index)
            if failure is not None:
                failures.append(failure)
                continue
            if response is None:
                failures.append(
                    NoteEvalFailure(
                        self.case.id,
                        step_index,
                        step.action,
                        "step returned no response",
                    )
                )
                continue
            mismatches.extend(self._mismatches(step, step_index, response))
            self.state.update(response, step)
        return NoteCaseResult(self.case.id, mismatches=mismatches, failures=failures)

    def _seed_fixture_chunks(self) -> None:
        if hasattr(self.conn, "note_rows"):
            return
        document_ids: dict[str, int] = {}
        for chunk in self.case.chunks:
            document_ids.setdefault(chunk.source_id, self._seed_fixture_document(chunk))
        rows = [
            (
                chunk.chunk_id,
                document_ids[chunk.source_id],
                chunk.source_id,
                index,
                chunk.text,
                len(chunk.text.split()),
                f"note-eval:{self.case.id}:{chunk.chunk_id}",
                _vector_literal([0.0] * self.settings.embedding_dimensions),
                Jsonb(chunk.metadata),
            )
            for index, chunk in enumerate(self.case.chunks, start=1)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, document_id, source_id, chunk_index,
                     text, token_count, chunk_hash, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                rows,
            )

    def _seed_fixture_document(self, chunk: RetrievedChunk) -> int:
        row = self.conn.execute(
            """
            INSERT INTO documents
                (source_id, title, author, language, license, metadata, content_hash)
            VALUES (%s, %s, %s, 'en', 'note-eval-fixture', %s, %s)
            ON CONFLICT (source_id) DO UPDATE
            SET source_id = EXCLUDED.source_id
            RETURNING id
            """,
            [
                chunk.source_id,
                str(chunk.metadata.get("title") or chunk.metadata.get("work") or chunk.source_id),
                chunk.metadata.get("author"),
                Jsonb({"note_eval_fixture": self.case.id, **chunk.metadata}),
                f"note-eval:{chunk.source_id}",
            ],
        ).fetchone()
        return int(dict(row)["id"])

    def _run_step(
        self,
        step: NoteEvalStep,
        step_index: int,
    ) -> tuple[ChatResponse | None, NoteEvalFailure | None]:
        try:
            if step.action == "save":
                return self.workflow.save(
                    SaveNoteCommand(
                        original_input=step.input or "",
                        note_text=step.note_text or step.input or "",
                        filters=step.filters,
                        top_k=step.top_k,
                        trace_id=f"{self.case.id}:{step_index}",
                        intent_confidence=1.0,
                    )
                ), None
            if step.action == "query":
                return self.workflow.query(
                    QueryNotesCommand(
                        query=step.query or step.input or "",
                        filters=step.filters,
                        top_k=step.top_k,
                        exact_work=step.exact_work,
                        mode=step.mode,
                        trace_id=f"{self.case.id}:{step_index}",
                        intent_confidence=1.0,
                    )
                ), None
            if step.action == "preview_edit":
                return self.workflow.preview_edit(
                    PreviewEditNoteCommand(
                        original_input=step.input or "",
                        note_text=step.note_text,
                        target_reference=self._target_reference(step.target),
                        note_context=self.state.note_context(),
                        filters=step.filters,
                        top_k=step.top_k,
                        trace_id=f"{self.case.id}:{step_index}",
                        intent_confidence=1.0,
                    )
                ), None
            if step.action == "preview_delete":
                return self.workflow.preview_delete(
                    PreviewDeleteNoteCommand(
                        target_reference=self._target_reference(step.target),
                        note_context=self.state.note_context(),
                        trace_id=f"{self.case.id}:{step_index}",
                        intent_confidence=1.0,
                    )
                ), None
            if step.action == "preview_delete_all":
                return self.workflow.preview_delete_all(
                    PreviewDeleteAllNotesCommand(
                        trace_id=f"{self.case.id}:{step_index}",
                        intent_confidence=1.0,
                    )
                ), None
            if step.action == "confirm_last":
                if not self.state.last_pending_action_id:
                    return None, NoteEvalFailure(
                        self.case.id,
                        step_index,
                        step.action,
                        "confirm_last requires a previous pending action",
                    )
                return self.workflow.confirm(
                    ConfirmPendingNoteActionCommand(
                        self.state.last_pending_action_id,
                        trace_id=f"{self.case.id}:{step_index}",
                    )
                ), None
            if step.action == "cancel_last":
                if not self.state.last_pending_action_id:
                    return None, NoteEvalFailure(
                        self.case.id,
                        step_index,
                        step.action,
                        "cancel_last requires a previous pending action",
                    )
                return self.workflow.cancel(
                    CancelPendingNoteActionCommand(
                        self.state.last_pending_action_id,
                        trace_id=f"{self.case.id}:{step_index}",
                    )
                ), None
            return None, NoteEvalFailure(self.case.id, step_index, step.action, "unknown action")
        except Exception as exc:
            return None, NoteEvalFailure(self.case.id, step_index, step.action, str(exc))

    def _target_reference(self, target: str | None) -> str | None:
        if target == "active":
            return None
        if target == "last_saved":
            return self.state.last_saved_note_id
        return target

    def _force_pending_status(self, status: str) -> None:
        if not self.state.last_pending_action_id:
            return
        if hasattr(self.conn, "force_pending_status"):
            self.conn.force_pending_status(self.state.last_pending_action_id, status)
            return
        if status == "expired":
            self.conn.execute(
                "UPDATE pending_note_actions SET expires_at = %s WHERE action_id = %s",
                [datetime.now(UTC) - timedelta(minutes=1), self.state.last_pending_action_id],
            )
        elif status == "consumed":
            self.conn.execute(
                "UPDATE pending_note_actions SET consumed_at = now() WHERE action_id = %s",
                [self.state.last_pending_action_id],
            )

    def _mismatches(
        self,
        step: NoteEvalStep,
        step_index: int,
        response: ChatResponse,
    ) -> list[NoteEvalMismatch]:
        mismatches = []
        for field_name, expected in step.expect.items():
            actual = self._actual_value(field_name, response)
            if not _matches_expected(field_name, expected, actual):
                mismatches.append(
                    NoteEvalMismatch(
                        case_id=self.case.id,
                        step_index=step_index,
                        action=step.action,
                        field=field_name,
                        expected=expected,
                        actual=actual,
                    )
                )
        return mismatches

    def _actual_value(self, field_name: str, response: ChatResponse) -> Any:
        if field_name in {
            "intent",
            "note_status",
            "note_query_status",
            "note_operation",
            "note_operation_status",
            "note_work",
        }:
            return getattr(response, field_name)
        if field_name == "answer_contains":
            return response.answer
        if field_name == "note_contains":
            values = [response.note or ""]
            values.extend(note.rewritten_note for note in response.retrieved_notes)
            return "\n".join(values)
        if field_name == "unsupported_contains":
            return "\n".join(response.unsupported)
        if field_name == "saved_note_count":
            return len(_note_rows(self.conn, note_ids=self.eval_note_ids))
        if field_name == "target_note_count":
            return len(response.target_note_ids)
        if field_name == "pending_action_present":
            return bool(response.pending_note_action_id)
        if field_name == "pending_action_consumed":
            return _pending_consumed(self.conn, self.state.last_pending_action_id)
        if field_name == "note_chunk_count":
            return len(response.note_chunk_ids or [])
        if field_name == "retrieved_note_count":
            return len(response.retrieved_notes)
        return None


@dataclass
class _StepState:
    last_response: ChatResponse | None = None
    last_pending_action_id: str | None = None
    last_saved_note_id: str | None = None
    active_note_id: str | None = None
    retrieved_note_ids: list[str] = field(default_factory=list)
    step_outputs_by_id: dict[str, ChatResponse] = field(default_factory=dict)

    def note_context(self) -> NoteContext | None:
        if not self.active_note_id and not self.retrieved_note_ids:
            return None
        return NoteContext(
            active_note_id=self.active_note_id,
            retrieved_note_ids=self.retrieved_note_ids,
        )

    def update(self, response: ChatResponse, step: NoteEvalStep) -> None:
        self.last_response = response
        self.step_outputs_by_id[str(len(self.step_outputs_by_id) + 1)] = response
        if response.pending_note_action_id:
            self.last_pending_action_id = response.pending_note_action_id
        if response.note_id:
            self.last_saved_note_id = response.note_id
            self.active_note_id = response.note_id
            self.retrieved_note_ids = [response.note_id]
        if response.retrieved_notes:
            self.retrieved_note_ids = [note.note_id for note in response.retrieved_notes]
            self.active_note_id = (
                self.retrieved_note_ids[0] if len(self.retrieved_note_ids) == 1 else None
            )
        if response.note_operation_status == "completed" and response.note_operation in {
            "delete",
            "delete_all",
        }:
            self.active_note_id = None
            self.retrieved_note_ids = []


class _DeterministicGroundingService:
    def __init__(self, settings: Settings, chunks_by_label: dict[str, RetrievedChunk]) -> None:
        self.settings = settings
        self.chunks_by_label = chunks_by_label
        self.current_step: NoteEvalStep | None = None

    def set_step(self, step: NoteEvalStep) -> None:
        self.current_step = step

    def prepare(
        self,
        *,
        original_input: str,
        note_text: str,
        filters: dict[str, Any],
        top_k: int | None,
        trace_id: str,
    ) -> GroundedNote:
        grounding = (self.current_step.grounding if self.current_step else None) or {}
        selected_chunks = self._selected_chunks(grounding)
        if not grounding.get("should_save", False):
            return GroundedNote(
                original_input=original_input,
                note_text=note_text,
                retrieved_chunks=selected_chunks,
                rejection_reason=str(grounding.get("rejection_reason") or _DEFAULT_REJECTION),
            )
        return GroundedNote(
            original_input=original_input,
            note_text=note_text,
            rewritten_note=str(grounding.get("rewritten_note") or note_text),
            inferred_work=str(grounding.get("inferred_work") or filters.get("work") or "Unknown"),
            selected_chunks=selected_chunks,
            retrieved_chunks=selected_chunks,
            citation_map=list(grounding.get("citation_map") or []),
            embedding=[0.0] * self.settings.embedding_dimensions,
        )

    def _selected_chunks(self, grounding: dict[str, Any]) -> list[RetrievedChunk]:
        if selected := grounding.get("selected_chunk"):
            chunk = self.chunks_by_label.get(str(selected))
            if chunk is None:
                raise ValueError(f"selected_chunk {selected} was not found in fixture chunks")
            return [chunk]
        if contains := grounding.get("chunk_text_contains"):
            normalized = _normalize(str(contains))
            for chunk in self.chunks_by_label.values():
                if normalized in _normalize(chunk.text):
                    return [chunk]
            raise ValueError(f"chunk_text_contains {contains!r} did not match fixture chunks")
        return []


class _DeterministicNoteRetrievalService:
    def __init__(self, conn, chunks: list[RetrievedChunk], note_ids: set[str]) -> None:
        self.conn = conn
        self.note_ids = note_ids
        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def list_all(
        self,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        exact_work: str | None = None,
    ) -> NoteRetrievalResult:
        notes = self._notes(filters or {}, exact_work=exact_work)
        limit = top_k or len(notes)
        return NoteRetrievalResult(
            notes=notes[:limit],
            has_more=len(notes) > limit,
            match_strategy="deterministic_list_all",
        )

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        exact_work: str | None = None,
        allow_work_fallback: bool = True,
    ) -> NoteRetrievalResult:
        query_text = _normalize(query)
        notes = [
            note
            for note in self._notes(filters or {}, exact_work=exact_work)
            if (
                not query_text
                or query_text in _normalize(note.rewritten_note)
                or query_text in _normalize(note.inferred_work)
                or any(token in _normalize(note.rewritten_note) for token in query_text.split())
            )
        ]
        limit = top_k or len(notes)
        return NoteRetrievalResult(
            notes=notes[:limit],
            has_more=len(notes) > limit,
            match_strategy="deterministic_search",
        )

    def _notes(self, filters: dict[str, Any], *, exact_work: str | None) -> list[RetrievedNote]:
        rows = _note_rows(self.conn, note_ids=self.note_ids)
        rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        notes = []
        for index, row in enumerate(rows, start=1):
            work = str(row.get("inferred_work") or "")
            requested_work = exact_work or filters.get("work")
            if requested_work and work.lower() != str(requested_work).lower():
                continue
            note_id = str(row["note_id"])
            notes.append(
                RetrievedNote(
                    label=f"N{index}",
                    note_id=note_id,
                    rewritten_note=str(row["rewritten_note"]),
                    original_input=str(row["original_input"]),
                    inferred_work=work,
                    matched_work=work,
                    source_id=row.get("source_id"),
                    created_at=_as_datetime(row.get("created_at")),
                    supporting_chunks=self._supporting_chunks(note_id),
                    combined_score=1.0,
                    reason="deterministic note match",
                )
            )
        return notes

    def _supporting_chunks(self, note_id: str) -> list[RetrievedChunk]:
        chunks = []
        for row in _note_chunk_rows(self.conn, note_id):
            chunk_id = str(row["chunk_id"])
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk is not None:
                chunks.append(
                    chunk.model_copy(update={"label": str(row.get("label") or chunk.label)})
                )
        return chunks


class _RollbackNoteEval(Exception):
    pass


class _ScopedNoteRepository(NoteRepository):
    def __init__(self, conn, settings: Settings, note_ids: set[str]) -> None:
        super().__init__(conn, settings)
        self.note_ids = note_ids

    def insert(self, *, note_id: str, grounded: GroundedNote, trace_id: str) -> None:
        super().insert(note_id=note_id, grounded=grounded, trace_id=trace_id)
        self.note_ids.add(note_id)

    def update(self, *, note_id: str, grounded: GroundedNote, trace_id: str) -> None:
        if note_id not in self.note_ids:
            return
        super().update(note_id=note_id, grounded=grounded, trace_id=trace_id)

    def fetch(self, note_id: str) -> dict[str, Any] | None:
        if note_id not in self.note_ids:
            return None
        return super().fetch(note_id)

    def list_all_rows(self) -> list[dict[str, Any]]:
        return [row for row in super().list_all_rows() if str(row["note_id"]) in self.note_ids]

    def existing_ids(self, note_ids: list[str]) -> list[str]:
        scoped_ids = [note_id for note_id in note_ids if note_id in self.note_ids]
        return super().existing_ids(scoped_ids)

    def delete_many(self, note_ids: list[str]) -> None:
        scoped_ids = [note_id for note_id in note_ids if note_id in self.note_ids]
        super().delete_many(scoped_ids)
        for note_id in scoped_ids:
            self.note_ids.discard(note_id)


def _step_from_row(row: dict[str, Any], prefix: str, labels: set[str]) -> NoteEvalStep:
    if not isinstance(row, dict):
        raise ValueError(f"{prefix}each step must be an object")
    action = _required_text(row, "action", prefix)
    if action not in NOTE_EVAL_ACTIONS:
        raise ValueError(f"{prefix}unsupported note eval action: {action}")
    expect = row.get("expect")
    if not isinstance(expect, dict) or not expect:
        raise ValueError(f"{prefix}step expect must be a non-empty object")
    unknown = set(expect) - NOTE_EVAL_EXPECTATIONS
    if unknown:
        raise ValueError(f"{prefix}unsupported expectation fields: {sorted(unknown)}")
    grounding = row.get("grounding")
    if grounding is not None:
        if not isinstance(grounding, dict):
            raise ValueError(f"{prefix}grounding must be an object")
        selected = grounding.get("selected_chunk")
        if selected is not None and str(selected) not in labels:
            raise ValueError(f"{prefix}grounding selected_chunk is not in chunks")
    pending_status = row.get("pending_status")
    if pending_status is not None and pending_status not in {"expired", "consumed"}:
        raise ValueError(f"{prefix}pending_status must be expired or consumed")
    filters = row.get("filters") or {}
    if not isinstance(filters, dict):
        raise ValueError(f"{prefix}filters must be an object")
    top_k = row.get("top_k")
    return NoteEvalStep(
        action=action,
        expect=expect,
        input=row.get("input"),
        query=row.get("query"),
        note_text=row.get("note_text"),
        target=row.get("target"),
        filters=filters,
        top_k=int(top_k) if top_k is not None else None,
        exact_work=row.get("exact_work"),
        mode=row.get("mode"),
        grounding=grounding,
        pending_status=pending_status,
    )


def _chunk_from_row(row: dict[str, Any], prefix: str) -> RetrievedChunk:
    if not isinstance(row, dict):
        raise ValueError(f"{prefix}each chunk must be an object")
    return RetrievedChunk(
        label=_required_text(row, "label", prefix),
        chunk_id=_required_text(row, "chunk_id", prefix),
        source_id=_required_text(row, "source_id", prefix),
        text=_required_text(row, "text", prefix),
        metadata=dict(row.get("metadata") or {}),
        combined_score=float(row.get("combined_score") or 1.0),
        reason=str(row.get("reason") or "fixture"),
    )


def _required_text(row: dict[str, Any], key: str, prefix: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}{key} must be a non-blank string")
    return value.strip()


def _required_list(row: dict[str, Any], key: str, prefix: str) -> list[Any]:
    value = row.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{prefix}{key} must be a list")
    return value


def _matches_expected(field_name: str, expected: Any, actual: Any) -> bool:
    if field_name.endswith("_contains"):
        return _normalize(str(expected)) in _normalize(str(actual))
    return actual == expected


def _note_rows(conn, note_ids: set[str] | None = None) -> list[dict[str, Any]]:
    if hasattr(conn, "note_rows"):
        rows = [dict(row) for row in conn.note_rows]
        if note_ids is not None:
            rows = [row for row in rows if str(row.get("note_id")) in note_ids]
        return rows
    rows = conn.execute(
        """
        SELECT note_id, original_input, rewritten_note, inferred_work, source_id, created_at
        FROM notes
        ORDER BY created_at DESC
        """
    ).fetchall()
    result = [dict(row) for row in rows]
    if note_ids is not None:
        result = [row for row in result if str(row.get("note_id")) in note_ids]
    return result


def _note_chunk_rows(conn, note_id: str) -> list[dict[str, Any]]:
    if hasattr(conn, "note_chunk_rows"):
        rows = []
        for row in conn.note_chunk_rows:
            if row[0] == note_id:
                rows.append(
                    {"note_id": row[0], "chunk_id": row[1], "rank": row[2], "label": row[3]}
                )
        return rows
    rows = conn.execute(
        """
        SELECT note_id, chunk_id, rank, label
        FROM note_chunks
        WHERE note_id = %s
        ORDER BY rank
        """,
        [note_id],
    ).fetchall()
    return [dict(row) for row in rows]


def _pending_consumed(conn, action_id: str | None) -> bool:
    if not action_id:
        return False
    if hasattr(conn, "pending_actions"):
        row = conn.pending_actions.get(action_id)
        return bool(row and row.get("consumed_at") is not None)
    row = conn.execute(
        """
        SELECT consumed_at
        FROM pending_note_actions
        WHERE action_id = %s
        """,
        [action_id],
    ).fetchone()
    return bool(row and dict(row).get("consumed_at") is not None)


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.now(UTC)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"
