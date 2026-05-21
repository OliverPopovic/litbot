import structlog

from litbot.config import Settings, get_settings
from litbot.generation.citations import validate_and_format_labels
from litbot.models import ChatResponse, RetrievedChunk, RetrievedNote
from litbot.notes.grounding import GroundedNote
from litbot.notes.retrieval import NoteRetrievalResult

logger = structlog.get_logger(__name__)


class NoteResponseFactory:
    """Sole construction point for note-related ChatResponse contracts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def saved(
        self,
        *,
        note_id: str,
        grounded: GroundedNote,
        original_input: str,
        trace_id: str,
        intent_confidence: float | None,
    ) -> ChatResponse:
        selected_chunk_ids = [chunk.chunk_id for chunk in grounded.selected_chunks]
        return ChatResponse(
            answer=f"Saved note for {grounded.inferred_work}:\n{grounded.rewritten_note}",
            citations=validate_and_format_labels(
                {chunk.label for chunk in grounded.selected_chunks},
                grounded.retrieved_chunks,
            ),
            retrieved_chunks=grounded.retrieved_chunks,
            prompt_version=self.settings.note_prompt_version,
            trace_id=trace_id,
            citation_map=grounded.citation_map,
            intent="note",
            intent_confidence=intent_confidence,
            note_status="saved",
            note_id=note_id,
            note=grounded.rewritten_note,
            original_note=original_input,
            note_work=grounded.inferred_work,
            note_chunk_ids=selected_chunk_ids,
        )

    def not_saved(
        self,
        *,
        original_input: str,
        note_text: str,
        chunks: list[RetrievedChunk],
        trace_id: str,
        reason: str,
        intent_confidence: float | None,
    ) -> ChatResponse:
        logger.info(
            "note_not_saved",
            trace_id=trace_id,
            rejection_reason=reason,
            chunk_count=len(chunks),
        )
        return ChatResponse(
            answer=f"I did not save that note: {reason}",
            citations=[],
            retrieved_chunks=chunks,
            prompt_version=self.settings.note_prompt_version,
            trace_id=trace_id,
            unsupported=[reason],
            intent="note",
            intent_confidence=intent_confidence,
            note_status="not_saved",
            note=note_text,
            original_note=original_input,
            note_chunk_ids=[],
            note_rejection_reason=reason,
        )

    def note_query(
        self,
        *,
        result: NoteRetrievalResult,
        trace_id: str,
        intent_confidence: float | None,
    ) -> ChatResponse:
        status = "found" if result.notes else "not_found"
        return ChatResponse(
            answer=_note_query_answer(result.notes, has_more=result.has_more),
            citations=[],
            retrieved_chunks=_unique_supporting_chunks(result.notes),
            retrieved_notes=result.notes,
            prompt_version=self.settings.note_prompt_version,
            trace_id=trace_id,
            intent="note_query",
            intent_confidence=intent_confidence,
            note_query_status=status,
            note_query_has_more=result.has_more,
        )

    def operation(
        self,
        *,
        answer: str,
        trace_id: str,
        intent: str | None,
        intent_confidence: float | None,
        operation: str | None,
        status: str,
        target_note_ids: list[str] | None = None,
        pending_note_action_id: str | None = None,
        retrieved_chunks: list[RetrievedChunk] | None = None,
        unsupported: list[str] | None = None,
        note: str | None = None,
        note_work: str | None = None,
        note_chunk_ids: list[str] | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            answer=answer,
            citations=[],
            retrieved_chunks=retrieved_chunks or [],
            prompt_version=self.settings.note_prompt_version,
            trace_id=trace_id,
            unsupported=unsupported or [],
            intent=intent,  # type: ignore[arg-type]
            intent_confidence=intent_confidence,
            note_operation=operation,  # type: ignore[arg-type]
            note_operation_status=status,  # type: ignore[arg-type]
            pending_note_action_id=pending_note_action_id,
            target_note_ids=target_note_ids or [],
            note=note,
            note_work=note_work,
            note_chunk_ids=note_chunk_ids,
        )


def _unique_supporting_chunks(notes: list[RetrievedNote]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    seen: set[str] = set()
    for note in notes:
        for chunk in note.supporting_chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks


def _note_query_answer(notes: list[RetrievedNote], *, has_more: bool) -> str:
    if not notes:
        return "I did not find any saved notes matching that request."
    lines = ["I found these saved notes:"]
    for note in notes:
        lines.append(f"- [{note.label}] {note.inferred_work}: {note.rewritten_note}")
    if has_more:
        lines.append("This is a capped preview; more saved notes exist.")
    return "\n".join(lines)
