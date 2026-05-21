from dataclasses import dataclass
from uuid import UUID

from litbot.models import NoteContext


@dataclass(frozen=True)
class TargetResolution:
    note_id: str | None
    status: str | None = None
    reason: str | None = None

    @property
    def failed(self) -> bool:
        return self.status is not None


class NoteTargetResolver:
    """Resolve explicit and contextual note references into stable note IDs."""

    def resolve(
        self,
        target_reference: str | None,
        note_context: NoteContext | None,
    ) -> TargetResolution:
        target_reference = _clean_text(target_reference)
        if target_reference:
            label_index = _note_label_index(target_reference)
            if label_index is not None:
                note_ids = note_context.retrieved_note_ids if note_context else []
                if 0 <= label_index < len(note_ids):
                    return _resolved_note_id(note_ids[label_index])
                return TargetResolution(
                    None,
                    "ambiguous",
                    "I could not match that note label to the current notes.",
                )
            return _resolved_note_id(target_reference)
        if note_context is None or not note_context.active_note_id:
            return TargetResolution(None, "ambiguous", "Please specify which saved note to use.")
        return _resolved_note_id(note_context.active_note_id)


def _note_label_index(value: str) -> int | None:
    candidate = value.strip()
    if len(candidate) < 2 or candidate[0].lower() != "n":
        return None
    suffix = candidate[1:]
    if not suffix.isdigit():
        return None
    return int(suffix) - 1


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolved_note_id(value: str) -> TargetResolution:
    cleaned = _clean_text(value)
    if cleaned is None:
        return TargetResolution(None, "ambiguous", "Please specify which saved note to use.")
    note_id = _canonical_uuid(cleaned)
    if note_id is None:
        return TargetResolution(cleaned, "not_found", "I could not find that saved note.")
    return TargetResolution(note_id)


def _canonical_uuid(value: str) -> str | None:
    try:
        return str(UUID(value))
    except ValueError:
        return None
