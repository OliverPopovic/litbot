import json
import re
from datetime import UTC, datetime
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from psycopg import Connection
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litbot.config import Settings, get_settings
from litbot.langchain import embed_query, make_chat_model
from litbot.models import RetrievedChunk, RetrievedNote

logger = structlog.get_logger(__name__)

NOTE_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

RELEVANCE_SYSTEM_PROMPT = """
You decide which saved reading notes are directly relevant to a user's exact literary question.
Keep only notes that address the question's specific topic, relationship, event, image, or claim.
Reject notes that merely share the same work, author, character, or broad theme.
""".strip()

RELEVANCE_DEVELOPER_PROMPT = """
Return valid structured data with relevant_note_ids only. Use only note_id values from the payload.
If unsure, omit the note.
""".strip()

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RELEVANCE_SYSTEM_PROMPT),
        ("system", RELEVANCE_DEVELOPER_PROMPT),
        ("human", "{user_payload}"),
    ]
)


class NoteRetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: list[RetrievedNote] = Field(default_factory=list)
    has_more: bool = False
    match_strategy: str | None = None


class NoteRelevancePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant_note_ids: list[str] = Field(default_factory=list)


class NoteRetrievalService:
    """Retrieve saved reading notes and their linked corpus evidence."""

    def __init__(self, conn: Connection, settings: Settings | None = None) -> None:
        self.conn = conn
        self.settings = settings or get_settings()

    def list_all(
        self,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        exact_work: str | None = None,
    ) -> NoteRetrievalResult:
        limit = _normalize_top_k(top_k, self.settings.note_query_top_k)
        filters = dict(filters or {})
        where, params = _note_filter_clause(
            filters,
            exact_work=_clean_text(exact_work or filters.get("work")),
        )
        prefix = f"WHERE {where}" if where else ""
        rows = self.conn.execute(
            f"""
            SELECT note_id, original_input, rewritten_note, inferred_work, source_id,
                   created_at
            FROM notes
            {prefix}
            ORDER BY created_at DESC
            LIMIT %s
            """,
            [*params, limit + 1],
        ).fetchall()
        has_more = len(rows) > limit
        notes = self._notes_from_rows([dict(row) for row in rows[:limit]], reason="recent note")
        _label_notes(notes)
        return NoteRetrievalResult(notes=notes, has_more=has_more, match_strategy="list_all")

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        exact_work: str | None = None,
        allow_work_fallback: bool = True,
    ) -> NoteRetrievalResult:
        filters = dict(filters or {})
        limit = _normalize_top_k(top_k, self.settings.note_query_top_k)
        exact_work = _clean_text(exact_work or filters.get("work"))
        if exact_work:
            exact_rows = self._hybrid_rows(
                query,
                filters=filters,
                limit=self.settings.note_candidate_top_k,
                exact_work=exact_work,
            )
            exact_notes = self._ranked_notes(exact_rows, limit)
            if exact_notes:
                for note in exact_notes:
                    note.matched_work = exact_work
                logger.info("note_search_matched_work", work=exact_work, strategy="exact")
                return NoteRetrievalResult(
                    notes=exact_notes,
                    has_more=len(exact_rows) > limit,
                    match_strategy="exact_work",
                )
            if not allow_work_fallback:
                return NoteRetrievalResult(match_strategy="exact_work")

        rows = self._hybrid_rows(
            query,
            filters=filters,
            limit=self.settings.note_candidate_top_k,
            exact_work=None if exact_work else _clean_text(filters.get("work")),
        )
        rows = [row for row in rows if self.passes_similarity_gate(row)]
        notes = self._ranked_notes(rows, limit)
        if exact_work:
            for note in notes:
                note.matched_work = note.inferred_work
            if notes:
                logger.info(
                    "note_search_matched_work",
                    requested_work=exact_work,
                    matched_work=notes[0].matched_work,
                    strategy="fallback",
                )
        return NoteRetrievalResult(
            notes=notes,
            has_more=len(rows) > limit,
            match_strategy="fallback" if exact_work else "hybrid",
        )

    def candidates_for_question(
        self,
        question: str,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedNote]:
        rows = self._hybrid_rows(
            question,
            filters=dict(filters or {}),
            limit=self.settings.note_candidate_top_k,
            exact_work=_clean_text((filters or {}).get("work")),
        )
        rows = [row for row in rows if self.passes_similarity_gate(row)]
        return self._ranked_notes(rows, self.settings.note_candidate_top_k)

    def passes_similarity_gate(self, row_or_note: dict[str, Any] | RetrievedNote) -> bool:
        vector_score = _score(row_or_note, "vector_score")
        lexical_score = _score(row_or_note, "lexical_score")
        trigram_score = _score(row_or_note, "trigram_score")
        return (
            vector_score >= self.settings.note_min_vector_score
            or lexical_score > 0
            or trigram_score >= self.settings.note_min_trigram_score
        )

    def _hybrid_rows(
        self,
        query: str,
        *,
        filters: dict[str, Any],
        limit: int,
        exact_work: str | None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        vector = _vector_literal(embed_query(query, self.settings))
        vector_rows = self._vector_search(vector, filters, limit, exact_work)
        lexical_rows = self._lexical_search(_clean_lexical_query(query), filters, limit, exact_work)
        trigram_rows = self._trigram_search(query, filters, limit, exact_work)
        return _merge_rows(vector_rows, lexical_rows, trigram_rows, self.settings.retrieval_rrf_k)

    def _vector_search(
        self,
        vector: str,
        filters: dict[str, Any],
        limit: int,
        exact_work: str | None,
    ) -> list[dict[str, Any]]:
        where, params = _note_filter_clause(filters, exact_work=exact_work)
        prefix = f"WHERE {where}" if where else ""
        rows = self.conn.execute(
            f"""
            SELECT note_id, original_input, rewritten_note, inferred_work, source_id,
                   created_at, 1 - (embedding <=> %s::vector) AS vector_score
            FROM notes
            {prefix}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            [vector, *params, vector, limit],
        ).fetchall()
        return [dict(row) for row in rows]

    def _lexical_search(
        self,
        query: str,
        filters: dict[str, Any],
        limit: int,
        exact_work: str | None,
    ) -> list[dict[str, Any]]:
        where, params = _note_filter_clause(filters, exact_work=exact_work)
        prefix = f"AND {where}" if where else ""
        rows = self.conn.execute(
            f"""
            SELECT note_id, original_input, rewritten_note, inferred_work, source_id,
                   created_at,
                   ts_rank_cd(to_tsvector('english', rewritten_note),
                              plainto_tsquery('english', %s)) AS lexical_score
            FROM notes
            WHERE to_tsvector('english', rewritten_note) @@ plainto_tsquery('english', %s)
              {prefix}
            ORDER BY lexical_score DESC
            LIMIT %s
            """,
            [query, query, *params, limit],
        ).fetchall()
        return [dict(row) for row in rows]

    def _trigram_search(
        self,
        query: str,
        filters: dict[str, Any],
        limit: int,
        exact_work: str | None,
    ) -> list[dict[str, Any]]:
        where, params = _note_filter_clause(filters, exact_work=exact_work)
        prefix = f"AND {where}" if where else ""
        rows = self.conn.execute(
            f"""
            SELECT note_id, original_input, rewritten_note, inferred_work, source_id,
                   created_at, word_similarity(%s, rewritten_note) AS trigram_score
            FROM notes
            WHERE word_similarity(%s, rewritten_note) > 0
              {prefix}
            ORDER BY trigram_score DESC
            LIMIT %s
            """,
            [query, query, *params, limit],
        ).fetchall()
        return [dict(row) for row in rows]

    def _ranked_notes(self, rows: list[dict[str, Any]], limit: int) -> list[RetrievedNote]:
        notes = self._notes_from_rows(rows[:limit], reason="")
        _label_notes(notes)
        return notes

    def _notes_from_rows(self, rows: list[dict[str, Any]], reason: str) -> list[RetrievedNote]:
        chunks_by_note = self._supporting_chunks([str(row["note_id"]) for row in rows])
        notes: list[RetrievedNote] = []
        for row in rows:
            row_reason = reason or _match_reason(row)
            note_id = str(row["note_id"])
            notes.append(
                RetrievedNote(
                    label="",
                    note_id=note_id,
                    rewritten_note=str(row["rewritten_note"]),
                    original_input=str(row["original_input"]),
                    inferred_work=str(row["inferred_work"]),
                    matched_work=str(row.get("matched_work") or row["inferred_work"]),
                    source_id=row.get("source_id"),
                    created_at=_as_datetime(row["created_at"]),
                    supporting_chunks=chunks_by_note.get(note_id, []),
                    combined_score=float(row.get("combined_score") or 0.0),
                    reason=row_reason,
                    vector_score=row.get("vector_score"),
                    lexical_score=row.get("lexical_score"),
                    trigram_score=row.get("trigram_score"),
                )
            )
        return notes

    def _supporting_chunks(self, note_ids: list[str]) -> dict[str, list[RetrievedChunk]]:
        if not note_ids:
            return {}
        rows = self.conn.execute(
            """
            SELECT nc.note_id, nc.rank, nc.label,
                   c.chunk_id, c.source_id, c.text, c.metadata
            FROM note_chunks AS nc
            LEFT JOIN chunks AS c ON c.chunk_id = nc.chunk_id
            WHERE nc.note_id = ANY(%s)
            ORDER BY nc.note_id, nc.rank
            """,
            [note_ids],
        ).fetchall()
        chunks_by_note: dict[str, list[RetrievedChunk]] = {}
        for row in rows:
            row = dict(row)
            if row.get("chunk_id") is None:
                continue
            note_id = str(row["note_id"])
            chunks_by_note.setdefault(note_id, []).append(
                RetrievedChunk(
                    label=str(row["label"]),
                    chunk_id=str(row["chunk_id"]),
                    source_id=str(row["source_id"]),
                    text=str(row["text"]),
                    metadata=dict(row.get("metadata") or {}),
                    combined_score=0.0,
                    reason="note support",
                )
            )
        return chunks_by_note


class NoteRelevanceService:
    """LLM relevance gate for supplemental notes; failures exclude notes."""

    def __init__(self, settings: Settings | None = None, model: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or make_chat_model(self.settings).with_structured_output(
            NoteRelevancePayload
        )

    def filter(self, question: str, notes: list[RetrievedNote]) -> list[RetrievedNote]:
        if not notes:
            return []
        payload = {
            "question": question,
            "candidate_notes": [
                {
                    "note_id": note.note_id,
                    "inferred_work": note.inferred_work,
                    "rewritten_note": note.rewritten_note,
                }
                for note in notes
            ],
        }
        try:
            result = self.model.invoke(
                RELEVANCE_PROMPT.invoke(
                    {"user_payload": json.dumps(payload, ensure_ascii=False)}
                )
            )
            parsed = _relevance_payload_from_model(result)
        except Exception as exc:
            logger.warning("note_relevance_filter_failed", error=str(exc))
            return []
        if not parsed.relevant_note_ids:
            return []
        by_id = {note.note_id: note for note in notes}
        relevant = [by_id[note_id] for note_id in parsed.relevant_note_ids if note_id in by_id]
        _label_notes(relevant)
        return relevant[: self.settings.question_note_top_k]


def _merge_rows(
    vector_rows: list[dict[str, Any]],
    lexical_rows: list[dict[str, Any]],
    trigram_rows: list[dict[str, Any]],
    rrf_k: int,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    rank_scores: dict[str, float] = {}

    for score_name, rows in [
        ("vector_score", vector_rows),
        ("lexical_score", lexical_rows),
        ("trigram_score", trigram_rows),
    ]:
        for rank, row in enumerate(rows, start=1):
            note_id = str(row["note_id"])
            if note_id not in by_id:
                by_id[note_id] = {
                    **row,
                    "vector_score": None,
                    "lexical_score": None,
                    "trigram_score": None,
                }
            by_id[note_id][score_name] = row.get(score_name)
            rank_scores[note_id] = rank_scores.get(note_id, 0.0) + 1 / (rrf_k + rank)

    rows = []
    for note_id, row in by_id.items():
        rows.append({**row, "combined_score": rank_scores[note_id]})
    rows.sort(key=lambda row: row["combined_score"], reverse=True)
    return rows


def _note_filter_clause(
    filters: dict[str, Any],
    *,
    exact_work: str | None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if exact_work:
        clauses.append("lower(inferred_work) = lower(%s)")
        params.append(exact_work)
    for key, value in filters.items():
        if value is None or key == "work":
            continue
        if isinstance(value, (dict, list)):
            clauses.append("work_metadata @> %s::jsonb")
            params.append(json.dumps({key: value}))
        else:
            clauses.append("work_metadata->>%s = %s")
            params.extend([key, str(value)])
    return (" AND ".join(clauses), params)


def _clean_lexical_query(query: str) -> str:
    tokens = NOTE_TOKEN_RE.findall(query.lower())
    if len(tokens) < 2:
        return query
    return " ".join(tokens)


def _clean_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_top_k(top_k: int | None, default: int) -> int:
    limit = default if top_k is None else top_k
    if limit < 1:
        raise ValueError("top_k must be at least 1")
    return limit


def _score(row_or_note: dict[str, Any] | RetrievedNote, name: str) -> float:
    value = row_or_note.get(name) if isinstance(row_or_note, dict) else getattr(row_or_note, name)
    return float(value or 0.0)


def _match_reason(row: dict[str, Any]) -> str:
    lanes = []
    if _score(row, "vector_score") > 0:
        lanes.append("vector")
    if _score(row, "lexical_score") > 0:
        lanes.append("lexical")
    if _score(row, "trigram_score") > 0:
        lanes.append("trigram")
    return " + ".join(lanes) + " note match" if lanes else "note match"


def _label_notes(notes: list[RetrievedNote]) -> None:
    for index, note in enumerate(notes, start=1):
        note.label = f"N{index}"


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.now(UTC)


def _relevance_payload_from_model(payload: object) -> NoteRelevancePayload:
    if isinstance(payload, NoteRelevancePayload):
        return payload
    if isinstance(payload, BaseModel):
        payload = payload.model_dump()
    if not isinstance(payload, dict):
        return NoteRelevancePayload()
    try:
        return NoteRelevancePayload.model_validate(payload)
    except ValidationError:
        return NoteRelevancePayload()


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"
