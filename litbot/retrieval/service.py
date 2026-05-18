import json
import re
from typing import Any

import structlog
from psycopg import Connection

from litbot.config import Settings, get_settings
from litbot.langchain import embed_query
from litbot.models import RetrievedChunk

logger = structlog.get_logger(__name__)

QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
QUERY_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "any",
    "are",
    "does",
    "find",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "passage",
    "retrieve",
    "say",
    "says",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "with",
}


class RetrievalService:
    """Hybrid semantic + lexical retriever over the litbot schema."""

    def __init__(self, conn: Connection, settings: Settings | None = None) -> None:
        self.conn = conn
        self.settings = settings or get_settings()

    def retrieve(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        filters = _normalize_filters(filters)
        limit = _normalize_top_k(top_k, self.settings.top_k)
        candidate_limit = _candidate_limit(
            limit,
            multiplier=self.settings.retrieval_candidate_multiplier,
            minimum=self.settings.retrieval_min_candidates,
            maximum=self.settings.retrieval_max_candidates,
        )

        query_vector = embed_query(question, self.settings)
        vector_rows = self._vector_search(query_vector, filters, candidate_limit)
        lexical_query = _clean_lexical_query(question, filters)
        lexical_rows = self._lexical_search(lexical_query, filters, candidate_limit)
        trigram_rows = self._trigram_search(question, filters, candidate_limit)
        merged = self._merge(vector_rows, lexical_rows, trigram_rows, limit)
        if self.settings.retrieval_include_neighbors:
            merged = self._expand_neighbors(merged, filters, limit)

        logger.info("retrieval_completed", top_k=limit, returned=len(merged), filters=filters)
        return merged

    def _vector_search(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        limit: int,
    ) -> list[dict]:
        where, params = _metadata_where_clause(filters)
        prefix = f"AND {where}" if where else ""
        vector = _vector_literal(query_vector)
        rows = self.conn.execute(
            f"""
            SELECT chunk_id, source_id, text, metadata,
                   1 - (embedding <=> %s::vector) AS vector_score
            FROM chunks
            WHERE 1=1 {prefix}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            [vector, *params, vector, limit],
        ).fetchall()
        return [dict(row) for row in rows]

    def _lexical_search(
        self,
        question: str,
        filters: dict[str, Any],
        limit: int,
    ) -> list[dict]:
        where, params = _metadata_where_clause(filters)
        prefix = f"AND {where}" if where else ""
        rows = self.conn.execute(
            f"""
            SELECT chunk_id, source_id, text, metadata,
                   ts_rank_cd(to_tsvector('english', text),
                              plainto_tsquery('english', %s)) AS lexical_score
            FROM chunks
            WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)
              {prefix}
            ORDER BY lexical_score DESC
            LIMIT %s
            """,
            [question, question, *params, limit],
        ).fetchall()
        return [dict(row) for row in rows]

    def _trigram_search(
        self,
        question: str,
        filters: dict[str, Any],
        limit: int,
    ) -> list[dict]:
        where, params = _metadata_where_clause(filters)
        prefix = f"AND {where}" if where else ""
        rows = self.conn.execute(
            f"""
            SELECT chunk_id, source_id, text, metadata,
                   word_similarity(%s, text) AS trigram_score
            FROM chunks
            WHERE word_similarity(%s, text) > 0
              {prefix}
            ORDER BY trigram_score DESC
            LIMIT %s
            """,
            [question, question, *params, limit],
        ).fetchall()
        return [dict(row) for row in rows]

    def _merge(
        self,
        vector_rows: list[dict],
        lexical_rows: list[dict],
        trigram_rows: list[dict],
        limit: int,
    ) -> list[RetrievedChunk]:
        by_id: dict[str, dict] = {}
        rank_scores: dict[str, float] = {}
        rrf_k = getattr(getattr(self, "settings", None), "retrieval_rrf_k", 60)

        for rank, row in enumerate(vector_rows, start=1):
            cid = row["chunk_id"]
            by_id[cid] = {**row, "lexical_score": None, "trigram_score": None}
            rank_scores[cid] = rank_scores.get(cid, 0.0) + _rrf_score(rank, rrf_k)

        for rank, row in enumerate(lexical_rows, start=1):
            cid = row["chunk_id"]
            if cid in by_id:
                by_id[cid]["lexical_score"] = row["lexical_score"]
            else:
                by_id[cid] = {**row, "vector_score": 0.0, "trigram_score": None}
            rank_scores[cid] = rank_scores.get(cid, 0.0) + _rrf_score(rank, rrf_k)

        for rank, row in enumerate(trigram_rows, start=1):
            cid = row["chunk_id"]
            if cid in by_id:
                by_id[cid]["trigram_score"] = row["trigram_score"]
            else:
                by_id[cid] = {**row, "vector_score": 0.0, "lexical_score": None}
            rank_scores[cid] = rank_scores.get(cid, 0.0) + _rrf_score(rank, rrf_k)

        ranked: list[RetrievedChunk] = []
        for row in by_id.values():
            combined = rank_scores[row["chunk_id"]]

            ranked.append(
                document_to_retrieved_chunk(
                    row,
                    vector_score=row.get("vector_score"),
                    lexical_score=row.get("lexical_score"),
                    trigram_score=row.get("trigram_score"),
                    combined_score=combined,
                    reason=_match_reason(row),
                )
            )

        ranked.sort(key=lambda chunk: chunk.combined_score, reverse=True)
        for i, chunk in enumerate(ranked[:limit], start=1):
            chunk.label = f"S{i}"
        return ranked[:limit]

    def _expand_neighbors(
        self,
        chunks: list[RetrievedChunk],
        filters: dict[str, Any],
        limit: int,
    ) -> list[RetrievedChunk]:
        window = self.settings.retrieval_neighbor_window
        if window < 1 or not chunks:
            return chunks

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        where, params = _metadata_where_clause(filters)
        prefix = f"AND neighbor.{where}" if where else ""
        rows = self.conn.execute(
            f"""
            WITH seeds AS (
                SELECT chunk_id, document_id, chunk_index
                FROM chunks
                WHERE chunk_id = ANY(%s)
            )
            SELECT DISTINCT seeds.chunk_id AS seed_chunk_id,
                            abs(neighbor.chunk_index - seeds.chunk_index) AS distance,
                            neighbor.chunk_index,
                            neighbor.chunk_id, neighbor.source_id, neighbor.text,
                            neighbor.metadata
            FROM chunks AS neighbor
            JOIN seeds ON neighbor.document_id = seeds.document_id
            WHERE neighbor.chunk_index BETWEEN seeds.chunk_index - %s
                                          AND seeds.chunk_index + %s
              {prefix}
            ORDER BY seed_chunk_id, distance, neighbor.chunk_index
            """,
            [chunk_ids, window, window, *params],
        ).fetchall()

        neighbors_by_seed: dict[str, list[dict]] = {}
        for row in rows:
            seed_id = str(row["seed_chunk_id"])
            neighbors_by_seed.setdefault(seed_id, []).append(dict(row))

        expanded: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                expanded.append(chunk)
                seen.add(chunk.chunk_id)
            for row in neighbors_by_seed.get(chunk.chunk_id, []):
                cid = str(row["chunk_id"])
                if cid in seen:
                    continue
                expanded.append(
                    document_to_retrieved_chunk(
                        row,
                        combined_score=chunk.combined_score,
                        reason="neighbor context",
                    )
                )
                seen.add(cid)

        for i, chunk in enumerate(expanded[:limit], start=1):
            chunk.label = f"S{i}"
        return expanded[:limit]


def _normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (filters or {}).items() if value is not None}


def _normalize_top_k(top_k: int | None, default: int) -> int:
    limit = default if top_k is None else top_k
    if limit < 1:
        raise ValueError("top_k must be at least 1")
    return limit


def _candidate_limit(top_k: int, *, multiplier: int, minimum: int, maximum: int) -> int:
    if multiplier < 1:
        raise ValueError("retrieval_candidate_multiplier must be at least 1")
    if minimum < 1:
        raise ValueError("retrieval_min_candidates must be at least 1")
    if maximum < minimum:
        raise ValueError("retrieval_max_candidates must be greater than or equal to minimum")
    return min(max(top_k * multiplier, minimum), maximum)


def _clean_lexical_query(question: str, filters: dict[str, Any]) -> str:
    filter_terms = _filter_terms(filters)
    tokens = [
        token
        for token in QUERY_TOKEN_RE.findall(question.lower())
        if token not in QUERY_STOP_WORDS and token not in filter_terms
    ]
    if len(tokens) < 2:
        return question
    return " ".join(tokens)


def _filter_terms(filters: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for key, value in filters.items():
        if key == "work" and isinstance(value, str):
            terms.update(QUERY_TOKEN_RE.findall(value.lower()))
    return terms


def _rrf_score(rank: int, rrf_k: int) -> float:
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if rrf_k < 1:
        raise ValueError("retrieval_rrf_k must be at least 1")
    return 1 / (rrf_k + rank)


def _metadata_where_clause(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    """Translate user filters into SQL predicates against the chunks.metadata JSONB column."""

    clauses: list[str] = []
    params: list[Any] = []
    for key, value in filters.items():
        if isinstance(value, (dict, list)):
            clauses.append("metadata @> %s::jsonb")
            params.append(json.dumps({key: value}))
        else:
            clauses.append("metadata->>%s = %s")
            params.extend([key, str(value)])
    return (" AND ".join(clauses), params)


def document_to_retrieved_chunk(
    row: dict,
    *,
    label: str = "",
    vector_score: float | None = None,
    lexical_score: float | None = None,
    trigram_score: float | None = None,
    combined_score: float = 0.0,
    reason: str = "",
) -> RetrievedChunk:
    metadata = dict(row.get("metadata") or {})
    return RetrievedChunk(
        label=label,
        chunk_id=str(row["chunk_id"]),
        source_id=str(row["source_id"]),
        text=str(row["text"]),
        metadata=metadata,
        vector_score=vector_score,
        lexical_score=lexical_score,
        trigram_score=trigram_score,
        combined_score=combined_score,
        reason=reason,
    )


def _match_reason(row: dict) -> str:
    lanes: list[str] = []
    if (row.get("vector_score") or 0.0) > 0:
        lanes.append("vector")
    if (row.get("lexical_score") or 0.0) > 0:
        lanes.append("lexical")
    if (row.get("trigram_score") or 0.0) > 0:
        lanes.append("trigram")
    if not lanes:
        return "neighbor context"
    return " + ".join(lanes) + " match"


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"
