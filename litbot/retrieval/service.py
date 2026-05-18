import json
from typing import Any

import structlog
from psycopg import Connection

from litbot.config import Settings, get_settings
from litbot.langchain import document_to_retrieved_chunk, embed_query
from litbot.models import RetrievedChunk

logger = structlog.get_logger(__name__)


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

        query_vector = embed_query(question, self.settings)
        vector_rows = self._vector_search(query_vector, filters, limit * 3)
        lexical_rows = self._lexical_search(question, filters, limit * 3)
        merged = self._merge(vector_rows, lexical_rows, limit)

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

    def _merge(
        self,
        vector_rows: list[dict],
        lexical_rows: list[dict],
        limit: int,
    ) -> list[RetrievedChunk]:
        by_id: dict[str, dict] = {}

        for row in vector_rows:
            by_id[row["chunk_id"]] = {**row, "lexical_score": None}

        for row in lexical_rows:
            cid = row["chunk_id"]
            if cid in by_id:
                by_id[cid]["lexical_score"] = row["lexical_score"]
            else:
                by_id[cid] = {**row, "vector_score": 0.0}

        # Normalize each score set to [0, 1] before combining so they're
        # on the same scale. Raw ts_rank_cd values are unbounded upward.
        v_scores = [row.get("vector_score") or 0.0 for row in by_id.values()]
        l_scores = [row["lexical_score"] or 0.0 for row in by_id.values()]
        v_max = max(max(v_scores), 1.0) if v_scores else 1.0
        l_max = max(l_scores) if l_scores else 1.0

        ranked: list[RetrievedChunk] = []
        for row in by_id.values():
            v = (row.get("vector_score") or 0.0) / (v_max or 1.0)
            lexical_score = row.get("lexical_score") or 0.0
            lexical_normalized = lexical_score / (l_max or 1.0)
            combined = 0.75 * v + 0.25 * lexical_normalized

            # Describe how this chunk was actually found.
            has_v = (row.get("vector_score") or 0.0) > 0
            has_l = lexical_score > 0
            if has_v and has_l:
                reason = "hybrid vector + lexical match"
            elif has_v:
                reason = "vector match only"
            else:
                reason = "lexical match only"

            ranked.append(
                document_to_retrieved_chunk(
                    row,
                    vector_score=row.get("vector_score"),
                    lexical_score=row.get("lexical_score"),
                    combined_score=combined,
                    reason=reason,
                )
            )

        ranked.sort(key=lambda chunk: chunk.combined_score, reverse=True)
        for i, chunk in enumerate(ranked[:limit], start=1):
            chunk.label = f"S{i}"
        return ranked[:limit]


def _normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (filters or {}).items() if value is not None}


def _normalize_top_k(top_k: int | None, default: int) -> int:
    limit = default if top_k is None else top_k
    if limit < 1:
        raise ValueError("top_k must be at least 1")
    return limit


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


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"
