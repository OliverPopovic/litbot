import json
from typing import Any

import structlog
from psycopg import Connection

from litbot.config import Settings, get_settings
from litbot.db import vector_literal
from litbot.models import RetrievedChunk
from litbot.openai_client import OpenAIModelClient

logger = structlog.get_logger(__name__)


class RetrievalService:
    """Hybrid semantic + lexical PostgreSQL retriever."""

    def __init__(
        self,
        conn: Connection,
        model_client: OpenAIModelClient,
        settings: Settings | None = None,
    ) -> None:
        self.conn = conn
        self.model_client = model_client
        self.settings = settings or get_settings()

    def retrieve(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        filters = filters or {}
        limit = top_k or self.settings.top_k
        query_embedding = self.model_client.embed_texts([question])[0]
        vector_rows = self._vector_search(query_embedding, filters, limit * 3)
        lexical_rows = self._lexical_search(question, filters, limit * 3)
        merged = self._merge_rows(vector_rows, lexical_rows, limit)
        logger.info("retrieval_completed", top_k=limit, returned=len(merged), filters=filters)
        return merged

    def _where_clause(self, filters: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in filters.items():
            if value is None:
                continue
            if key in {"author", "title", "genre", "language", "license"}:
                clauses.append(f"COALESCE(c.metadata->>%s, d.{key}) = %s")
                params.extend([key, value])
            else:
                clauses.append("c.metadata @> %s::jsonb")
                params.append(json.dumps({key: value}))
        return ("WHERE " + " AND ".join(clauses), params) if clauses else ("", [])

    def _vector_search(
        self,
        embedding: list[float],
        filters: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        where, params = self._where_clause(filters)
        rows = self.conn.execute(
            f"""
            SELECT c.chunk_id, c.source_id, c.text, c.metadata, d.title, d.author,
                   1 - (c.embedding <=> %s::vector) AS vector_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {where}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            [vector_literal(embedding), *params, vector_literal(embedding), limit],
        ).fetchall()
        return list(rows)

    def _lexical_search(
        self,
        question: str,
        filters: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        where, params = self._where_clause(filters)
        prefix = f"{where} AND" if where else "WHERE"
        rows = self.conn.execute(
            f"""
            SELECT c.chunk_id, c.source_id, c.text, c.metadata, d.title, d.author,
                   ts_rank_cd(
                       to_tsvector('english', c.text),
                       plainto_tsquery('english', %s)
                   ) AS lexical_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {prefix} to_tsvector('english', c.text) @@ plainto_tsquery('english', %s)
            ORDER BY lexical_score DESC
            LIMIT %s
            """,
            [question, *params, question, limit],
        ).fetchall()
        return list(rows)

    def _merge_rows(
        self,
        vector_rows: list[dict[str, Any]],
        lexical_rows: list[dict[str, Any]],
        limit: int,
    ) -> list[RetrievedChunk]:
        by_id: dict[str, dict[str, Any]] = {}
        for row in vector_rows:
            by_id[row["chunk_id"]] = dict(row)
        for row in lexical_rows:
            existing = by_id.setdefault(row["chunk_id"], dict(row))
            existing["lexical_score"] = row.get("lexical_score")

        ranked: list[RetrievedChunk] = []
        for row in by_id.values():
            vector_score = float(row.get("vector_score") or 0.0)
            lexical_score = float(row.get("lexical_score") or 0.0)
            combined = (0.75 * vector_score) + (0.25 * min(lexical_score, 1.0))
            ranked.append(
                RetrievedChunk(
                    label="",
                    chunk_id=row["chunk_id"],
                    source_id=row["source_id"],
                    text=row["text"],
                    metadata=row["metadata"] or {},
                    vector_score=vector_score or None,
                    lexical_score=lexical_score or None,
                    combined_score=combined,
                    reason="hybrid vector/lexical match",
                )
            )
        ranked.sort(key=lambda item: item.combined_score, reverse=True)
        for index, item in enumerate(ranked[:limit], start=1):
            item.label = f"S{index}"
        return ranked[:limit]
