import json
from typing import Any

import structlog
from langchain_core.documents import Document
from psycopg import Connection

from litbot.config import Settings, get_settings
from litbot.langchain import document_to_retrieved_chunk, ensure_lexical_index, make_vector_store
from litbot.models import RetrievedChunk

logger = structlog.get_logger(__name__)


class RetrievalService:
    """Hybrid semantic + lexical retriever backed by LangChain PGVector."""

    def __init__(
        self,
        conn: Connection,
        settings: Settings | None = None,
    ) -> None:
        self.conn = conn
        self.settings = settings or get_settings()
        self.vector_store = make_vector_store(self.settings)

    def retrieve(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        filters = filters or {}
        limit = top_k or self.settings.top_k
        vector_rows = self._vector_search(question, filters, limit * 3)
        lexical_rows = self._lexical_search(question, filters, limit * 3)
        merged = self._merge_rows(vector_rows, lexical_rows, limit)
        logger.info("retrieval_completed", top_k=limit, returned=len(merged), filters=filters)
        return merged

    def _vector_search(
        self,
        question: str,
        filters: dict[str, Any],
        limit: int,
    ) -> list[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(
            query=question,
            k=limit,
            filter=_to_pgvector_filter(filters),
        )

    def _lexical_search(
        self,
        question: str,
        filters: dict[str, Any],
        limit: int,
    ) -> list[tuple[Document, float]]:
        ensure_lexical_index(self.conn)
        where, params = _metadata_where_clause(filters)
        prefix = f"AND {where}" if where else ""
        rows = self.conn.execute(
            f"""
            SELECT e.document, e.cmetadata,
                   ts_rank_cd(
                       to_tsvector('english', e.document),
                       plainto_tsquery('english', %s)
                   ) AS lexical_score
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = %s
              AND to_tsvector('english', e.document) @@ plainto_tsquery('english', %s)
              {prefix}
            ORDER BY lexical_score DESC
            LIMIT %s
            """,
            [question, self.settings.vector_collection_name, question, *params, limit],
        ).fetchall()
        return [
            (
                Document(
                    id=str((row["cmetadata"] or {}).get("chunk_id") or ""),
                    page_content=row["document"],
                    metadata=row["cmetadata"] or {},
                ),
                float(row["lexical_score"] or 0.0),
            )
            for row in rows
        ]

    def _merge_rows(
        self,
        vector_rows: list[tuple[Document, float]],
        lexical_rows: list[tuple[Document, float]],
        limit: int,
    ) -> list[RetrievedChunk]:
        by_id: dict[str, dict[str, Any]] = {}
        for document, distance in vector_rows:
            chunk_id = _document_chunk_id(document)
            by_id[chunk_id] = {
                "document": document,
                "vector_score": _distance_to_similarity(distance),
                "lexical_score": None,
            }
        for document, lexical_score in lexical_rows:
            chunk_id = _document_chunk_id(document)
            existing = by_id.setdefault(
                chunk_id,
                {"document": document, "vector_score": 0.0, "lexical_score": None},
            )
            existing["lexical_score"] = lexical_score

        ranked: list[RetrievedChunk] = []
        for row in by_id.values():
            vector_score = float(row.get("vector_score") or 0.0)
            lexical_score = float(row.get("lexical_score") or 0.0)
            combined = (0.75 * vector_score) + (0.25 * min(lexical_score, 1.0))
            ranked.append(
                document_to_retrieved_chunk(
                    row["document"],
                    vector_score=vector_score or None,
                    lexical_score=lexical_score or None,
                    combined_score=combined,
                )
            )
        ranked.sort(key=lambda item: item.combined_score, reverse=True)
        for index, item in enumerate(ranked[:limit], start=1):
            item.label = f"S{index}"
        return ranked[:limit]


def _to_pgvector_filter(filters: dict[str, Any]) -> dict[str, Any] | None:
    conditions = []
    for key, value in filters.items():
        if value is not None:
            conditions.append({key: {"$eq": value}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _metadata_where_clause(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            clauses.append("e.cmetadata @> %s::jsonb")
            params.append(json.dumps({key: value}))
        else:
            clauses.append("e.cmetadata->>%s = %s")
            params.extend([key, str(value)])
    return (" AND ".join(clauses), params)


def _document_chunk_id(document: Document) -> str:
    return str(document.metadata.get("chunk_id") or document.id or "")


def _distance_to_similarity(distance: float) -> float:
    # LangChain PGVector returns a distance, while the previous retriever used cosine similarity.
    return max(0.0, 1.0 - float(distance))
