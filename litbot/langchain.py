from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from psycopg import Connection

from litbot.config import Settings, get_settings
from litbot.models import RetrievedChunk, TextChunk


def langchain_connection_string(database_url: str) -> str:
    """Return a SQLAlchemy-compatible psycopg connection string for LangChain."""

    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def make_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    settings = settings or get_settings()
    kwargs: dict[str, Any] = {
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
    }
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    return OpenAIEmbeddings(**kwargs)


def make_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    settings = settings or get_settings()
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "temperature": 0.2,
        "timeout": settings.request_timeout_seconds,
    }
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    return ChatOpenAI(**kwargs)


def make_vector_store(
    settings: Settings | None = None,
    *,
    pre_delete_collection: bool = False,
) -> PGVector:
    settings = settings or get_settings()
    return PGVector(
        embeddings=make_embeddings(settings),
        connection=langchain_connection_string(settings.database_url),
        collection_name=settings.vector_collection_name,
        distance_strategy=DistanceStrategy.COSINE,
        embedding_length=settings.embedding_dimensions,
        pre_delete_collection=pre_delete_collection,
        use_jsonb=True,
    )


def chunk_to_document(chunk: TextChunk) -> Document:
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "chunk_hash": chunk.chunk_hash,
        }
    )
    return Document(id=chunk.chunk_id, page_content=chunk.text, metadata=metadata)


def document_to_retrieved_chunk(
    document: Document,
    *,
    label: str = "",
    vector_score: float | None = None,
    lexical_score: float | None = None,
    combined_score: float = 0.0,
    reason: str = "hybrid vector/lexical match",
) -> RetrievedChunk:
    metadata = dict(document.metadata)
    chunk_id = str(metadata.get("chunk_id") or document.id or "")
    source_id = str(metadata.get("source_id") or "")
    return RetrievedChunk(
        label=label,
        chunk_id=chunk_id,
        source_id=source_id,
        text=document.page_content,
        metadata=metadata,
        vector_score=vector_score,
        lexical_score=lexical_score,
        combined_score=combined_score,
        reason=reason,
    )


def delete_source_documents(
    conn: Connection,
    source_id: str,
    settings: Settings | None = None,
) -> None:
    """Delete one source from the LangChain PGVector collection by metadata."""

    settings = settings or get_settings()
    conn.execute(
        """
        DELETE FROM langchain_pg_embedding e
        USING langchain_pg_collection c
        WHERE e.collection_id = c.uuid
          AND c.name = %s
          AND e.cmetadata->>'source_id' = %s
        """,
        (settings.vector_collection_name, source_id),
    )


def ensure_lexical_index(conn: Connection) -> None:
    """Add a full-text index on LangChain PGVector document content."""

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS langchain_pg_embedding_document_fts_idx
        ON langchain_pg_embedding
        USING gin (to_tsvector('english', document))
        """
    )
