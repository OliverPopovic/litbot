from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from litbot.config import Settings, get_settings
from litbot.models import RetrievedChunk


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


def embed_texts(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """Embed a batch of strings and return one float vector per string."""

    return make_embeddings(settings).embed_documents(texts)


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    """Embed a single query string."""

    return make_embeddings(settings).embed_query(text)


def document_to_retrieved_chunk(
    row: dict,
    *,
    label: str = "",
    vector_score: float | None = None,
    lexical_score: float | None = None,
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
        combined_score=combined_score,
        reason=reason,
    )
