from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LitBotModel(BaseModel):
    """Shared Pydantic defaults for API and internal transfer objects."""

    model_config = ConfigDict(extra="forbid")


class DocumentMetadata(LitBotModel):
    source_id: str
    title: str
    license: str
    author: str
    translator: str | None = None
    editor: str | None = None
    publication_year: int
    edition: str | None = None
    genre: str
    language: str = "en"
    uri: str
    version: str = "1"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_id",
        "title",
        "license",
        "author",
        "genre",
        "language",
        "uri",
        "version",
    )
    @classmethod
    def _require_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("metadata")
    @classmethod
    def _require_work_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        # The corpus stores human citation context under metadata.work; keep that invariant here
        # instead of duplicating sidecar validation in the parser.
        if not value.get("work"):
            raise ValueError("metadata.work is required")
        return value


class ParsedDocument(LitBotModel):
    metadata: DocumentMetadata
    text: str


class TextChunk(LitBotModel):
    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    token_count: int
    chunk_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(LitBotModel):
    label: str
    chunk_id: str
    source_id: str
    text: str
    metadata: dict[str, Any]
    combined_score: float
    reason: str
    vector_score: float | None = None
    lexical_score: float | None = None


class ChatRequest(LitBotModel):
    question: str
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def _require_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class Citation(LitBotModel):
    label: str
    source_id: str
    chunk_id: str
    reference: str


class ChatResponse(LitBotModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    prompt_version: str
    trace_id: str
    citation_map: list[dict[str, Any]] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
