from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
import json


class SerializableDataclass:
    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def model_dump_json(self, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(), default=str, indent=indent)


@dataclass
class DocumentMetadata(SerializableDataclass):
    source_id: str
    title: str
    license: str
    author: str | None = None
    translator: str | None = None
    editor: str | None = None
    publication_year: int | None = None
    edition: str | None = None
    genre: str | None = None
    language: str = "en"
    uri: str | None = None
    version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument(SerializableDataclass):
    metadata: DocumentMetadata
    text: str


@dataclass
class TextChunk(SerializableDataclass):
    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    token_count: int
    chunk_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk(SerializableDataclass):
    label: str
    chunk_id: str
    source_id: str
    text: str
    metadata: dict[str, Any]
    combined_score: float
    reason: str
    vector_score: float | None = None
    lexical_score: float | None = None


@dataclass
class ChatRequest(SerializableDataclass):
    question: str
    filters: dict[str, Any] = field(default_factory=dict)
    top_k: int | None = None


@dataclass
class Citation(SerializableDataclass):
    label: str
    source_id: str
    chunk_id: str
    reference: str


@dataclass
class ChatResponse(SerializableDataclass):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    prompt_version: str
    trace_id: str
    citation_map: list[dict[str, Any]] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
