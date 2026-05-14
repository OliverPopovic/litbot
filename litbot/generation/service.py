import uuid
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from litbot.config import Settings, get_settings
from litbot.generation.citations import validate_and_format_citations, validate_and_format_labels
from litbot.generation.prompts import build_prompt_value
from litbot.langchain import make_chat_model
from litbot.models import ChatResponse, RetrievedChunk

logger = structlog.get_logger(__name__)


class CitationMapItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = ""
    sources: list[str] = Field(default_factory=list)


class GenerationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = ""
    citation_map: list[CitationMapItem] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)


class GenerationService:
    """Grounded answer generator with LangChain structured output and citation validation."""

    def __init__(self, settings: Settings | None = None, model: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or make_chat_model(self.settings).with_structured_output(
            GenerationPayload
        )

    def answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        trace_id: str | None = None,
    ) -> ChatResponse:
        trace_id = trace_id or str(uuid.uuid4())
        if not chunks:
            return ChatResponse(
                answer=(
                    "I could not find enough evidence in the approved corpus to "
                    "answer that question."
                ),
                citations=[],
                retrieved_chunks=[],
                unsupported=["No relevant chunks were retrieved."],
                prompt_version=self.settings.prompt_version,
                trace_id=trace_id,
            )

        payload = self.model.invoke(build_prompt_value(question, chunks))
        payload_dict = _payload_to_dict(payload)
        answer = str(payload_dict.get("answer", "")).strip()
        citation_map = payload_dict.get("citation_map") or []
        unsupported = payload_dict.get("unsupported") or []
        citations = validate_and_format_citations(answer, chunks)
        if not citations:
            citations = validate_and_format_labels(_labels_from_citation_map(citation_map), chunks)
        logger.info("generation_completed", trace_id=trace_id, citation_count=len(citations))
        return ChatResponse(
            answer=answer,
            citations=citations,
            citation_map=citation_map,
            retrieved_chunks=chunks,
            unsupported=unsupported,
            prompt_version=self.settings.prompt_version,
            trace_id=trace_id,
        )


def _payload_to_dict(payload: object) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        return payload.model_dump()
    if isinstance(payload, dict):
        return payload
    return {}


def _labels_from_citation_map(citation_map: object) -> set[str]:
    labels: set[str] = set()
    if not isinstance(citation_map, list):
        return labels
    for item in citation_map:
        if not isinstance(item, dict):
            continue
        sources = item.get("sources")
        if not isinstance(sources, list):
            continue
        labels.update(source for source in sources if isinstance(source, str))
    return labels
