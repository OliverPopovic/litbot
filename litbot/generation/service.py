import uuid

import structlog

from litbot.config import Settings, get_settings
from litbot.generation.citations import validate_and_format_citations, validate_and_format_labels
from litbot.generation.prompts import build_messages
from litbot.models import ChatResponse, RetrievedChunk
from litbot.openai_client import OpenAIModelClient

logger = structlog.get_logger(__name__)


class GenerationService:
    """Grounded answer generator with citation validation."""

    def __init__(self, model_client: OpenAIModelClient, settings: Settings | None = None) -> None:
        self.model_client = model_client
        self.settings = settings or get_settings()

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

        payload = self.model_client.generate_json(build_messages(question, chunks))
        answer = str(payload.get("answer", "")).strip()
        citation_map = payload.get("citation_map") or []
        unsupported = payload.get("unsupported") or []
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
