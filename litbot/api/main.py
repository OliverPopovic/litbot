import uuid

import structlog
from fastapi import FastAPI, Header

from litbot.config import get_settings
from litbot.db import get_connection
from litbot.generation.service import GenerationService
from litbot.models import ChatRequest, ChatResponse
from litbot.observability.logging import configure_logging
from litbot.openai_client import OpenAIModelClient
from litbot.retrieval.service import RetrievalService

configure_logging()
logger = structlog.get_logger(__name__)
app = FastAPI(title="LitBot Literary RAG API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, x_trace_id: str | None = Header(default=None)) -> ChatResponse:
    trace_id = x_trace_id or str(uuid.uuid4())
    settings = get_settings()
    model_client = OpenAIModelClient(settings)
    with get_connection(settings) as conn:
        chunks = RetrievalService(conn, model_client, settings).retrieve(
            request.question,
            filters=request.filters,
            top_k=request.top_k,
        )
    logger.info("chat_request", trace_id=trace_id, chunk_count=len(chunks))
    return GenerationService(model_client, settings).answer(
        request.question,
        chunks,
        trace_id=trace_id,
    )
