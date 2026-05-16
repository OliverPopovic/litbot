import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from litbot.config import get_settings
from litbot.db import close_pool, get_connection
from litbot.generation.service import GenerationService
from litbot.models import ChatRequest, ChatResponse
from litbot.observability.logging import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        close_pool()


app = FastAPI(title="LitBot Literary RAG API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=str(request.url), error=str(exc))
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, x_trace_id: str | None = Header(default=None)) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be blank")

    trace_id = x_trace_id or str(uuid.uuid4())
    settings = get_settings()

    from litbot.retrieval.service import RetrievalService

    with get_connection(settings) as conn:
        chunks = RetrievalService(conn, settings).retrieve(
            request.question,
            filters=request.filters,
            top_k=request.top_k,
        )

    logger.info("chat_request", trace_id=trace_id, chunk_count=len(chunks))
    return GenerationService(settings).answer(request.question, chunks, trace_id=trace_id)
