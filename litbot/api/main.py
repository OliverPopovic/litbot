import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from litbot.chat import handle_chat_request
from litbot.config import get_settings
from litbot.db import close_pool, get_connection
from litbot.models import ChatRequest, ChatResponse
from litbot.observability.logging import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)
UI_DIR = Path(__file__).resolve().parents[1] / "ui"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        close_pool()


app = FastAPI(title="LitBot Literary RAG API", version="0.1.0", lifespan=lifespan)
app.mount("/ui/static", StaticFiles(directory=UI_DIR / "static"), name="ui-static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=str(request.url), error=str(exc))
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=FileResponse)
def ui() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, x_trace_id: str | None = Header(default=None)) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be blank")

    trace_id = x_trace_id or str(uuid.uuid4())
    settings = get_settings()

    with get_connection(settings) as conn:
        response = handle_chat_request(conn, settings, request, trace_id=trace_id)

    logger.info(
        "chat_request_completed",
        trace_id=trace_id,
        intent=response.intent,
        note_status=response.note_status,
        chunk_count=len(response.retrieved_chunks),
    )
    return response
