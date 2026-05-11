from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from dotenv import load_dotenv

from litbot.api.main import app as fastapi_app
from litbot.config import get_settings
from litbot.db import get_connection
from litbot.evaluation.golden import load_golden_questions, score_answers
from litbot.generation.service import GenerationService
from litbot.ingestion.store import IngestionService
from litbot.observability.logging import configure_logging
from litbot.openai_client import OpenAIModelClient
from litbot.retrieval.service import RetrievalService

app = typer.Typer(help="LitBot ingestion, serving, and evaluation commands.")


@app.callback()
def main() -> None:
    load_dotenv()
    configure_logging()


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind host.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Run the FastAPI application."""

    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Document file to ingest.")],
    metadata: Annotated[Path | None, typer.Option(help="Optional sidecar metadata JSON.")] = None,
) -> None:
    """Parse, chunk, embed, and store a document."""

    settings = get_settings()
    model_client = OpenAIModelClient(settings)
    with get_connection(settings) as conn:
        chunks = IngestionService(conn, model_client).ingest_path(path, metadata)
    typer.echo(f"Ingested {len(chunks)} chunks from {path}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer from the corpus.")],
) -> None:
    """Ask a question against the configured corpus."""

    settings = get_settings()
    model_client = OpenAIModelClient(settings)
    with get_connection(settings) as conn:
        chunks = RetrievalService(conn, model_client, settings).retrieve(question)
    response = GenerationService(model_client, settings).answer(question, chunks)
    typer.echo(response.model_dump_json(indent=2))


@app.command("eval")
def evaluate(
    answers_jsonl: Annotated[Path, typer.Argument(help="JSONL file of generated answers.")],
) -> None:
    """Score a JSONL answer export for simple citation and coverage metrics."""

    rows = load_golden_questions(answers_jsonl)
    result = score_answers(rows)
    typer.echo(
        {
            "total": result.total,
            "answered": result.answered,
            "cited": result.cited,
            "unsupported": result.unsupported,
            "citation_rate": result.citation_rate,
        }
    )
