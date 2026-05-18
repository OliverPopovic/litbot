import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from dotenv import load_dotenv

from litbot.api.main import app as fastapi_app
from litbot.config import get_settings
from litbot.corpus import fetch_public_domain_corpus
from litbot.db import close_pool, get_connection
from litbot.evaluation.golden import load_golden_questions, score_answers
from litbot.evaluation.retrieval import load_retrieval_cases, result_to_dict, score_retrieval
from litbot.generation.service import GenerationService
from litbot.ingestion.store import IngestionService
from litbot.observability.logging import configure_logging
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
    try:
        with get_connection(settings) as conn:
            chunks = IngestionService(conn, settings).ingest_path(path, metadata)
    finally:
        close_pool()
    typer.echo(f"Ingested {len(chunks)} chunks from {path}")


@app.command()
def reindex(
    corpus_dir: Annotated[
        Path,
        typer.Argument(help="Directory of corpus files to reingest."),
    ] = Path("corpus"),
) -> None:
    """Recreate first-party document and chunk rows, then reingest corpus files."""

    settings = get_settings()
    ingested = 0
    try:
        with get_connection(settings) as conn:
            conn.execute("DELETE FROM documents")
            conn.commit()
            service = IngestionService(conn, settings)
            for path in sorted(corpus_dir.rglob("*")):
                if path.suffix.lower() not in {".txt", ".md", ".html", ".htm", ".pdf"}:
                    continue
                if not path.with_suffix(path.suffix + ".json").exists():
                    continue
                ingested += len(service.ingest_path(path))
    finally:
        close_pool()
    typer.echo(f"Reindexed {ingested} chunks from {corpus_dir}")


@app.command("fetch-corpus")
def fetch_corpus(
    target: Annotated[
        Path,
        typer.Option(help="Directory where downloaded corpus files and sidecars are written."),
    ] = Path(".litbot_corpus/public_domain"),
) -> None:
    """Download the default public-domain corpus and metadata sidecars."""

    paths = fetch_public_domain_corpus(target)
    typer.echo(f"Wrote {len(paths)} files to {target}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer from the corpus.")],
) -> None:
    """Ask a question against the configured corpus."""

    settings = get_settings()
    try:
        with get_connection(settings) as conn:
            chunks = RetrievalService(conn, settings).retrieve(question)
    finally:
        close_pool()
    response = GenerationService(settings).answer(question, chunks)
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


@app.command("eval-retrieval")
def evaluate_retrieval(
    cases_jsonl: Annotated[Path, typer.Argument(help="JSONL file of retrieval golden cases.")],
) -> None:
    """Score retrieval against golden expected chunks without running generation."""

    cases = load_retrieval_cases(cases_jsonl)
    settings = get_settings()
    try:
        with get_connection(settings) as conn:
            service = RetrievalService(conn, settings)
            result = score_retrieval(
                cases,
                lambda question, filters, k: service.retrieve(question, filters=filters, top_k=k),
            )
    finally:
        close_pool()
    typer.echo(json.dumps(result_to_dict(result), indent=2))
