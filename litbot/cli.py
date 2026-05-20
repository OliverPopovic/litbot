import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from dotenv import load_dotenv

from litbot.api.main import app as fastapi_app
from litbot.chat import handle_chat_request
from litbot.config import get_settings
from litbot.corpus import fetch_public_domain_corpus
from litbot.db import close_pool, get_connection
from litbot.evaluation.golden import load_golden_questions, score_answers
from litbot.evaluation.retrieval import load_retrieval_cases, result_to_dict, score_retrieval
from litbot.ingestion.store import IngestionService
from litbot.models import ChatRequest, ChatResponse, RetrievedChunk, RetrievedNote
from litbot.observability.logging import configure_logging
from litbot.retrieval.service import RetrievalService

app = typer.Typer(help="LitBot ingestion, serving, and evaluation commands.")


@app.callback()
def main() -> None:
    load_dotenv()
    configure_logging(stream=sys.stderr, renderer="console")


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
    _success("Ingested document", chunks=len(chunks), path=str(path))


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
    _success("Reindexed corpus", chunks=ingested, path=str(corpus_dir))


@app.command("fetch-corpus")
def fetch_corpus(
    target: Annotated[
        Path,
        typer.Option(help="Directory where downloaded corpus files and sidecars are written."),
    ] = Path(".litbot_corpus/public_domain"),
) -> None:
    """Download the default public-domain corpus and metadata sidecars."""

    paths = fetch_public_domain_corpus(target)
    _success("Fetched corpus", files=len(paths), path=str(target))


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer from the corpus.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Write the raw ChatResponse JSON."),
    ] = False,
) -> None:
    """Ask a question against the configured corpus."""

    settings = get_settings()
    try:
        with get_connection(settings) as conn:
            response = handle_chat_request(conn, settings, ChatRequest(question=question))
    finally:
        close_pool()
    if json_output:
        typer.echo(response.model_dump_json(indent=2))
        return
    _render_chat_response(response)


@app.command("eval")
def evaluate(
    answers_jsonl: Annotated[Path, typer.Argument(help="JSONL file of generated answers.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Write machine-readable JSON."),
    ] = False,
) -> None:
    """Score a JSONL answer export for simple citation and coverage metrics."""

    rows = load_golden_questions(answers_jsonl)
    result = score_answers(rows)
    payload = {
        "total": result.total,
        "answered": result.answered,
        "cited": result.cited,
        "unsupported": result.unsupported,
        "citation_rate": result.citation_rate,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    _render_metrics("Answer evaluation", payload)


@app.command("eval-retrieval")
def evaluate_retrieval(
    cases_jsonl: Annotated[Path, typer.Argument(help="JSONL file of retrieval golden cases.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Write machine-readable JSON."),
    ] = False,
) -> None:
    """Score retrieval against golden expected chunks without running generation."""

    configure_logging(stream=sys.stderr, renderer="console")
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
    payload = result_to_dict(result)
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    _render_metrics("Retrieval evaluation", payload)


def _success(title: str, **fields: Any) -> None:
    typer.secho(f"\n{title}", fg=typer.colors.GREEN, bold=True)
    for key, value in fields.items():
        typer.echo(f"  {key.replace('_', ' ').title():<10} {value}")


def _render_chat_response(response: ChatResponse) -> None:
    if response.note_status:
        _render_note_response(response)
        return
    if response.note_query_status:
        _render_note_query_response(response)
        return

    typer.secho("\nAnswer", fg=typer.colors.GREEN, bold=True)
    typer.echo(_indent(response.answer.strip() or "No answer returned."))

    if response.citations:
        typer.secho("\nCitations", fg=typer.colors.BLUE, bold=True)
        for citation in response.citations:
            typer.echo(
                f"  [{citation.label}] {citation.reference} "
                f"({citation.source_id}, {citation.chunk_id})"
            )

    if response.unsupported:
        typer.secho("\nUnsupported", fg=typer.colors.YELLOW, bold=True)
        for item in response.unsupported:
            typer.echo(f"  - {item}")

    if response.retrieved_notes:
        typer.secho("\nRelevant Notes", fg=typer.colors.BLUE, bold=True)
        for note in response.retrieved_notes:
            _render_retrieved_note(note)

    if response.retrieved_chunks:
        typer.secho("\nRetrieved Context", fg=typer.colors.MAGENTA, bold=True)
        for chunk in response.retrieved_chunks:
            _render_chunk(chunk)

    typer.secho("\nRun", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Trace ID       {response.trace_id}")
    typer.echo(f"  Prompt version {response.prompt_version}")


def _render_note_query_response(response: ChatResponse) -> None:
    title = "Notes Found" if response.note_query_status == "found" else "Notes Not Found"
    color = typer.colors.GREEN if response.note_query_status == "found" else typer.colors.YELLOW
    typer.secho(f"\n{title}", fg=color, bold=True)
    typer.echo(_indent(response.answer.strip() or "No notes returned."))

    if response.retrieved_notes:
        typer.secho("\nStored Notes", fg=typer.colors.BLUE, bold=True)
        for note in response.retrieved_notes:
            _render_retrieved_note(note)

    if response.note_query_has_more:
        typer.secho("\nLimit", fg=typer.colors.YELLOW, bold=True)
        typer.echo("  This is a capped preview; pagination is not available in v1.")

    if response.retrieved_chunks:
        typer.secho("\nSupporting Chunks", fg=typer.colors.MAGENTA, bold=True)
        for chunk in response.retrieved_chunks:
            _render_chunk(chunk)

    typer.secho("\nRun", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Trace ID       {response.trace_id}")
    typer.echo(f"  Prompt version {response.prompt_version}")


def _render_note_response(response: ChatResponse) -> None:
    status = response.note_status.replace("_", " ").title()
    color = typer.colors.GREEN if response.note_status == "saved" else typer.colors.YELLOW
    typer.secho(f"\nNote {status}", fg=color, bold=True)
    if response.note_status == "saved":
        typer.echo(_indent(response.note or "No note returned."))
        typer.echo("\nStored note differs from the original input when LitBot rewrites it.")
    else:
        typer.echo(_indent(response.note_rejection_reason or "The note was not saved."))

    if response.original_note:
        typer.secho("\nOriginal Input", fg=typer.colors.BLUE, bold=True)
        typer.echo(_indent(response.original_note))

    if response.note_work:
        typer.secho("\nWork", fg=typer.colors.BLUE, bold=True)
        typer.echo(f"  {response.note_work}")

    if response.note_chunk_ids:
        typer.secho("\nSupporting Chunks", fg=typer.colors.MAGENTA, bold=True)
        for chunk_id in response.note_chunk_ids:
            typer.echo(f"  - {chunk_id}")

    if response.retrieved_chunks:
        typer.secho("\nRetrieved Context", fg=typer.colors.MAGENTA, bold=True)
        for chunk in response.retrieved_chunks:
            _render_chunk(chunk)

    typer.secho("\nRun", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Trace ID       {response.trace_id}")
    typer.echo(f"  Prompt version {response.prompt_version}")


def _render_chunk(chunk: RetrievedChunk) -> None:
    work = chunk.metadata.get("work") or chunk.metadata.get("title") or chunk.source_id
    score_parts = [f"score={chunk.combined_score:.3f}"]
    if chunk.vector_score is not None:
        score_parts.append(f"vector={chunk.vector_score:.3f}")
    if chunk.lexical_score is not None:
        score_parts.append(f"lexical={chunk.lexical_score:.3f}")
    if chunk.trigram_score is not None:
        score_parts.append(f"trigram={chunk.trigram_score:.3f}")
    typer.echo(f"  [{chunk.label}] {work}  {chunk.chunk_id}  {' '.join(score_parts)}")
    typer.echo(_indent(_preview(chunk.text), spaces=4))


def _render_retrieved_note(note: RetrievedNote) -> None:
    score_parts = [f"score={note.combined_score:.3f}"]
    if note.vector_score is not None:
        score_parts.append(f"vector={note.vector_score:.3f}")
    if note.lexical_score is not None:
        score_parts.append(f"lexical={note.lexical_score:.3f}")
    if note.trigram_score is not None:
        score_parts.append(f"trigram={note.trigram_score:.3f}")
    typer.echo(f"  [{note.label}] {note.inferred_work}  {note.note_id}  {' '.join(score_parts)}")
    typer.echo(_indent(note.rewritten_note, spaces=4))
    if note.supporting_chunks:
        chunk_ids = ", ".join(chunk.chunk_id for chunk in note.supporting_chunks)
        typer.echo(f"    Supporting chunks: {chunk_ids}")


def _render_metrics(title: str, payload: dict[str, Any]) -> None:
    typer.secho(f"\n{title}", fg=typer.colors.GREEN, bold=True)
    for key, value in payload.items():
        if isinstance(value, dict):
            typer.echo(f"  {key.replace('_', ' ').title()}:")
            for child_key, child_value in value.items():
                typer.echo(f"    {child_key}: {child_value}")
        else:
            typer.echo(f"  {key.replace('_', ' ').title():<16} {value}")


def _indent(text: str, spaces: int = 2) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else "" for line in text.splitlines())


def _preview(text: str, limit: int = 260) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 1].rstrip()}..."
