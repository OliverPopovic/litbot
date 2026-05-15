# LitBot

LitBot is a small literary RAG chatbot. It ingests approved literary texts, chunks them with
stable citation metadata, stores them in PostgreSQL through LangChain PGVector, and answers
questions with grounded citations.

The current implementation is intentionally compact: it is an API and CLI around a local corpus,
not a finished product with authentication, user accounts, UI, advanced reranking, or production
monitoring.

## What It Does

- Parses `.txt`, `.md`, `.html`, and `.pdf` files with JSON metadata sidecars.
- Splits documents with LangChain text splitters while preserving LitBot's stable citation
  metadata and poetry-aware line grouping.
- Stores chunks as LangChain `Document` objects in a PGVector collection.
- Retrieves with hybrid search:
  - LangChain PGVector similarity search for semantic matching.
  - PostgreSQL full-text search for names, quotes, and exact phrasing.
- Sends retrieved chunks to an OpenAI chat model through LangChain.
- Requires structured model output containing `answer`, `citation_map`, and `unsupported`.
- Validates citation labels against the retrieved chunks before returning the response.

## Tools

- **FastAPI:** implemented for `/health` and `/chat`; lifecycle cleanup uses FastAPI lifespan.
- **Typer:** implemented for `serve`, `ingest`, `reindex`, `ask`, and `eval` CLI commands.
- **Pydantic:** implemented for request, response, metadata, chunk, and citation models.
- **LangChain OpenAI:** implemented for chat models, embeddings, prompt templates, and structured
  model output.
- **LangChain text splitters:** implemented for document chunking. LitBot still adds custom
  citation IDs, metadata flattening, token estimation, and poetry pre-processing around it.
- **LangChain PGVector:** implemented for vector storage and semantic retrieval. Source deletion
  and lexical search still query LangChain PGVector tables directly, so those adapters should be
  revisited if LangChain exposes a cleaner abstraction for those operations.
- **PostgreSQL 16 + pgvector:** implemented through Docker Compose and migration setup.
- **PostgreSQL full-text search:** implemented for lexical retrieval. This is custom SQL by
  design today, not a LangChain retriever abstraction.
- **Beautiful Soup + pdfplumber:** implemented for HTML and PDF parsing; TXT/MD use direct UTF-8
  file reads.
- **pytest + Ruff:** implemented for tests and linting.
- **LangGraph:** not implemented. Add it only if the generation/retrieval flow grows into a
  multi-step state graph that benefits from orchestration.
- **Deployment tooling:** not implemented beyond local Docker Compose for Postgres and Uvicorn for
  local serving.

## Project Layout

- `litbot/api/`: FastAPI app with `/health` and `/chat`.
- `litbot/cli.py`: `serve`, `ingest`, `reindex`, `ask`, and `eval` commands.
- `litbot/ingestion/`: parsing, normalization, chunking, and LangChain document storage.
- `litbot/retrieval/`: hybrid PGVector plus PostgreSQL full-text retrieval.
- `litbot/generation/`: LangChain prompt construction, structured generation, and citation validation.
- `litbot/langchain.py`: LangChain factories and conversion helpers.
- `litbot/models.py`: Pydantic request, response, metadata, chunk, and citation models.
- `corpus/`: small public-domain sample corpus.
- `examples/`: tiny Frankenstein excerpt for quick ingestion tests.
- `migrations/`: PostgreSQL extension/index setup. LangChain creates its own PGVector tables.

## Setup

Create an environment file:

```bash
touch .env
```

Set `OPENAI_API_KEY` in `.env` before running ingestion or generation.

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Install the package:

```bash
uv sync --extra dev
```

If you are not using uv:

```bash
python -m pip install -e '.[dev]'
```

## Ingest Documents

Ingest one document:

```bash
uv run litbot ingest examples/frankenstein_excerpt.txt
```

Recreate the LangChain PGVector collection and ingest the local corpus:

```bash
uv run litbot reindex corpus
```

Ingestion expects a metadata sidecar next to the source file:

```text
examples/frankenstein_excerpt.txt
examples/frankenstein_excerpt.txt.json
```

Required metadata fields are enforced by the Pydantic `DocumentMetadata` model in
`litbot/models.py`:

- `source_id`
- `title`
- `author`
- `publication_year`
- `genre`
- `language`
- `license`
- `uri`
- `version`
- `metadata.work`

## Run The API

Start the server:

```bash
uv run litbot serve --port 8000
```

Ask a question:

```bash
curl -s http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question":"How does Victor describe the creature when it comes to life?","filters":{"work":"Frankenstein"}}'
```

The response includes the generated answer, validated citations, retrieved chunks, unsupported
claims, prompt version, and trace ID.

## CLI Usage

Ask from the terminal:

```bash
uv run litbot ask "How does Victor react when the creature comes to life?"
```

Score a JSONL answer export with the lightweight evaluator:

```bash
uv run litbot eval path/to/answers.jsonl
```

## Configuration

Configuration is loaded from environment variables, with `LITBOT_` prefixes where applicable.

- `OPENAI_API_KEY`: required for OpenAI embeddings and chat generation.
- `LITBOT_DATABASE_URL`: PostgreSQL connection string.
- `LITBOT_LLM_MODEL`: OpenAI chat model. Default: `gpt-4.1-mini`.
- `LITBOT_EMBEDDING_MODEL`: OpenAI embedding model. Default: `text-embedding-3-small`.
- `LITBOT_EMBEDDING_DIMENSIONS`: embedding dimension. Default: `1536`.
- `LITBOT_VECTOR_COLLECTION_NAME`: LangChain PGVector collection. Default: `litbot_chunks`.
- `LITBOT_TOP_K`: default number of retrieved chunks. Default: `8`.
- `LITBOT_PROMPT_VERSION`: prompt version label returned in responses.

## Current Retrieval And Generation Flow

1. The user question is embedded by LangChain PGVector during similarity search.
2. Metadata filters are translated into PGVector JSONB filters.
3. A lexical search runs against `langchain_pg_embedding.document`.
4. Vector and lexical results are merged with the current weighted formula.
5. Final chunks are labeled `S1`, `S2`, etc.
6. LangChain builds a chat prompt containing the question and retrieved source payload.
7. `ChatOpenAI.with_structured_output()` returns typed answer data.
8. Citation labels in the answer or citation map are validated against retrieved chunks.

## Implementation Plan

- ~~Implement core RAG path for a local literary corpus.~~
- ~~Implement LangChain framework instead of custom functions where it simplifies the app.~~
  - ~~Use LangChain PGVector for vector storage and semantic search.~~
  - ~~Use LangChain OpenAI wrappers for embeddings and chat generation.~~
  - ~~Use LangChain prompt templates and structured output for generation.~~
  - ~~Use LangChain text splitters for chunking.~~
  - Keep improving custom adapters where the app still reaches into LangChain PGVector tables for
    source deletion and lexical search.
- ~~Optimize project structure and simplify framework boundaries.~~
  - ~~Move schema validation into Pydantic models.~~
  - ~~Keep LangChain conversion helpers isolated in `litbot/langchain.py`.~~
  - ~~Keep retrieval, generation, ingestion, API, CLI, and evaluation in separate modules.~~
  - Consider extracting lexical retrieval into its own retriever class if the hybrid search logic
    grows.
- Ensure evaluation is on a good level.
  - ~~Provide a lightweight JSONL evaluator for answer/citation/unsupported counts.~~
  - Add a curated golden set with expected retrieval targets and citation expectations.
  - Add quality checks for groundedness, quote accuracy, and retrieval recall.
- Populate database.
  - ~~Provide sample public-domain corpus files and sidecar metadata.~~
  - ~~Provide `litbot ingest` and `litbot reindex` commands.~~
  - Run `uv run litbot reindex corpus` against the target Postgres instance before demo or
    deployment.
  - Add a larger approved corpus after metadata, licensing, and evaluation expectations are clear.
- Decide whether LangGraph is useful.
  - Current flow is linear enough that LangGraph is not implemented.
  - Revisit LangGraph if the app needs explicit multi-step orchestration, retries, tool routing,
    human review, or evaluation loops.
- Deployment.
  - UX:
    - Current UX is CLI plus API only.
    - Add a minimal web UI or hosted chat surface if non-technical users need access.
  - Publishing:
    - Current setup is local Docker Compose for Postgres and local Uvicorn serving.
    - Add production deployment config, environment management, migrations, logging, and monitoring
      before publishing.

## Tests

Run the suite:

```bash
uv run pytest
```

Run Ruff:

```bash
uv run ruff check .
```

The current tests cover API health, Pydantic metadata validation, parsing, LangChain-backed
chunking, citation validation, LangChain document conversion, prompt assembly, structured
generation mapping, and hybrid retrieval ranking.
