# LitBot

LitBot is a small literary RAG chatbot. It ingests approved literary texts, chunks them with
stable citation metadata, stores them in PostgreSQL through LangChain PGVector, and answers
questions with grounded citations.

The current implementation is intentionally compact: it is an API and CLI around a local corpus,
not a finished product with authentication, user accounts, UI, advanced reranking, or production
monitoring.

## What It Does

- Parses `.txt`, `.md`, `.html`, and `.pdf` files with JSON metadata sidecars.
- Splits documents into stable chunks while preserving paragraph and poetry boundaries.
- Stores chunks as LangChain `Document` objects in a PGVector collection.
- Retrieves with hybrid search:
  - LangChain PGVector similarity search for semantic matching.
  - PostgreSQL full-text search for names, quotes, and exact phrasing.
- Sends retrieved chunks to an OpenAI chat model through LangChain.
- Requires structured model output containing `answer`, `citation_map`, and `unsupported`.
- Validates citation labels against the retrieved chunks before returning the response.

## Stack

- **API:** FastAPI
- **CLI:** Typer
- **LLM + embeddings:** `langchain-openai`
- **Vector store:** `langchain-postgres` PGVector
- **Database:** PostgreSQL 16 with `pgvector`
- **Parsing:** Beautiful Soup for HTML, pdfplumber for PDF, direct UTF-8 text for TXT/MD
- **Tests:** pytest and Ruff

## Project Layout

- `litbot/api/`: FastAPI app with `/health` and `/chat`.
- `litbot/cli.py`: `serve`, `ingest`, `reindex`, `ask`, and `eval` commands.
- `litbot/ingestion/`: parsing, normalization, chunking, and LangChain document storage.
- `litbot/retrieval/`: hybrid PGVector plus PostgreSQL full-text retrieval.
- `litbot/generation/`: LangChain prompt construction, structured generation, and citation validation.
- `litbot/langchain.py`: LangChain factories and conversion helpers.
- `litbot/models.py`: request, response, chunk, and citation dataclasses.
- `corpus/`: small public-domain sample corpus.
- `examples/`: tiny Frankenstein excerpt for quick ingestion tests.
- `migrations/`: PostgreSQL extension/index setup. LangChain creates its own PGVector tables.

## Setup

Create an environment file:

```bash
cp .env.example .env
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

Required metadata fields are enforced by `litbot/ingestion/parsers.py`:

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

## Tests

Run the suite:

```bash
uv run pytest
```

Run Ruff:

```bash
uv run ruff check .
```

The current tests cover chunking, parsing, citation validation, LangChain document conversion,
prompt assembly, structured generation mapping, and hybrid retrieval ranking.

