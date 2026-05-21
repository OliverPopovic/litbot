# LitBot

LitBot is a small literary RAG chatbot. It ingests approved literary texts, chunks them with
stable citation metadata, stores documents/chunks in first-party PostgreSQL tables with pgvector
embeddings, and answers questions with grounded citations.

The current implementation is intentionally compact: it is an API and CLI around a local corpus,
not a finished product with authentication, user accounts, UI, advanced reranking, or production
monitoring.

## What It Does

- Parses `.txt`, `.md`, `.html`, and `.pdf` files with JSON metadata sidecars.
- Splits documents with LangChain text splitters while preserving LitBot's stable citation
  metadata and poetry-aware line grouping.
- Stores source metadata in `documents` and chunk text, metadata, hashes, token counts, and
  embeddings in `chunks`.
- Retrieves with hybrid search over LitBot-owned tables:
  - pgvector cosine search over `chunks.embedding` for semantic matching.
  - PostgreSQL full-text search over `chunks.text` for names, quotes, and exact phrasing.
  - PostgreSQL trigram search over `chunks.text` for fuzzy phrase and quote matching.
  - Reciprocal Rank Fusion over semantic, full-text, and trigram candidate ranks.
- Sends retrieved chunks to an OpenAI chat model through LangChain.
- Requires structured model output containing `answer`, `citation_map`, and `unsupported`.
- Validates citation labels against the retrieved chunks before returning the response.
- Classifies `/chat` and `litbot ask` inputs as questions, reading-note writes, or saved-note
  retrieval. High-confidence notes are grounded in retrieved corpus chunks, rewritten for concise
  factuality, embedded, and stored globally with supporting chunk links. Note ownership is not
  implemented yet.

## Tools

- **FastAPI:** implemented for `/health` and `/chat`; lifecycle cleanup uses FastAPI lifespan.
- **Typer:** implemented for `serve`, `ingest`, `reindex`, `ask`, and `eval` CLI commands.
- **Pydantic:** implemented for request, response, metadata, chunk, and citation models.
- **LangChain OpenAI:** implemented for chat models, embeddings, prompt templates, and structured
  model output.
- **LangChain text splitters:** implemented for document chunking. LitBot still adds custom
  citation IDs, metadata flattening, token estimation, and poetry pre-processing around it.
- **PostgreSQL 16 + pgvector:** implemented through Docker Compose, migrations, first-party
  `documents`/`chunks` tables, and direct psycopg inserts/queries.
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
- `litbot/chat.py`: Shared question/note orchestration for API and CLI surfaces.
- `litbot/cli.py`: `serve`, `ingest`, `reindex`, `ask`, and `eval` commands.
- `litbot/ingestion/`: parsing, normalization, chunking, embedding, and first-party table storage.
- `litbot/retrieval/`: hybrid pgvector plus PostgreSQL full-text retrieval over `chunks`.
- `litbot/generation/`: LangChain prompt construction, structured generation, and citation validation.
- `litbot/intent.py` and `litbot/notes/`: intent classification plus grounded note rewriting,
  storage, and retrieval.
- `litbot/langchain.py`: LangChain OpenAI factories and retrieval-row conversion helpers.
- `litbot/models.py`: Pydantic request, response, metadata, chunk, and citation models.
- `corpus/`: small public-domain sample corpus.
- `examples/`: tiny Frankenstein excerpt for quick ingestion tests.
- `migrations/`: PostgreSQL extension, table, and index setup for the LitBot schema.

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

Apply the database schema:

```bash
psql $LITBOT_DATABASE_URL -f migrations/001_init.sql
psql $LITBOT_DATABASE_URL -f migrations/002_trigram_index.sql
psql $LITBOT_DATABASE_URL -f migrations/003_global_notes.sql
psql $LITBOT_DATABASE_URL -f migrations/004_pending_note_actions.sql
```

## Ingest Documents

Ingest one document:

```bash
uv run litbot ingest examples/frankenstein_excerpt.txt
```

Download the larger public-domain evaluation corpus:

```bash
uv run litbot fetch-corpus
```

Clear first-party document/chunk rows and ingest the local corpus:

```bash
uv run litbot reindex corpus
```

Reindex the downloaded public-domain corpus:

```bash
uv run litbot reindex .litbot_corpus/public_domain
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
claims, prompt version, and trace ID. Blank questions return a 422 error, and unexpected server
errors return a structured JSON `{"error":"Internal server error"}` response.

Save a reading note through the same `/chat` route:

```bash
curl -s http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question":"Save this note: Hamlet opens with anxious uncertainty at the watch."}'
```

The classifier routes high-confidence note requests to the note workflow. Low-confidence note
classifications fall back to normal question answering so LitBot does not accidentally write or
retrieve notes. A saved note response includes `intent`, `intent_confidence`, `note_status`,
`note_id`, `note`, `original_note`, `note_work`, and `note_chunk_ids`. A rejected note returns
`note_status="not_saved"` and `note_rejection_reason` without inserting rows. Explicit note
retrieval uses `intent="note_query"` and returns `retrieved_notes`; broad note lists are capped
previews in v1.

Edit or delete requests also go through `/chat`. They return
`note_operation_status="pending_confirmation"` and a `pending_note_action_id`; send a second
request with that ID and `confirm_note_action=true` to execute it, or `cancel_note_action=true` to
cancel. Pending actions expire after 10 minutes and can be confirmed only once.

## CLI Usage

Ask from the terminal:

```bash
uv run litbot ask "How does Victor react when the creature comes to life?"
```

Save a reading note from the terminal:

```bash
uv run litbot ask "Save this note: Hamlet opens with anxious uncertainty at the watch."
```

When a note is saved, CLI output highlights the rewritten stored note and shows the original input
separately.

Edit and delete requests from the CLI prompt for confirmation before execution. The CLI keeps the
last retrieved single note in `~/.litbot/state.json`, so follow-up requests like “delete this” can
refer to it; set `LITBOT_CLI_STATE_PATH` to override that state location.

Score a JSONL answer export with the lightweight evaluator:

```bash
uv run litbot eval path/to/answers.jsonl
```

Score retrieval directly against the golden retrieval fixture after reindexing the larger corpus:

```bash
uv run litbot eval-retrieval tests/fixtures/retrieval_golden.jsonl
```

## Configuration

Configuration is loaded from environment variables, with `LITBOT_` prefixes where applicable.

- `OPENAI_API_KEY`: required for OpenAI embeddings and chat generation.
- `LITBOT_DATABASE_URL`: PostgreSQL connection string.
- `LITBOT_LLM_MODEL`: OpenAI chat model. Default: `gpt-4.1-mini`.
- `LITBOT_EMBEDDING_MODEL`: OpenAI embedding model. Default: `text-embedding-3-small`.
- `LITBOT_EMBEDDING_DIMENSIONS`: embedding dimension. Default: `1536`; must match the migration's
  `chunks.embedding vector(1536)` column unless the schema is changed too.
- `LITBOT_TOP_K`: default number of retrieved chunks. Default: `8`.
- `LITBOT_RETRIEVAL_CANDIDATE_MULTIPLIER`: candidate expansion multiplier for each retrieval
  lane. Default: `8`.
- `LITBOT_RETRIEVAL_MIN_CANDIDATES`: minimum candidates per retrieval lane. Default: `50`.
- `LITBOT_RETRIEVAL_MAX_CANDIDATES`: maximum candidates per retrieval lane. Default: `200`.
- `LITBOT_RETRIEVAL_RRF_K`: Reciprocal Rank Fusion constant. Default: `60`.
- `LITBOT_RETRIEVAL_INCLUDE_NEIGHBORS`: include adjacent chunks around retrieved seeds. Default:
  `false`.
- `LITBOT_RETRIEVAL_NEIGHBOR_WINDOW`: adjacent chunk distance when neighbor expansion is enabled.
  Default: `1`.
- `LITBOT_PROMPT_VERSION`: prompt version label returned in responses.
- `LITBOT_NOTE_PROMPT_VERSION`: prompt version label returned for note rewriting/storage.
- `LITBOT_INTENT_CONFIDENCE_THRESHOLD`: minimum classifier confidence required to route to note
  writing or note retrieval. Default: `0.65`; below this LitBot answers as a question.
- `LITBOT_NOTE_QUERY_TOP_K`: maximum notes returned for explicit note retrieval. Default: `20`.
- `LITBOT_QUESTION_NOTE_TOP_K`: maximum supplemental notes shown after ordinary answers. Default:
  `3`.
- `LITBOT_NOTE_CANDIDATE_TOP_K`: note candidates considered before LLM relevance filtering.
  Default: `12`.
- `LITBOT_NOTE_MIN_VECTOR_SCORE`: minimum note vector score for relevance-filter eligibility.
  Default: `0.35`.
- `LITBOT_NOTE_MIN_TRIGRAM_SCORE`: minimum note trigram score for relevance-filter eligibility.
  Default: `0.18`.

## Current Retrieval And Generation Flow

1. The user question is embedded with LangChain `OpenAIEmbeddings`.
2. Metadata filters are normalized and translated into SQL predicates against `chunks.metadata`.
3. Semantic search runs a pgvector cosine-distance query against `chunks.embedding`.
4. Full-text lexical search runs PostgreSQL full-text search against `chunks.text`.
5. Trigram lexical search runs PostgreSQL fuzzy phrase matching against `chunks.text`.
6. Vector, full-text, and trigram result sets are merged by `chunk_id`.
7. Chunks are ranked with Reciprocal Rank Fusion over the candidate ranks.
8. If enabled, neighbor expansion adds adjacent same-document chunks around retrieved seeds.
9. Final chunks are labeled `S1`, `S2`, etc., and each chunk records which lane found it.
10. LangChain builds a chat prompt containing the question and retrieved source payload.
11. `ChatOpenAI.with_structured_output()` returns typed answer data.
12. Citation labels in the answer or citation map are validated against retrieved chunks.

## Global Note Flow

1. `/chat` or `litbot ask` receives the input and creates or propagates a trace ID.
2. `IntentService` classifies the input as `question`, `note`, `note_query`, `note_edit`,
   `note_delete`, or `note_delete_all`.
3. Note classifications below `LITBOT_INTENT_CONFIDENCE_THRESHOLD` fall back to question answering.
4. For note intent, `NoteService` retrieves evidence using the extracted note text and any supplied
   work filter.
5. The note prompt rewrites the note, infers a work from corpus metadata, and returns selected
   supporting chunk IDs.
6. LitBot saves only nonblank rewritten notes with at least one selected retrieved chunk and an
   inferred corpus work.
7. The rewritten note embedding, note metadata, and `note_chunks` links are inserted in one
   transaction, so failed chunk-link inserts roll back the note row too.
8. Edit and delete requests create a pending action first; confirmation locks and consumes that
   action in the same transaction as the edit or hard delete.

Explicit `note_query` requests return stored note text directly, with linked corpus chunks when
available. Ordinary question answering can append strictly relevant saved notes after the cited
answer; notes are not treated as corpus evidence or citations. See `docs/note/notes.md` for the
design details and the current global-note ownership limitation.

## Implementation Plan

- ~~Implement core RAG path for a local literary corpus.~~
- ~~Implement LangChain framework where it simplifies model, prompt, and chunking boundaries.~~
  - ~~Use LangChain OpenAI wrappers for embeddings and chat generation.~~
  - ~~Use LangChain prompt templates and structured output for generation.~~
  - ~~Use LangChain text splitters for chunking.~~
  - ~~Move vector storage and retrieval onto LitBot-owned PostgreSQL tables.~~
  - ~~Keep storage-specific SQL in ingestion/retrieval modules instead of `litbot/langchain.py`.~~
- ~~Optimize project structure and simplify framework boundaries.~~
  - ~~Move schema validation into Pydantic models.~~
  - ~~Keep LangChain helpers isolated in `litbot/langchain.py`.~~
  - ~~Keep retrieval, generation, ingestion, API, CLI, and evaluation in separate modules.~~
  - Consider extracting lexical retrieval into its own retriever class if the hybrid search logic
    grows.
- Ensure evaluation is on a good level.
  - ~~Provide a lightweight JSONL evaluator for answer/citation/unsupported counts.~~
  - Add a curated golden set with expected retrieval targets and citation expectations.
  - Add quality checks for groundedness, quote accuracy, and retrieval recall.
- Broaden regression coverage around runtime boundaries.
  - Add mocked end-to-end `/chat` route tests that exercise request validation, retrieval,
    generation, trace IDs, and response serialization without requiring live OpenAI or Postgres.
  - Add database-backed integration tests for migrations, ingestion idempotency, metadata filters,
    vector/lexical retrieval, and connection-pool lifecycle behavior.
  - Keep unit tests for settings, request models, retrieval ranking, and citation validation close
    to the modules they protect.
- Harden API error handling.
  - Expand FastAPI exception handlers beyond the generic 500 handler to cover validation,
    database, retrieval, generation, and citation-validation failures with predictable response
    bodies and trace-aware logging.
- Populate database.
  - ~~Provide sample public-domain corpus files and sidecar metadata.~~
  - ~~Provide `litbot ingest` and `litbot reindex` commands.~~
  - Run `psql $LITBOT_DATABASE_URL -f migrations/001_init.sql` before first ingestion.
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
chunking, citation validation, retrieval-row conversion, prompt assembly, structured generation
mapping, metadata filter SQL generation, and hybrid retrieval ranking.
