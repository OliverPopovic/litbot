# LitBot System Overview

LitBot is a compact literary retrieval-augmented generation (RAG) chatbot. It ingests approved literary texts and metadata, chunks the texts into stable citation units, stores those chunks in PostgreSQL through LangChain PGVector, retrieves evidence with hybrid semantic and lexical search, and asks an OpenAI chat model to answer with grounded citations.

The system is intentionally small right now: it has a FastAPI API, a Typer CLI, local corpus ingestion, PostgreSQL/pgvector storage, LangChain integrations, structured generation, citation validation, lightweight evaluation, and structured logging. It does not yet include authentication, user accounts, a web UI, production deployment automation, advanced reranking, long-term conversation memory, or production monitoring.

## Major Components

- **API (`litbot/api/main.py`)**: exposes `/health` and `/chat`. The `/chat` endpoint creates or accepts a trace ID, retrieves evidence, logs the request, and returns a generated answer.
- **CLI (`litbot/cli.py`)**: provides local commands for serving the API, ingesting one document, reindexing a corpus directory, asking one question, and scoring answer exports.
- **Configuration (`litbot/config.py`)**: loads runtime settings from environment variables, including database URL, OpenAI chat model, embedding model, vector collection name, retrieval depth, prompt version, request timeout, and API key.
- **Database access (`litbot/db.py`)**: manages a process-wide psycopg connection pool for application SQL.
- **LangChain adapter layer (`litbot/langchain.py`)**: centralizes OpenAI embeddings, OpenAI chat model creation, PGVector setup, conversion between LitBot chunks and LangChain documents, source deletion, and the lexical full-text index.
- **Ingestion (`litbot/ingestion/`)**: parses `.txt`, `.md`, `.html`, and `.pdf` files with JSON sidecar metadata; normalizes text; chunks text with stable IDs and metadata; deletes previous chunks for a source; and stores fresh embeddings.
- **Retrieval (`litbot/retrieval/service.py`)**: combines PGVector semantic similarity with PostgreSQL full-text search, merges scores, applies metadata filters, ranks chunks, and labels returned evidence as `S1`, `S2`, and so on.
- **Generation (`litbot/generation/`)**: builds the grounded prompt, invokes an OpenAI chat model through LangChain structured output, collects `answer`, `citation_map`, and `unsupported`, and validates citation labels against retrieved chunks.
- **Models (`litbot/models.py`)**: defines Pydantic schemas for metadata, parsed documents, chunks, retrieved chunks, chat requests, citations, and chat responses.
- **Evaluation (`litbot/evaluation/golden.py`)**: scores JSONL answer exports with simple answerability, citation, and unsupported-claim metrics.
- **Observability (`litbot/observability/logging.py`)**: configures structured JSON logs with structlog.

## Request Lifecycle

1. A caller sends `POST /chat` with a question, optional metadata filters, and an optional `top_k` override.
2. The API creates a trace ID if `X-Trace-Id` was not supplied.
3. The API opens a PostgreSQL connection from the process-wide pool.
4. `RetrievalService` runs vector search through LangChain PGVector and lexical search through custom PostgreSQL full-text SQL over the same LangChain PGVector storage tables.
5. Retrieval merges vector and lexical candidates, computes a combined score, sorts results, and assigns labels such as `S1` and `S2`.
6. The API logs the request and retrieved chunk count.
7. `GenerationService` builds a prompt that contains the user question and the retrieved source payload.
8. The OpenAI chat model is invoked through LangChain structured output and is expected to return JSON-compatible fields: `answer`, `citation_map`, and `unsupported`.
9. LitBot validates citations by checking bracketed labels in the answer, or labels from `citation_map` as a fallback, against the retrieved chunks.
10. The API returns `ChatResponse` with the answer, validated citations, retrieved chunks, prompt version, trace ID, unsupported claims, and timestamp.

## Ingestion Flow

1. `litbot ingest` or `litbot reindex` receives a source path.
2. The parser loads the adjacent JSON metadata sidecar and validates it with `DocumentMetadata`.
3. The parser extracts text from TXT/MD directly, strips common HTML chrome for HTML, or extracts page text from PDFs.
4. Text is normalized at the ingestion boundary so downstream chunk hashes are deterministic for the normalized content.
5. The chunker uses LangChain `RecursiveCharacterTextSplitter` with LitBot token estimation, default target size of 550 estimated tokens, and default overlap of 80 estimated tokens.
6. For poetry, LitBot groups lines into small stanza-like units before splitting so line structure remains visible in retrieved evidence.
7. Each chunk receives flattened metadata, a deterministic `chunk_id`, token count, and SHA-256 content hash.
8. Reingestion is source-id idempotent: existing rows for that `source_id` are deleted from the configured LangChain PGVector collection before new chunks are embedded and stored.
9. LangChain PGVector embeds and persists the chunks as LangChain `Document` objects.

## Retrieval Flow

1. Filters are normalized by dropping `None` values.
2. Vector retrieval calls LangChain PGVector `similarity_search_with_score` with cosine distance and the configured OpenAI embedding model.
3. Lexical retrieval ensures a GIN full-text index exists on `langchain_pg_embedding.document`, then queries `to_tsvector('english', document)` with `plainto_tsquery`.
4. Metadata filters are translated twice: once into LangChain PGVector's metadata filter dialect and once into SQL predicates against `cmetadata`.
5. Vector distance is converted to a similarity score, lexical score is capped at `1.0`, and the current combined score is `0.75 * vector_score + 0.25 * lexical_score`.
6. Results are deduplicated by `chunk_id`, sorted by combined score, trimmed to `top_k`, and labeled sequentially.

## Database Structure

LitBot has two database shapes to keep in mind:

- **Active runtime storage**: LangChain PGVector owns `langchain_pg_collection` and `langchain_pg_embedding`. Current ingestion, retrieval, source deletion, metadata filtering, and lexical search operate against these tables. Chunk text is stored in `document`, metadata is stored in JSONB `cmetadata`, embeddings are stored by PGVector, and collection name comes from `LITBOT_VECTOR_COLLECTION_NAME`.
- **Project migration schema**: `migrations/001_init.sql` creates `documents` and `chunks` tables with pgvector, JSONB metadata indexes, full-text indexes, and document/chunk metadata fields. The current application path does not yet write to those first-party tables; they should be treated as a schema direction or future consolidation point unless the code is updated to use them.

Important configured defaults:

- PostgreSQL URL: `postgresql://litbot:litbot@localhost:5432/litbot`.
- Chat model: `gpt-4.1-mini`.
- Embedding model: `text-embedding-3-small`.
- Embedding dimensions: `1536`.
- Vector collection: `litbot_chunks`.
- Default retrieval depth: `8`.
- Prompt version: `litbot-grounded-v1`.

## Simple System Diagram

```mermaid
flowchart LR
    Source[Approved source file + JSON metadata] --> Parser[Parser + metadata validation]
    Parser --> Chunker[Chunker + stable citation metadata]
    Chunker --> Store[LangChain PGVector in PostgreSQL]
    User[User or CLI question] --> API[FastAPI /chat or CLI ask]
    API --> Retriever[Hybrid retrieval: vector + full-text]
    Store --> Retriever
    Retriever --> Generator[Grounded OpenAI chat generation]
    Generator --> Validator[Citation validation]
    Validator --> Response[ChatResponse with answer, citations, chunks, trace ID]
```

## Current Boundaries and Gaps

- Retrieval and generation are linear, not a multi-step agent or LangGraph workflow.
- Memory is corpus memory only; there is no durable conversation memory or user profile memory.
- Safety is primarily grounded-answer prompting plus citation validation, not a comprehensive moderation, policy, or prompt-injection defense layer.
- Evaluation is a lightweight JSONL scoring utility, not a full regression harness with golden expected answers and retrieval metrics.
- Observability is structured logging with trace IDs, not distributed tracing, metrics, dashboards, or user feedback collection.
- Infrastructure is local-first: Docker Compose provides PostgreSQL, while serving is local Uvicorn through the CLI.
