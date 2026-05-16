# LitBot System Overview

LitBot is a compact literary retrieval-augmented generation (RAG) chatbot. It ingests approved literary texts and metadata, chunks the texts into stable citation units, stores documents and chunks in first-party PostgreSQL tables with pgvector embeddings, retrieves evidence with hybrid semantic and lexical search, and asks an OpenAI chat model to answer with grounded citations.

The system is intentionally small right now: it has a FastAPI API, a Typer CLI, local corpus ingestion, PostgreSQL/pgvector storage, LangChain integrations at model/prompt/chunking boundaries, structured generation, citation validation, lightweight evaluation, and structured logging. It does not yet include authentication, user accounts, a web UI, production deployment automation, advanced reranking, long-term conversation memory, or production monitoring.

## Major Components

- **API (`litbot/api/main.py`)**: exposes `/health` and `/chat`. The `/chat` endpoint validates non-blank questions, creates or accepts a trace ID, retrieves evidence, logs the request, and returns a generated answer. Unexpected exceptions are logged and returned as structured JSON errors.
- **CLI (`litbot/cli.py`)**: provides local commands for serving the API, ingesting one document, reindexing a corpus directory, asking one question, and scoring answer exports.
- **Configuration (`litbot/config.py`)**: loads runtime settings from environment variables, including database URL, OpenAI chat model, embedding model/dimensions, retrieval depth, prompt version, request timeout, and API key.
- **Database access (`litbot/db.py`)**: manages a process-wide psycopg connection pool for application SQL.
- **LangChain adapter layer (`litbot/langchain.py`)**: centralizes OpenAI embeddings, OpenAI chat model creation, batch/query embedding helpers, and conversion from database rows to `RetrievedChunk`. It no longer owns storage setup or PGVector tables.
- **Ingestion (`litbot/ingestion/`)**: parses `.txt`, `.md`, `.html`, and `.pdf` files with JSON sidecar metadata; normalizes text; chunks text with stable IDs and metadata; deletes previous rows for a source; embeds chunk texts; and inserts into `documents` and `chunks`.
- **Retrieval (`litbot/retrieval/service.py`)**: combines pgvector semantic similarity over `chunks.embedding` with PostgreSQL full-text search over `chunks.text`, merges normalized scores, applies metadata filters against `chunks.metadata`, ranks chunks, records match reasons, and labels returned evidence as `S1`, `S2`, and so on.
- **Generation (`litbot/generation/`)**: builds the grounded prompt, invokes an OpenAI chat model through LangChain structured output, collects `answer`, `citation_map`, and `unsupported`, and validates citation labels against retrieved chunks.
- **Models (`litbot/models.py`)**: defines Pydantic schemas for metadata, parsed documents, chunks, retrieved chunks, chat requests, citations, and chat responses.
- **Evaluation (`litbot/evaluation/golden.py`)**: scores JSONL answer exports with simple answerability, citation, and unsupported-claim metrics.
- **Observability (`litbot/observability/logging.py`)**: configures structured JSON logs with structlog.

## Request Lifecycle

1. A caller sends `POST /chat` with a question, optional metadata filters, and an optional `top_k` override.
2. The API rejects blank questions with a 422 response.
3. The API creates a trace ID if `X-Trace-Id` was not supplied.
4. The API opens a PostgreSQL connection from the process-wide pool.
5. `RetrievalService` embeds the question with LangChain `OpenAIEmbeddings`.
6. Retrieval runs vector search against `chunks.embedding` and lexical search against `chunks.text`, applying the same normalized metadata filters to both SQL queries.
7. Retrieval merges vector and lexical candidates by `chunk_id`, normalizes score sets, computes a combined score, sorts results, records a match reason, and assigns labels such as `S1` and `S2`.
8. The API logs the request and retrieved chunk count.
9. `GenerationService` builds a prompt that contains the user question and the retrieved source payload.
10. The OpenAI chat model is invoked through LangChain structured output and is expected to return JSON-compatible fields: `answer`, `citation_map`, and `unsupported`.
11. LitBot validates citations by checking bracketed labels in the answer, or labels from `citation_map` as a fallback, against the retrieved chunks.
12. The API returns `ChatResponse` with the answer, validated citations, retrieved chunks, prompt version, trace ID, unsupported claims, and timestamp.

## Ingestion Flow

1. `litbot ingest` or `litbot reindex` receives a source path.
2. The parser loads the adjacent JSON metadata sidecar and validates it with `DocumentMetadata`.
3. The parser extracts text from TXT/MD directly, strips common HTML chrome for HTML, or extracts page text from PDFs.
4. Text is normalized at the ingestion boundary so downstream chunk hashes are deterministic for the normalized content.
5. The chunker uses LangChain `RecursiveCharacterTextSplitter` with LitBot token estimation, default target size of 550 estimated tokens, and default overlap of 80 estimated tokens.
6. For poetry, LitBot groups lines into small stanza-like units before splitting so line structure remains visible in retrieved evidence.
7. Each chunk receives flattened metadata, a deterministic `chunk_id`, token count, and SHA-256 content hash.
8. Reingestion is source-id idempotent: existing `documents` rows for that `source_id` are deleted, cascading to `chunks`, before fresh rows are inserted.
9. The source metadata is upserted into `documents` and the generated primary key is used as `chunks.document_id`.
10. Chunk text is embedded with LangChain `OpenAIEmbeddings` in batches of up to 500 inputs.
11. Chunk rows are inserted directly with psycopg into `chunks`, including `chunk_id`, `source_id`, text, token count, chunk hash, vector embedding, and JSONB metadata.
12. `litbot reindex` deletes all rows from `documents` before walking the corpus directory and ingesting files with sidecar metadata.

## Retrieval Flow

1. Filters are normalized by dropping `None` values.
2. The question is embedded with LangChain `OpenAIEmbeddings` using the configured embedding model and dimensions.
3. Vector retrieval queries `chunks` with `ORDER BY embedding <=> %s::vector`, returning `1 - distance` as `vector_score` for up to `top_k * 3` candidates.
4. Lexical retrieval queries `chunks` with `to_tsvector('english', text) @@ plainto_tsquery('english', question)`, returning `ts_rank_cd` as `lexical_score` for up to `top_k * 3` candidates.
5. Metadata filters are translated into SQL predicates against `chunks.metadata`: scalar values compare `metadata->>key` as strings, and list/object values use JSONB containment.
6. Results are merged by `chunk_id`; chunks found by both passes keep both scores, vector-only chunks have no lexical score, and lexical-only chunks use `vector_score = 0.0`.
7. Vector and lexical score sets are each normalized before applying the weighted sum: `0.75 * normalized_vector + 0.25 * normalized_lexical`.
8. Each returned chunk gets a `reason` of `hybrid vector + lexical match`, `vector match only`, or `lexical match only`.
9. The sorted top `k` chunks are labeled `S1`, `S2`, and so on.

## Storage Model

The migration creates the runtime schema. The important tables are:

- `documents`: one row per approved source, keyed by unique `source_id`, with bibliographic metadata, license, URI, version, JSONB metadata, and ingestion timestamp.
- `chunks`: one row per retrievable citation unit, keyed by unique `chunk_id`, with `document_id`, `source_id`, `chunk_index`, text, token count, chunk hash, `embedding vector(1536)`, JSONB metadata, and creation timestamp.

Important indexes:

- `chunks_document_index_idx`: unique `(document_id, chunk_index)` for stable per-document ordering.
- `chunks_embedding_hnsw_idx`: HNSW cosine index for pgvector nearest-neighbor search.
- `chunks_metadata_gin_idx` and `documents_metadata_gin_idx`: JSONB GIN indexes for metadata filters.
- `chunks_text_fts_idx`: GIN full-text index over `to_tsvector('english', text)`.

LangChain PGVector tables are no longer part of the active storage path. LangChain remains in use for OpenAI wrappers, prompt templates, structured output, and recursive text splitting.

## Data Boundaries

- **External input boundary**: corpus files and sidecar JSON metadata are parsed and validated before chunking.
- **Persistence boundary**: only the LitBot schema in `documents` and `chunks` is the source of truth for retrieval.
- **Model boundary**: OpenAI embeddings and chat completions are accessed through LangChain wrappers.
- **Citation boundary**: the generator sees only labeled retrieved chunks; final citations must match labels returned by retrieval.

## Local Runtime Dependencies

- PostgreSQL with the `vector` and `pg_trgm` extensions.
- The migration in `migrations/001_init.sql` must be applied before ingestion.
- `OPENAI_API_KEY` is required for ingestion and answer generation.
- Docker Compose is provided for local PostgreSQL, but migrations are still applied explicitly with `psql`.

## Current Non-Goals

LitBot currently does not provide:

- user authentication or authorization;
- a browser UI;
- hosted deployment manifests;
- multi-turn conversation memory;
- query decomposition or planning;
- cross-encoder or LLM reranking;
- production-grade evaluation dashboards;
- ingestion audit tables beyond the current `documents`/`chunks` schema;
- automatic migration management.
