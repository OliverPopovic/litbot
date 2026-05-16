# Retrieval Baseline

This document captures the current baseline for LitBot's retrieval mechanism. It is descriptive, not aspirational: it records what the code does today so future retrieval experiments can be compared against a stable reference point.

## Baseline Summary

LitBot currently uses a single-pass hybrid retrieval flow over the first-party `chunks` table:

1. Normalize user metadata filters by dropping `None` values.
2. Embed the user question with LangChain `OpenAIEmbeddings`.
3. Run semantic search with pgvector cosine distance over `chunks.embedding`.
4. Run lexical search with PostgreSQL full-text search over `chunks.text`.
5. Merge both result sets by `chunk_id`.
6. Normalize vector and lexical score sets independently.
7. Compute one weighted combined score per chunk.
8. Sort by combined score.
9. Return the top `k` chunks labeled `S1`, `S2`, and so on.

There is no query rewriting, query decomposition, reranker, diversity pass, feedback loop, or multi-step retrieval planner in the current baseline.

## Runtime Entry Points

The retrieval implementation is centered on `RetrievalService`. The API passes `/chat` request fields directly into the service:

- `question`: the natural-language user question.
- `filters`: optional metadata filters.
- `top_k`: optional result count override.

The CLI `ask` command also uses `RetrievalService`, but it currently calls retrieval with only the question and therefore relies on the configured default `top_k` and no metadata filters.

## Configuration Defaults

The current retrieval-related defaults are:

| Setting | Default | Role |
| --- | --- | --- |
| `LITBOT_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model used by LangChain OpenAI embeddings. |
| `LITBOT_EMBEDDING_DIMENSIONS` | `1536` | Embedding dimensionality passed to OpenAI embeddings; must match `chunks.embedding vector(1536)` unless the migration changes too. |
| `LITBOT_TOP_K` | `8` | Default number of chunks returned when `top_k` is not provided. |
| `LITBOT_DATABASE_URL` | `postgresql://litbot:litbot@localhost:5432/litbot` | PostgreSQL database used for first-party storage, vector search, and lexical SQL. |

`LITBOT_VECTOR_COLLECTION_NAME` may still exist in older environments, but active retrieval no longer uses a LangChain PGVector collection name.

## Storage Baseline

Retrieval reads from LitBot-owned tables created by `migrations/001_init.sql`.

The active storage path is:

1. Ingestion parses a source document and splits it into stable `TextChunk` records.
2. Ingestion deletes the prior `documents` row for the same `source_id`; `ON DELETE CASCADE` removes its old chunks.
3. Ingestion upserts source metadata into `documents` and gets the document primary key.
4. Ingestion embeds chunk text with LangChain `OpenAIEmbeddings` in batches.
5. Ingestion inserts rows directly into `chunks` with psycopg.
6. Retrieval reads chunk text, metadata, and embeddings from `chunks`.

The `documents` table stores source-level metadata, including:

- `source_id`
- `title`
- `author`, `translator`, and `editor` when present
- `publication_year`
- `edition`
- `genre`
- `language`
- `license`
- `uri`
- `version`
- JSONB `metadata`
- `content_hash`
- `ingested_at`

The `chunks` table stores retrievable evidence units, including:

- `chunk_id`
- `document_id`
- `source_id`
- `chunk_index`
- `text`
- `token_count`
- `chunk_hash`
- `embedding vector(1536)`
- JSONB `metadata`
- `created_at`

The chunk metadata includes flattened source metadata such as work, author, title, genre, license, and any custom sidecar fields when present. Retrieval filters operate against this chunk-level JSONB metadata.

## Index Baseline

The migration creates the indexes retrieval depends on:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS chunks_document_index_idx
    ON chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx
    ON chunks USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS chunks_text_fts_idx
    ON chunks USING gin (to_tsvector('english', text));
CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx
    ON documents USING gin (metadata jsonb_path_ops);
```

LitBot does not create lexical indexes at request time. The migration is the source of truth for retrieval schema and indexes.

## Chunking Context Feeding Retrieval

Retrieval quality depends on the current chunking baseline:

- Documents are split with LangChain `RecursiveCharacterTextSplitter`.
- Default target chunk size is `550` estimated tokens.
- Default overlap is `80` estimated tokens.
- Token counts are approximate and based on a local regex tokenizer, not the OpenAI tokenizer.
- Poetry and poem genres receive a small pre-pass that groups lines into stanza-like units of up to eight non-empty lines before recursive splitting.
- Chunk IDs are deterministic for stable source text and splitting settings: `<source_id>:<chunk_index>:<first_10_chars_of_chunk_hash>`.

## Semantic Retrieval Pass

The semantic pass uses direct SQL over pgvector:

- Embeddings: `langchain_openai.OpenAIEmbeddings`.
- Stored vector column: `chunks.embedding`.
- Distance operator: `<=>` with `vector_cosine_ops`.
- Candidate count: `top_k * 3`.

The query embeds the question first, converts the embedding to a pgvector literal, and orders by cosine distance:

```sql
SELECT chunk_id, source_id, text, metadata,
       1 - (embedding <=> %s::vector) AS vector_score
FROM chunks
WHERE 1=1 -- plus metadata predicates when filters exist
ORDER BY embedding <=> %s::vector
LIMIT %s
```

The raw vector score is:

```text
vector_score = 1 - cosine_distance
```

The HNSW index on `chunks.embedding` is intended to support this ordered nearest-neighbor query.

## Lexical Retrieval Pass

The lexical pass is custom PostgreSQL SQL over `chunks.text`. It exists so exact names, quotations, and phrasing can compete with semantic matches.

The lexical query:

- applies metadata predicates when filters are present;
- matches `to_tsvector('english', text)` against `plainto_tsquery('english', question)`;
- ranks matches with `ts_rank_cd`;
- returns up to `top_k * 3` lexical candidates.

Representative SQL:

```sql
SELECT chunk_id, source_id, text, metadata,
       ts_rank_cd(to_tsvector('english', text),
                  plainto_tsquery('english', %s)) AS lexical_score
FROM chunks
WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)
  -- plus metadata predicates when filters exist
ORDER BY lexical_score DESC
LIMIT %s
```

The current baseline uses PostgreSQL's English text search configuration and plain query parsing. It does not use phrase search, trigram search, fuzzy matching, synonym dictionaries, or custom literary-language dictionaries.

## Metadata Filter Handling

Filter handling has one SQL translation path shared by semantic and lexical retrieval.

### Normalization

All filters first pass through normalization:

- `None` values are dropped.
- Other values are retained.

For example:

```json
{"work": "Frankenstein", "author": null}
```

becomes:

```json
{"work": "Frankenstein"}
```

### SQL Filter Translation

Normalized filters are translated into SQL predicates against `chunks.metadata`:

- scalar values use `metadata->>%s = %s` and compare as strings;
- list or object values use JSONB containment with `metadata @> %s::jsonb`.

For example, this filter:

```json
{"work": "Frankenstein", "tags": ["gothic"], "edition": {"volume": 1}}
```

produces predicates equivalent to:

```sql
metadata->>'work' = 'Frankenstein'
AND metadata @> '{"tags": ["gothic"]}'::jsonb
AND metadata @> '{"edition": {"volume": 1}}'::jsonb
```

The same predicates are applied to vector and lexical queries.

## Merge And Ranking Formula

After both retrieval passes finish, results are merged by chunk ID.

The merge rules are:

- A vector result starts with its vector score and no lexical score.
- A lexical-only result starts with `vector_score = 0.0` and its lexical score.
- A chunk returned by both passes keeps one row payload and receives both scores.
- Missing scores are treated as `0.0` for ranking.

Before weighting, score sets are normalized independently:

```text
normalized_vector = vector_score / vector_max
normalized_lexical = lexical_score / lexical_max
```

The combined score formula is:

```text
combined_score = (0.75 * normalized_vector) + (0.25 * normalized_lexical)
```

Important implications:

- Semantic similarity remains the dominant signal.
- Raw PostgreSQL `ts_rank_cd` values are not clamped directly into the weighted sum; they are scaled relative to the best lexical candidate in the merged set.
- Lexical-only matches can outrank weak vector matches when they are the best exact-text candidate.
- The weights are hard-coded today, not configurable through settings.

## Returned Retrieval Payload

The final result is a list of `RetrievedChunk` objects. Each returned chunk includes:

- `label`: assigned after ranking as `S1`, `S2`, etc.
- `chunk_id`
- `source_id`
- `text`
- `metadata`
- `combined_score`
- `vector_score`, when present
- `lexical_score`, when present
- `reason`

The `reason` field describes how the chunk was found:

- `hybrid vector + lexical match`
- `vector match only`
- `lexical match only`

The labels are generated only for the final top `k` rows after sorting. Generation and citation validation use these labels as the source identifiers exposed to the model.

## Observability Baseline

After retrieval completes, the service logs a `retrieval_completed` event with:

- requested `top_k` limit;
- number of chunks returned;
- normalized filters.

There are no persisted per-query retrieval diagnostics, no recall metrics, no stored rank traces, and no built-in dashboard. The returned chunk payload does include scores and `reason`, which can be used for lightweight debugging.

## Known Non-Baseline Features

The following are explicitly not part of the current retrieval baseline:

- cross-encoder reranking;
- LLM reranking;
- Maximal Marginal Relevance or diversity ranking;
- query rewriting;
- query decomposition;
- multi-hop retrieval;
- retry-on-insufficient-evidence retrieval;
- phrase-specific lexical search;
- fuzzy matching or typo tolerance;
- configurable retrieval weights;
- LangChain PGVector storage as the active source of truth;
- persistent retrieval evaluation datasets;
- retrieval dashboards or quality monitoring.

## Baseline Tests

The current retrieval unit tests cover the core mechanics of the baseline:

- Dropping `None` filter values during normalization.
- SQL metadata predicate generation for scalar, list, and object values.
- Hybrid merge behavior, ranking, final `S1`/`S2` label assignment, and match reasons.

These tests validate ranking mechanics and filter translation, but they do not measure retrieval quality against a golden corpus.

## Change Control Guidance

Future retrieval changes should be compared against this baseline before replacing it. In particular, changes should record:

- whether the candidate pool size changed from `top_k * 3`;
- whether the `0.75` vector / `0.25` lexical weighting changed;
- whether score normalization changed;
- whether lexical scoring changed from PostgreSQL `ts_rank_cd`;
- whether metadata filter semantics changed;
- whether the storage source of truth changed away from first-party `documents`/`chunks` tables;
- whether evaluation now includes retrieval recall, citation precision, answer groundedness, latency, or cost.
