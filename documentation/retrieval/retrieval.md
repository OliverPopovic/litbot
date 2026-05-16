# Retrieval Baseline

This document captures the current baseline for LitBot's retrieval mechanism. It is descriptive, not aspirational: it records what the code does today so future retrieval experiments can be compared against a stable reference point.

## Baseline Summary

LitBot currently uses a single-pass hybrid retrieval flow:

1. Normalize user metadata filters by dropping `None` values.
2. Run semantic search through LangChain PGVector.
3. Run lexical search through PostgreSQL full-text search over the same LangChain PGVector storage table.
4. Merge both result sets by `chunk_id`.
5. Compute one weighted combined score per chunk.
6. Sort by combined score.
7. Return the top `k` chunks labeled `S1`, `S2`, and so on.

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
| `LITBOT_EMBEDDING_DIMENSIONS` | `1536` | Embedding dimensionality passed to OpenAI embeddings and PGVector. |
| `LITBOT_VECTOR_COLLECTION_NAME` | `litbot_chunks` | LangChain PGVector collection name. |
| `LITBOT_TOP_K` | `8` | Default number of chunks returned when `top_k` is not provided. |
| `LITBOT_DATABASE_URL` | `postgresql://litbot:litbot@localhost:5432/litbot` | PostgreSQL database used by PGVector and lexical SQL. |

## Storage Baseline

Retrieval reads from LangChain-owned PGVector tables rather than first-party `documents` or `chunks` tables.

The active storage path is:

1. Ingestion parses a source document and splits it into stable `TextChunk` records.
2. Each `TextChunk` is converted to a LangChain `Document`.
3. The LangChain `Document` is written into the configured PGVector collection.
4. Retrieval reads the stored document text and JSONB metadata from `langchain_pg_embedding`, joined to `langchain_pg_collection` by collection UUID.

The metadata stored on each LangChain document includes at least:

- `chunk_id`
- `source_id`
- `chunk_index`
- `token_count`
- `chunk_hash`
- flattened source metadata such as work, author, title, genre, license, or custom sidecar fields when present

The current implementation also has custom SQL that depends directly on LangChain's table layout. That coupling is part of the baseline.

## Chunking Context Feeding Retrieval

Retrieval quality depends on the current chunking baseline:

- Documents are split with LangChain `RecursiveCharacterTextSplitter`.
- Default target chunk size is `550` estimated tokens.
- Default overlap is `80` estimated tokens.
- Token counts are approximate and based on a local regex tokenizer, not the OpenAI tokenizer.
- Poetry and poem genres receive a small pre-pass that groups lines into stanza-like units of up to eight non-empty lines before recursive splitting.
- Chunk IDs are deterministic for stable source text and splitting settings: `<source_id>:<chunk_index>:<first_10_chars_of_chunk_hash>`.

## Semantic Retrieval Pass

The semantic pass uses LangChain PGVector:

- Vector store: `langchain_postgres.PGVector`.
- Embeddings: `langchain_openai.OpenAIEmbeddings`.
- Distance strategy: cosine distance.
- Query method: `similarity_search_with_score`.
- Candidate count: `top_k * 3`.

The vector store returns distances. LitBot converts each returned distance into a similarity-like score with:

```text
vector_score = max(0.0, 1.0 - distance)
```

This conversion preserves the previous retriever's expectation that higher scores rank better.

## Lexical Retrieval Pass

The lexical pass is custom PostgreSQL SQL over the same LangChain PGVector table. It exists so exact names, quotations, and phrasing can compete with semantic matches.

Before querying, LitBot ensures this GIN full-text index exists:

```sql
CREATE INDEX IF NOT EXISTS langchain_pg_embedding_document_fts_idx
ON langchain_pg_embedding
USING gin (to_tsvector('english', document))
```

The lexical query:

- joins `langchain_pg_embedding` to `langchain_pg_collection`;
- restricts rows to the configured collection name;
- applies metadata predicates when filters are present;
- matches `to_tsvector('english', e.document)` against `plainto_tsquery('english', question)`;
- ranks matches with `ts_rank_cd`;
- returns up to `top_k * 3` lexical candidates.

The current baseline uses PostgreSQL's English text search configuration and plain query parsing. It does not use phrase search, trigram search, fuzzy matching, synonym dictionaries, or custom literary-language dictionaries.

## Metadata Filter Handling

Filter handling has two dialects because semantic and lexical retrieval use different APIs.

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

### PGVector Filter Translation

For the semantic pass, filters are translated into LangChain PGVector's JSONB filter dialect:

- a single filter becomes `{field: {"$eq": value}}`;
- multiple filters become an `$and` list of equality checks.

### SQL Filter Translation

For the lexical pass, filters are translated into SQL predicates against `e.cmetadata`:

- scalar values use `e.cmetadata->>%s = %s` and compare as strings;
- list or object values use JSONB containment with `e.cmetadata @> %s::jsonb`.

Both passes receive the same normalized filter object, but exact semantics may differ slightly because one path uses LangChain's filter dialect and the other uses direct SQL.

## Merge And Ranking Formula

After both retrieval passes finish, results are merged by chunk ID.

The merge rules are:

- A vector result starts with its converted vector score and no lexical score.
- A lexical-only result starts with `vector_score = 0.0` and its lexical score.
- A chunk returned by both passes keeps one document payload and receives both scores.
- Missing scores are treated as `0.0` for ranking.

The current combined score formula is:

```text
combined_score = (0.75 * vector_score) + (0.25 * min(lexical_score, 1.0))
```

Important implications:

- Semantic similarity is the dominant signal.
- Lexical score contributes at most 25% of the total combined score because it is capped at `1.0` before weighting.
- Lexical-only matches can outrank weak vector matches when their capped lexical contribution is stronger than the vector match's combined score.
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
- `reason`, currently defaulted to `hybrid vector/lexical match`

The labels are generated only for the final top `k` rows after sorting. Generation and citation validation use these labels as the source identifiers exposed to the model.

## Observability Baseline

After retrieval completes, the service logs a `retrieval_completed` event with:

- requested `top_k` limit;
- number of chunks returned;
- normalized filters.

There are no persisted per-query retrieval diagnostics, no recall metrics, no stored rank traces, and no built-in explanation of whether a final chunk came from vector search, lexical search, or both beyond the optional score fields returned on each chunk.

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
- first-party retrieval tables as the source of truth;
- persistent retrieval evaluation datasets;
- retrieval dashboards or quality monitoring.

## Baseline Tests

The current retrieval unit tests cover the core mechanics of the baseline:

- PGVector metadata filter translation.
- Dropping `None` filter values during normalization.
- SQL metadata predicate generation for scalar, list, and object values.
- Hybrid merge behavior, ranking, and final `S1`/`S2` label assignment.

These tests validate ranking mechanics and filter translation, but they do not measure retrieval quality against a golden corpus.

## Change Control Guidance

Future retrieval changes should be compared against this baseline before replacing it. In particular, changes should record:

- whether the candidate pool size changed from `top_k * 3`;
- whether the `0.75` vector / `0.25` lexical weighting changed;
- whether lexical scoring changed from PostgreSQL `ts_rank_cd`;
- whether metadata filter semantics changed;
- whether the storage source of truth changed away from LangChain PGVector tables;
- whether evaluation now includes retrieval recall, citation precision, answer groundedness, latency, or cost.
