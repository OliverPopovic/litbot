# AI Stack Decision Points

This is not a final architecture decision record. It is a review aid extracted from the current system overview and codebase so the project owner can decide what should remain simple, what should be made explicit, and what should change before LitBot grows.

## Current AI Stack Snapshot

| Area | Current state | Decision status |
| --- | --- | --- |
| LLM | OpenAI chat model through LangChain, default `gpt-4.1-mini`, temperature `0.2`, structured output schema. | Good enough for early RAG; model choice and quality/cost targets need review. |
| Embeddings | OpenAI `text-embedding-3-small`, 1536 dimensions. | Practical default; needs benchmark against corpus and query types. |
| Retrieval memory | PostgreSQL + LangChain PGVector collection, with semantic and lexical retrieval over chunk text and JSONB metadata. | Strong simple baseline; storage ownership needs clarification. |
| Conversation memory | None. Each request is stateless except for corpus retrieval. | Explicit product decision needed. |
| Tools/integrations | FastAPI, Typer, LangChain OpenAI, LangChain PGVector, PostgreSQL full-text search, Beautiful Soup, pdfplumber. | Minimal and understandable; future integrations should be added only for specific gaps. |
| Planning/orchestration | Linear retrieve-then-generate flow. LangGraph is not implemented. | Keep linear unless real multi-step needs appear. |
| Guardrails/safety | Grounded prompt, structured output, citation validation, no-answer behavior when no chunks are retrieved. | Useful baseline; not enough for hostile or production settings. |
| Feedback/evaluation | Structured logs and basic JSONL scoring for answerability, citation presence, and unsupported fields. | Needs a reproducible evaluation plan before optimizing retrieval or models. |
| Infrastructure | Local Docker Compose PostgreSQL and local Uvicorn serving. | Fine for development; production path undecided. |

## Decision Points for Review

### 1. LLM model policy

**Facts today**

- The default chat model is `gpt-4.1-mini`.
- Generation uses LangChain `ChatOpenAI` with temperature `0.2` and a request timeout setting.
- The model is asked for structured output with `answer`, `citation_map`, and `unsupported`.

**Decision needed**

Choose a model policy instead of a single implicit default. For example:

- Keep one low-cost default model for all requests.
- Add a higher-quality model option for difficult interpretive questions.
- Route by task type, corpus size, or user-selected quality level.

**Questions to answer**

- What matters most right now: cost, latency, citation fidelity, literary reasoning quality, or reproducibility?
- Should prompt and model versions be pinned together in evaluation reports?
- Should temperature stay nonzero for readability, or move closer to deterministic behavior for reproducible experiments?

### 2. Embedding and chunking strategy

**Facts today**

- Embeddings use `text-embedding-3-small` with 1536 dimensions.
- Chunks target about 550 estimated tokens with 80 estimated tokens of overlap.
- Chunk IDs are deterministic for normalized text and split output.
- Poetry gets a small line-grouping pre-pass.

**Decision needed**

Decide whether the current chunking and embedding defaults are the long-term baseline or simply a starting point.

**Questions to answer**

- Do literary questions need larger chunks for context or smaller chunks for precise citations?
- Should prose, drama, poetry, and criticism use different chunking profiles?
- Should evaluation track retrieval recall by source, passage, and citation label before changing embeddings?
- Is one embedding model enough, or do names, quotes, and archaic language require a different retrieval strategy?

### 3. Retrieval ranking and reranking

**Facts today**

- Retrieval is hybrid: semantic PGVector search plus PostgreSQL full-text search.
- Combined score currently weights vector similarity at 75% and lexical score at 25%.
- Lexical search is valuable for exact names, quotes, and phrasing.
- There is no cross-encoder reranker, LLM reranker, diversity pass, or query rewriting.

**Decision needed**

Decide how much retrieval quality to buy with complexity.

**Questions to answer**

- Are current answers failing because retrieval misses evidence, or because generation misuses evidence?
- Should ranking weights be made configurable and evaluated experimentally?
- Should there be per-query retrieval diagnostics that explain which chunks came from vector search, lexical search, or both?
- Is reranking worth the extra latency and cost for the intended users?

### 4. Storage ownership: LangChain tables versus first-party schema

**Facts today**

- The active code writes to LangChain PGVector tables.
- Custom source deletion and lexical search query LangChain-owned tables directly.
- The migration creates first-party `documents` and `chunks` tables, but the current ingestion path does not write to them.

**Decision needed**

Pick one storage source of truth.

**Options**

- Continue using LangChain PGVector tables and document the coupling.
- Move to first-party `documents` and `chunks` tables and use LangChain only at the boundaries.
- Keep LangChain PGVector for embeddings but add first-party tables for document registry, ingestion audit, and corpus governance.

**Questions to answer**

- How important is avoiding direct dependencies on LangChain table internals?
- Do we need ingestion history, source versioning, and auditability soon?
- Should metadata validation failures and license decisions be persisted?

### 5. Memory model

**Facts today**

- LitBot has durable corpus memory through the vector database.
- LitBot does not have durable conversation memory, user memory, session summaries, or preferences.

**Decision needed**

Define whether LitBot should remember only texts or also remember users and conversations.

**Questions to answer**

- Is LitBot intended to be a research assistant that supports multi-turn projects?
- If yes, should memory be explicit user-saved notes rather than automatic hidden memory?
- What privacy, deletion, and export expectations come with conversation memory?

### 6. Tools and external integrations

**Facts today**

- Ingestion reads local files with sidecar metadata.
- There is no library catalog integration, web search, citation resolver, OCR pipeline, LMS integration, or UI integration.

**Decision needed**

Decide whether LitBot stays a closed-corpus assistant or becomes an integrated research workflow tool.

**Questions to answer**

- Should answers ever use sources outside the approved corpus?
- Should users be able to upload sources, or should ingestion remain curated/admin-only?
- Which integrations would improve reproducibility rather than add surface area?

### 7. Planning and orchestration

**Facts today**

- The request path is one retrieval pass followed by one generation pass.
- There is no explicit planner, query decomposition, multi-hop loop, retry loop, or tool router.

**Decision needed**

Keep orchestration simple unless there is evidence that linear RAG cannot handle the intended questions.

**Questions to answer**

- What question types require multi-step planning: comparison across works, quote finding, theme synthesis, or bibliography tasks?
- Should query decomposition happen before retrieval for comparative questions?
- Should the system retry retrieval when generated output reports missing evidence?

### 8. Guardrails and safety

**Facts today**

- Prompts instruct the model to use only provided sources and not invent citations.
- Retrieved text is explicitly described as evidence, not instructions.
- Citation labels are validated after generation.
- Unsupported claims can be returned, but there is no automated policy layer or adversarial prompt-injection test suite.

**Decision needed**

Define the safety level needed for the intended deployment.

**Questions to answer**

- Should unsupported answers be blocked, flagged, or simply returned with caveats?
- Should the API reject empty, abusive, or prompt-injection-like questions?
- Should retrieved corpus text be treated as potentially malicious even if the corpus is curated?
- Should there be automated tests for citation fabrication and instruction injection?

### 9. Observation, feedback, and evaluation loop

**Facts today**

- Responses include trace IDs and prompt version.
- Logs record retrieval and generation events.
- The evaluator counts answered rows, citation presence, and unsupported rows.
- There is no persistent feedback table, retrieval recall dataset, answer-quality rubric, or dashboard.

**Decision needed**

Create a reproducible evaluation loop before making large retrieval, prompt, or model changes.

**Questions to answer**

- What is the first golden dataset: factual lookup, quote identification, interpretation, comparison, or all of these?
- Should each golden question include expected source IDs/chunks, expected answer traits, or both?
- What should be measured: retrieval recall, citation precision, answer groundedness, unsupported rate, latency, and cost?
- Where should failed examples be stored so they become regression tests rather than anecdotes?

### 10. Infrastructure and deployment

**Facts today**

- PostgreSQL runs locally through Docker Compose.
- Uvicorn serves the FastAPI app locally through the CLI.
- There is no production deployment config, migration runner, secrets management plan, authentication, rate limiting, or monitoring.

**Decision needed**

Decide whether LitBot is still a local research prototype or is moving toward a hosted product/service.

**Questions to answer**

- Who is the first real user: developer, researcher, classroom, or public web user?
- Does the system need authentication before any hosted deployment?
- Should migrations be formalized before the storage schema is changed?
- What operational metrics matter first: latency, error rate, retrieval quality, token cost, or user feedback?

## Suggested Near-Term Decision Order

1. Define the first evaluation set and metrics.
2. Decide whether active storage should remain LangChain-owned or move toward first-party tables.
3. Decide whether LitBot remains stateless or needs explicit session/project memory.
4. Benchmark the current LLM, embedding, chunking, and retrieval weights before replacing them.
5. Add guardrail tests that match the expected deployment risk.
6. Revisit orchestration only after evaluation shows that linear RAG is insufficient.
