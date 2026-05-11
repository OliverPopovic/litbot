# Literary RAG Chatbot Design

## Goal and scope

Build an API-based GPT retrieval-augmented generation (RAG) chatbot that answers user questions about literary works, authors, themes, characters, poems, historical context, and biographies. The assistant should prioritize grounded responses from an approved corpus, clearly cite sources, and state uncertainty when the corpus does not support an answer.

## High-level architecture

1. **Client application** sends a user question and optional filters such as author, work, genre, school course, language, or date range.
2. **Chat API service** authenticates the request, normalizes the question, applies safety and content policies, and coordinates retrieval and generation.
3. **Retriever** embeds the question, searches a vector database, optionally runs metadata filters, and reranks candidate chunks.
4. **Prompt builder** formats the user question, retrieved context, citation metadata, and response instructions for the LLM.
5. **LLM generation** produces a grounded answer, citing the retrieved chunks and refusing or qualifying unsupported claims.
6. **Citation post-processor** validates cited chunk identifiers, formats bibliographic citations, and attaches source links or page references.
7. **Observability layer** logs latency, retrieval scores, prompt versions, errors, and user feedback without storing sensitive user data unnecessarily.

## Document ingestion pipeline

The ingestion pipeline should be repeatable, idempotent, and versioned so that source updates can be traced to answer changes.

1. **Collect documents** from public-domain texts, licensed ebooks, scholarly notes, biographies, poetry collections, study guides, and institution-approved PDFs or HTML pages.
2. **Normalize text** by extracting clean UTF-8 text, preserving stanza breaks, act/scene divisions, page numbers, headings, footnotes, and translator/editor notes when licensed.
3. **Attach metadata** to every source document, including `source_id`, title, author, translator, editor, publication year, edition, genre, language, license, URI, page range, chapter, act, scene, poem title, and ingestion timestamp.
4. **Split into chunks** using structure-aware rules: poems by stanza or line groups, plays by scene and speaker turns, novels by paragraphs or chapter sections, and biographies by headed sections.
5. **Control chunk size** with a target of 300-800 tokens and 10-20% overlap for prose; use smaller chunks for poetry to avoid breaking line-level meaning.
6. **Generate embeddings** for each chunk with a production embedding model, storing the vector alongside the text, metadata, chunk hash, document version, and token count.
7. **Store chunks** in a vector database that supports approximate nearest-neighbor search, metadata filtering, hybrid lexical search, and namespace separation by corpus or tenant.
8. **Validate ingestion** with checks for empty chunks, duplicate hashes, missing licenses, malformed citations, token outliers, and source counts per work.

## Retrieval pipeline

At query time, retrieval should combine semantic similarity with literary metadata and lexical precision.

1. **Classify the query intent** as plot, character, theme, quote lookup, poem interpretation, biography, comparison, historical context, or bibliography.
2. **Rewrite or expand the query** when useful, adding canonical names, aliases, spellings, and work titles; for example, map “the creature” to “Frankenstein's creature” when the target work is known.
3. **Embed the query** using the same embedding family used for document chunks.
4. **Search the vector database** with optional metadata filters such as author, work, poem, era, language, or grade level.
5. **Run hybrid search** by combining vector similarity with keyword/BM25 matching, especially for names, quotations, uncommon phrases, line numbers, and titles.
6. **Rerank candidates** with a cross-encoder or LLM-based reranker to prioritize chunks that directly answer the question.
7. **Assemble context** from the top 4-10 chunks, respecting a token budget and avoiding near-duplicate chunks from the same passage unless continuity is needed.
8. **Return retrieval diagnostics** such as chunk IDs, scores, metadata, and reasons for inclusion for logging and citation validation.

## Sending retrieved chunks to the LLM

The LLM should receive only the information needed to answer the question and should be explicitly instructed to ground claims in the provided context.

```text
System:
You are a literary research assistant. Answer using the provided sources. If the sources do not contain enough evidence, say what is missing. Do not invent quotes, page numbers, biographical details, or publication facts. Cite every factual claim that depends on the sources.

Developer:
Use concise, student-friendly prose. Separate interpretation from textual evidence. Prefer primary text evidence over secondary commentary. Cite sources in bracketed form such as [S1] or [S2, S4].

User question:
{question}

Retrieved sources:
[S1]
source_id: moby-dick-1851-ch42
work: Moby-Dick
chapter: 42
page: 189
chunk_text: ...

[S2]
source_id: melville-biography-licensed-p12
work: Herman Melville biography
page: 12
chunk_text: ...
```

For structured API implementations, send the prompt as messages and pass the sources as structured JSON in the request body or tool result. Keep each source label stable throughout the response so the post-processor can verify that every citation refers to a retrieved chunk.

## Citation handling and source metadata

Citations should be generated from chunk metadata, not inferred by the model.

- Use short inline citations such as `[S1]` during generation, then convert them to user-facing citations like `Moby-Dick, ch. 42, p. 189` or `Shelley, Frankenstein, vol. 1, ch. 4`.
- Keep quote citations line- or page-specific when possible; poetry citations should preserve poem title and line numbers.
- Store source metadata fields for `source_id`, `chunk_id`, `title`, `author`, `translator`, `edition`, `publication_year`, `publisher`, `license`, `uri`, `page_start`, `page_end`, `chapter`, `act`, `scene`, `line_start`, and `line_end`.
- Require the generation output to include a machine-readable citation map, for example `{ "claim": "...", "sources": ["S1"] }`, when building study tools or teacher-facing products.
- Reject or flag citations that reference chunks not included in the retrieval context.
- If the answer uses general literary knowledge not present in the retrieved chunks, label it as uncited background or run another retrieval pass.

## Recommended tools

- **LLM API:** OpenAI GPT models for answer generation, summarization, query rewriting, and optional reranking.
- **Embeddings:** OpenAI embedding models or another model with strong multilingual semantic retrieval if the corpus includes translated or non-English works.
- **Vector database:** Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch/OpenSearch vector search, or PostgreSQL with pgvector for smaller deployments.
- **Document parsing:** Unstructured, Apache Tika, pdfplumber, Beautiful Soup, Pandoc, or custom TEI/XML parsers for scholarly editions.
- **Chunk orchestration:** LangChain, LlamaIndex, Haystack, or a lightweight custom pipeline for tighter control.
- **Evaluation:** RAGAS, DeepEval, promptfoo, custom golden-question sets, citation precision checks, and human review by literature specialists.
- **Observability:** OpenTelemetry, structured logs, trace IDs, vector search metrics, prompt versioning, and feedback capture.

## Error cases and mitigations

| Error case | Mitigation |
| --- | --- |
| No relevant chunks are found | Ask a clarifying question, broaden metadata filters, or state that the corpus lacks support. |
| Low similarity or conflicting sources | Show uncertainty, compare sources explicitly, and prefer primary texts or authoritative editions. |
| The user requests an exact quote not in context | Run quote-focused lexical search; if still missing, say the exact quote was not found. |
| Retrieved chunks are too long | Summarize or trim around the most relevant sentences before generation while preserving citation IDs. |
| Model hallucinates citations | Validate citations against retrieved source IDs and regenerate with stricter instructions if invalid. |
| Ambiguous titles or characters | Ask for clarification or use metadata to present likely matches, such as different works named “The Tempest.” |
| Copyright-restricted text | Store and retrieve only licensed content, limit displayed excerpts, and cite metadata without exposing disallowed text. |
| Biographical claims vary by source | Cite the source used, include dates and uncertainty, and avoid unsupported speculation. |
| Multilingual or translated passages | Track original language and translator metadata, and avoid mixing editions without warning. |
| Vector database outage | Fall back to cached results, lexical search, or a graceful error message with retry guidance. |

## Concise implementation plan

1. **Define corpus policy:** choose approved literary texts, biographies, editions, translations, and licensing rules.
2. **Build ingestion MVP:** parse documents, normalize text, attach metadata, chunk sources, embed chunks, and store them in a vector database.
3. **Implement retrieval API:** accept a question and filters, run vector plus lexical search, rerank candidates, and return top chunks with metadata.
4. **Implement generation API:** construct grounded prompts, call the LLM, require citations, and post-process citation labels into readable references.
5. **Add evaluation:** create golden questions for plot, themes, characters, poems, author biographies, and quote lookup; measure answer faithfulness and citation accuracy.
6. **Add guardrails:** handle empty retrieval, ambiguous questions, copyrighted content, invalid citations, prompt injection in documents, and unsupported claims.
7. **Ship iteratively:** start with a small public-domain corpus, add monitoring and feedback, then expand to licensed sources and advanced filters.
