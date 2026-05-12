# LitBot Corpus

This folder contains local source files that can be ingested into the RAG database.

## Public-domain workflow

Add one source file plus one JSON metadata sidecar:

```text
corpus/public_domain/pride-prejudice-1813-ch1.txt
corpus/public_domain/pride-prejudice-1813-ch1.txt.json
```

Supported source formats are `.txt`, `.md`, `.html`, and `.pdf`.

Required metadata fields:

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

Use stable `source_id` values. Re-ingesting the same `source_id` updates the document and
replaces its chunks.

```bash
uv run litbot ingest corpus/public_domain/pride-prejudice-1813-ch1.txt
```

