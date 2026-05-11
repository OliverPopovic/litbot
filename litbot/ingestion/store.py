import hashlib
import json
from pathlib import Path

import structlog
from psycopg import Connection

from litbot.db import vector_literal
from litbot.ingestion.chunking import chunk_document
from litbot.ingestion.parsers import parse_document
from litbot.models import ParsedDocument, TextChunk
from litbot.openai_client import OpenAIModelClient

logger = structlog.get_logger(__name__)


class IngestionService:
    """Idempotent parser/chunker/embedder for approved literary documents."""

    def __init__(self, conn: Connection, model_client: OpenAIModelClient) -> None:
        self.conn = conn
        self.model_client = model_client

    def ingest_path(self, path: Path, metadata_path: Path | None = None) -> list[TextChunk]:
        parsed = parse_document(path, metadata_path)
        return self.ingest_document(parsed)

    def ingest_document(self, parsed: ParsedDocument) -> list[TextChunk]:
        chunks = chunk_document(parsed.text, parsed.metadata)
        if not chunks:
            raise ValueError(f"No chunks produced for {parsed.metadata.source_id}")
        if not parsed.metadata.license.strip():
            raise ValueError(f"Missing license for {parsed.metadata.source_id}")

        embeddings = self.model_client.embed_texts([chunk.text for chunk in chunks])
        document_id = self._upsert_document(parsed)
        self.conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._insert_chunk(document_id, chunk, embedding)
        self.conn.commit()
        logger.info(
            "document_ingested",
            source_id=parsed.metadata.source_id,
            chunk_count=len(chunks),
        )
        return chunks

    def _upsert_document(self, parsed: ParsedDocument) -> int:
        metadata = parsed.metadata
        content_hash = hashlib.sha256(parsed.text.encode("utf-8")).hexdigest()
        row = self.conn.execute(
            """
            INSERT INTO documents (
                source_id, title, author, translator, editor, publication_year, edition,
                genre, language, license, uri, version, metadata, content_hash
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (source_id) DO UPDATE SET
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                translator = EXCLUDED.translator,
                editor = EXCLUDED.editor,
                publication_year = EXCLUDED.publication_year,
                edition = EXCLUDED.edition,
                genre = EXCLUDED.genre,
                language = EXCLUDED.language,
                license = EXCLUDED.license,
                uri = EXCLUDED.uri,
                version = EXCLUDED.version,
                metadata = EXCLUDED.metadata,
                content_hash = EXCLUDED.content_hash,
                ingested_at = now()
            RETURNING id
            """,
            (
                metadata.source_id,
                metadata.title,
                metadata.author,
                metadata.translator,
                metadata.editor,
                metadata.publication_year,
                metadata.edition,
                metadata.genre,
                metadata.language,
                metadata.license,
                metadata.uri,
                metadata.version,
                json.dumps(metadata.metadata),
                content_hash,
            ),
        ).fetchone()
        return int(row["id"])

    def _insert_chunk(self, document_id: int, chunk: TextChunk, embedding: list[float]) -> None:
        self.conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, document_id, source_id, chunk_index, text, token_count,
                chunk_hash, embedding, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)
            """,
            (
                chunk.chunk_id,
                document_id,
                chunk.source_id,
                chunk.chunk_index,
                chunk.text,
                chunk.token_count,
                chunk.chunk_hash,
                vector_literal(embedding),
                json.dumps(chunk.metadata),
            ),
        )
