from pathlib import Path

import structlog
from psycopg import Connection
from psycopg.types.json import Jsonb

from litbot.config import Settings, get_settings
from litbot.ingestion.chunking import chunk_document
from litbot.ingestion.parsers import parse_document
from litbot.langchain import embed_texts
from litbot.models import DocumentMetadata, ParsedDocument, TextChunk

logger = structlog.get_logger(__name__)

_EMBEDDING_BATCH_SIZE = 500


class IngestionService:
    """Idempotent parser/chunker/embedder for approved literary documents."""

    def __init__(self, conn: Connection, settings: Settings | None = None) -> None:
        self.conn = conn
        self.settings = settings or get_settings()

    def ingest_path(self, path: Path, metadata_path: Path | None = None) -> list[TextChunk]:
        parsed = parse_document(path, metadata_path)
        return self.ingest_document(parsed)

    def ingest_document(self, parsed: ParsedDocument) -> list[TextChunk]:
        chunks = chunk_document(parsed.text, parsed.metadata)
        if not chunks:
            raise ValueError(f"No chunks produced for {parsed.metadata.source_id}")
        if not parsed.metadata.license.strip():
            raise ValueError(f"Missing license for {parsed.metadata.source_id}")

        # Remove any prior data for this source so reingestion is idempotent.
        self._delete_source(parsed.metadata.source_id)

        # Upsert the document row and get its primary key.
        doc_id = self._upsert_document(parsed.metadata)

        # Embed chunk texts in safe batches, then insert them directly into our schema.
        vectors = self._embed_chunks(chunks)
        self._insert_chunks(chunks, vectors, doc_id)

        self.conn.commit()
        logger.info(
            "document_ingested",
            source_id=parsed.metadata.source_id,
            chunk_count=len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _delete_source(self, source_id: str) -> None:
        self.conn.execute(
            "DELETE FROM documents WHERE source_id = %s",
            (source_id,),
        )

    def _upsert_document(self, metadata: DocumentMetadata) -> int:
        row = self.conn.execute(
            """
            INSERT INTO documents
                (source_id, title, author, translator, editor,
                 publication_year, edition, genre, language,
                 license, uri, version, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_id) DO UPDATE SET
                title            = EXCLUDED.title,
                author           = EXCLUDED.author,
                translator       = EXCLUDED.translator,
                editor           = EXCLUDED.editor,
                publication_year = EXCLUDED.publication_year,
                edition          = EXCLUDED.edition,
                genre            = EXCLUDED.genre,
                language         = EXCLUDED.language,
                license          = EXCLUDED.license,
                uri              = EXCLUDED.uri,
                version          = EXCLUDED.version,
                metadata         = EXCLUDED.metadata,
                ingested_at      = now()
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
                Jsonb(metadata.metadata),
            ),
        ).fetchone()
        return int(row["id"])

    def _embed_chunks(self, chunks: list[TextChunk]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), _EMBEDDING_BATCH_SIZE):
            batch = chunks[start : start + _EMBEDDING_BATCH_SIZE]
            vectors.extend(embed_texts([chunk.text for chunk in batch], self.settings))
        return vectors

    def _insert_chunks(
        self,
        chunks: list[TextChunk],
        vectors: list[list[float]],
        doc_id: int,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Embedding count does not match chunk count")

        rows = [
            (
                chunk.chunk_id,
                doc_id,
                chunk.source_id,
                chunk.chunk_index,
                chunk.text,
                chunk.token_count,
                chunk.chunk_hash,
                _vector_literal(vector),
                Jsonb(chunk.metadata),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.conn.executemany(
            """
            INSERT INTO chunks
                (chunk_id, document_id, source_id, chunk_index,
                 text, token_count, chunk_hash, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            rows,
        )


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"
