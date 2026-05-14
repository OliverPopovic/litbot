from pathlib import Path

import structlog
from psycopg import Connection

from litbot.config import Settings, get_settings
from litbot.ingestion.chunking import chunk_document
from litbot.ingestion.parsers import parse_document
from litbot.langchain import chunk_to_document, delete_source_documents, make_vector_store
from litbot.models import ParsedDocument, TextChunk

logger = structlog.get_logger(__name__)


class IngestionService:
    """Idempotent parser/chunker/embedder for approved literary documents."""

    def __init__(self, conn: Connection, settings: Settings | None = None) -> None:
        self.conn = conn
        self.settings = settings or get_settings()
        self.vector_store = make_vector_store(self.settings)

    def ingest_path(self, path: Path, metadata_path: Path | None = None) -> list[TextChunk]:
        parsed = parse_document(path, metadata_path)
        return self.ingest_document(parsed)

    def ingest_document(self, parsed: ParsedDocument) -> list[TextChunk]:
        chunks = chunk_document(parsed.text, parsed.metadata)
        if not chunks:
            raise ValueError(f"No chunks produced for {parsed.metadata.source_id}")
        if not parsed.metadata.license.strip():
            raise ValueError(f"Missing license for {parsed.metadata.source_id}")

        delete_source_documents(self.conn, parsed.metadata.source_id, self.settings)
        self.conn.commit()
        documents = [chunk_to_document(chunk) for chunk in chunks]
        self.vector_store.add_documents(documents, ids=[chunk.chunk_id for chunk in chunks])
        logger.info(
            "document_ingested",
            source_id=parsed.metadata.source_id,
            chunk_count=len(chunks),
        )
        return chunks
