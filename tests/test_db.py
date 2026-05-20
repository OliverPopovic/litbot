from pathlib import Path

from litbot import db
from litbot.config import Settings


class FakePool:
    instances: list["FakePool"] = []

    def __init__(self, conninfo: str, **_kwargs) -> None:
        self.conninfo = conninfo
        self.closed = False
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True


def test_get_pool_recreates_pool_when_database_url_changes(monkeypatch) -> None:
    db.close_pool()
    FakePool.instances.clear()
    monkeypatch.setattr(db, "ConnectionPool", FakePool)

    first = db.get_pool(Settings(database_url="postgresql://example/one"))
    second = db.get_pool(Settings(database_url="postgresql://example/two"))

    assert first is not second
    assert first.closed
    assert second.conninfo == "postgresql://example/two"

    db.close_pool()


def test_trigram_index_migration_exists() -> None:
    migration = (Path("migrations") / "002_trigram_index.sql").read_text(encoding="utf-8")

    assert "chunks_text_trgm_idx" in migration
    assert "gin_trgm_ops" in migration


def test_global_notes_migration_defines_storage_and_indexes() -> None:
    migration = (Path("migrations") / "003_global_notes.sql").read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS notes" in migration
    assert "CREATE TABLE IF NOT EXISTS note_chunks" in migration
    assert "embedding      vector(1536)" in migration
    assert "notes_embedding_hnsw_idx" in migration
    assert "notes_model_idx" in migration
    assert "notes_prompt_version_idx" in migration
    assert "notes_inferred_work_idx" in migration
    assert "note_chunks_chunk_id_idx" in migration
