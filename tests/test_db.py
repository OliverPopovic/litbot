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
