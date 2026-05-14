from collections.abc import Iterator
from contextlib import contextmanager

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from litbot.config import Settings, get_settings

_pool: ConnectionPool | None = None


def get_pool(settings: Settings | None = None) -> ConnectionPool:
    """Return the process-wide PostgreSQL connection pool."""

    global _pool
    settings = settings or get_settings()
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            kwargs={"row_factory": dict_row},
            min_size=1,
            max_size=10,
            open=True,
        )
    return _pool


def close_pool() -> None:
    """Close the process-wide PostgreSQL connection pool, if it was opened."""

    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_connection(settings: Settings | None = None) -> Iterator[Connection]:
    pool = get_pool(settings)
    with pool.connection() as conn:
        yield conn
