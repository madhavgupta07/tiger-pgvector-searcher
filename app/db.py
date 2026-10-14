"""Database access: connection pool, extension setup, and pgvector adaptation."""

from __future__ import annotations

import pathlib
import threading

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from . import config

_SCHEMA_PATH = pathlib.Path(__file__).with_name("schema.sql")

_pool: ConnectionPool | None = None
_lock = threading.Lock()


def configure(conn: psycopg.Connection) -> None:
    """Register the pgvector adapter on every pooled connection."""
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                if not config.DATABASE_URL:
                    raise RuntimeError(
                        "DATABASE_URL is not set. Copy .env.example to .env and "
                        "paste your Tiger Cloud connection string "
                        "(tiger db uri <service-id> --with-password)."
                    )
                _pool = ConnectionPool(
                    config.DATABASE_URL,
                    min_size=1,
                    max_size=6,
                    configure=configure,
                    open=True,
                    kwargs={"autocommit": True, "application_name": "tiger-vector-search"},
                )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def schema_sql() -> str:
    return _SCHEMA_PATH.read_text(encoding="utf-8").format(dim=config.EMBEDDING_DIM)


def apply_schema() -> list[str]:
    """Create extensions, hypertable, and indexes. Idempotent."""
    sql = schema_sql()
    statements: list[str] = []
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            if cur.description:
                statements = [row[0] for row in cur.fetchall()]
    return statements
