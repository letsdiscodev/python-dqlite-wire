"""The top-level package logger has a logging.NullHandler attached at import,
per the Python logging HOWTO convention for libraries."""

from __future__ import annotations

import logging

import dqlitewire  # noqa: F401 -- import for side effect


def test_top_level_logger_has_null_handler() -> None:
    logger = logging.getLogger("dqlitewire")
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers), (
        "library top-level logger must have a NullHandler attached per "
        "Python logging HOWTO convention (every well-behaved library — "
        "psycopg, aiosqlite, asyncpg, urllib3 — does this)"
    )
