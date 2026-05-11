"""Pin: the top-level package logger has a ``logging.NullHandler``
attached, per the Python logging HOWTO convention for libraries.

The handler is added at package import time; this test runs the same
operation a user's downstream code would: ``getLogger("dqlitewire")``
and inspect handlers.
"""

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
