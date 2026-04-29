"""Pin: ``primary_sqlite_code`` passes dqlite-namespace codes (>= 1000)
through unchanged.

The function masks the SQLite extended-code byte to recover the
primary SQLite code (``code & 0xFF``) for codes that fit in the
SQLite namespace, but **passes dqlite-namespace codes through
unchanged**. Without this pass-through, ``DQLITE_PARSE`` (1005)
would mask to ``0xED == 237``, which is meaningless.

The earlier ``test_constants.py`` pins covered the SQLite-side
identity case for low-byte primaries (0/1/5/10) and the masking
case for extended codes. The dqlite-namespace pass-through branch
itself was never directly pinned, so a future refactor could
silently regress.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import (
    DQLITE_NOTFOUND,
    DQLITE_PARSE,
    DQLITE_PROTO,
    primary_sqlite_code,
)


@pytest.mark.parametrize(
    "code",
    [
        DQLITE_PROTO,  # 1001
        DQLITE_NOTFOUND,  # 1002
        DQLITE_PARSE,  # 1005
    ],
)
def test_primary_sqlite_code_dqlite_namespace_passthrough(code: int) -> None:
    """Codes >= 1000 are dqlite-only; masking with 0xFF would yield
    a meaningless low-byte (233/234/237). Pass through unchanged."""
    assert primary_sqlite_code(code) == code


def test_primary_sqlite_code_extended_sqlite_code_still_masked() -> None:
    """Sanity: extended SQLite codes still mask to their primary.
    SQLITE_CONSTRAINT_UNIQUE = 2067 = 0x813 → 0x13 = 19 (SQLITE_CONSTRAINT)."""
    assert primary_sqlite_code(2067) == 19
