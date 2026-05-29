"""Pin: ``primary_sqlite_code`` passes dqlite-namespace codes (>= 1000)
through unchanged (masking with 0xFF would yield a meaningless byte)."""

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
    assert primary_sqlite_code(code) == code


def test_primary_sqlite_code_extended_sqlite_code_still_masked() -> None:
    """Extended SQLite codes still mask to their primary (2067 -> 19)."""
    assert primary_sqlite_code(2067) == 19
