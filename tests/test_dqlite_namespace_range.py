"""``is_dqlite_namespace_code`` uses the numeric range ``[1000, 1024)`` rather
than a hard-coded set, so a future dqlite code is auto-included instead of
masking through ``primary_sqlite_code(code & 0xFF)`` to a phantom primary.
Extended SQLite codes (e.g. ``SQLITE_IOERR_NOT_LEADER = 10250``) sit outside
the range and keep masking correctly."""

from __future__ import annotations

import pytest

from dqlitewire.constants import (
    DQLITE_NOTFOUND,
    DQLITE_PARSE,
    DQLITE_PROTO,
    is_dqlite_namespace_code,
    primary_sqlite_code,
)


@pytest.mark.parametrize("code", [DQLITE_PROTO, DQLITE_NOTFOUND, DQLITE_PARSE])
def test_known_namespace_codes_passthrough(code: int) -> None:
    assert is_dqlite_namespace_code(code) is True
    assert primary_sqlite_code(code) == code


def test_hypothetical_future_namespace_code_passthrough() -> None:
    """A code added upstream later (e.g. 1010) must pass through unchanged
    instead of masking to 0xF2 = 242."""
    future_code = 1010
    assert is_dqlite_namespace_code(future_code) is True
    assert primary_sqlite_code(future_code) == future_code


def test_entire_namespace_range_treated_as_namespace() -> None:
    for code in range(1000, 1024):
        assert is_dqlite_namespace_code(code) is True
        assert primary_sqlite_code(code) == code


def test_below_namespace_range_is_not_namespace() -> None:
    """999 is just below the range and must be excluded from namespace
    handling."""
    assert is_dqlite_namespace_code(999) is False


def test_at_or_above_namespace_range_is_not_namespace() -> None:
    """1024 and above are not namespace; ``SQLITE_IOERR_NOT_LEADER = 10250`` is
    a real extended code that must still mask to SQLITE_IOERR (10)."""
    assert is_dqlite_namespace_code(1024) is False
    assert is_dqlite_namespace_code(10250) is False
    assert primary_sqlite_code(10250) == 10
