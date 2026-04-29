"""Pin: ``is_dqlite_namespace_code`` uses a numeric range
``[1000, 1024)`` rather than a hard-coded set, so future
dqlite-namespace additions are auto-included instead of silently
masking through ``primary_sqlite_code(code & 0xFF)``.

The set ``{DQLITE_PROTO, DQLITE_NOTFOUND, DQLITE_PARSE}`` was
exhaustive at the time of writing; if upstream ever adds a fourth
namespace code the masked low byte would yield a phantom SQLite
primary that the dispatch tables would route to the wrong dbapi
exception class. The range-based check eliminates the silent-mask
hazard while preserving the existing pass-through behaviour.

Extended SQLite codes (``SQLITE_IOERR_NOT_LEADER = 10250``,
``SQLITE_IOERR_LEADERSHIP_LOST = 10762``) live OUTSIDE
``[1000, 1024)`` so they continue to mask correctly via
``code & 0xFF``.
"""

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
    """A code added upstream after this release (e.g., 1010) must
    pass through unchanged instead of masking to 0xF2 = 242."""
    future_code = 1010
    assert is_dqlite_namespace_code(future_code) is True
    assert primary_sqlite_code(future_code) == future_code


def test_entire_namespace_range_treated_as_namespace() -> None:
    for code in range(1000, 1024):
        assert is_dqlite_namespace_code(code) is True
        assert primary_sqlite_code(code) == code


def test_below_namespace_range_is_not_namespace() -> None:
    """999 is just below the namespace range. It's a hypothetical
    extended SQLite code (low byte 0xE7, high byte 0x03 → primary
    231) — a meaningless primary, but the range check correctly
    excludes it from namespace handling."""
    assert is_dqlite_namespace_code(999) is False


def test_at_or_above_namespace_range_is_not_namespace() -> None:
    """1024 and above are NOT namespace. 1024 = 0x400 is just out of
    range; ``SQLITE_IOERR_NOT_LEADER = 10250`` is a real extended
    code and must still mask correctly."""
    assert is_dqlite_namespace_code(1024) is False
    # SQLITE_IOERR_NOT_LEADER masks to SQLITE_IOERR (10).
    assert is_dqlite_namespace_code(10250) is False
    assert primary_sqlite_code(10250) == 10
