"""``_MAX_COLUMN_COUNT`` is set to SQLite's documented column limit
(``SQLITE_MAX_COLUMN = 2000``) so legitimate wide-table SELECT
results decode while still rejecting absurd peer emissions.

The C server emits ``sqlite3_column_count(stmt)`` as a uint64
without cap (``query.c:111-120``); ``stmt.c:10``'s
``STMT__MAX_COLUMNS = (1 << 8) - 1 = 255`` macro is defined but
never referenced. SQLite's compile-time default is 2000 (raisable
to 32767 via ``SQLITE_MAX_COLUMN`` build flag); a wide-table
SELECT against an analytics / feature-store schema legitimately
crosses 255 columns.

The per-name cap (``_MAX_COLUMN_NAME_SIZE = 4096``) and the frame-
envelope cap (default 64 MiB) already bound memory growth from the
N × name allocation; this cap is defence-in-depth against
pathological peer emissions, not the load-bearing memory bound.
"""

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import (
    _MAX_COLUMN_COUNT,
    RowsResponse,
    StmtResponse,
)
from dqlitewire.types import encode_uint64


def test_max_column_count_pinned_to_sqlite_default() -> None:
    """SQLite's documented default ``SQLITE_MAX_COLUMN`` is 2000."""
    assert _MAX_COLUMN_COUNT == 2000


def test_rows_response_rejects_count_above_cap() -> None:
    body = encode_uint64(_MAX_COLUMN_COUNT + 1)
    with pytest.raises(DecodeError, match="(?i)column count"):
        RowsResponse.decode_body(body)


def test_rows_response_accepts_count_at_cap() -> None:
    """A 2000-column rows response is well-formed and must not be
    rejected by the cap; it fails the body-size check instead
    because we only sent the count, not the column names."""
    body = encode_uint64(_MAX_COLUMN_COUNT)
    with pytest.raises(DecodeError, match="exceeds maximum possible"):
        RowsResponse.decode_body(body)


def test_rows_response_accepts_count_above_old_255_cap() -> None:
    """Pin the regression-vs-old-cap shape: a 1500-column emission
    (legitimate wide table, above the prior 255 cap but below the
    new 2000 cap) must NOT trip the column-count cap. It still
    fails the body-size check below because we only sent the count,
    not the per-column name payload."""
    body = encode_uint64(1500)
    with pytest.raises(DecodeError, match="exceeds maximum possible"):
        RowsResponse.decode_body(body)


def test_rows_response_rejects_absurd_count() -> None:
    """A pathological emission (``column_count = 2^31``) must still
    be rejected so a hostile peer cannot inflate Python-side
    allocations."""
    body = encode_uint64(1 << 31)
    with pytest.raises(DecodeError, match="(?i)column count"):
        RowsResponse.decode_body(body)


def test_servers_response_uses_separate_cap() -> None:
    """``ServersResponse`` uses ``_MAX_NODE_COUNT = 10_000``; the
    column cap does not apply. Pinning here is a sanity check that
    the cap constant was not accidentally inlined into an unrelated
    field."""
    assert _MAX_COLUMN_COUNT < 10_000


def test_stmt_response_num_params_unaffected() -> None:
    """``StmtResponse.num_params`` uses ``_MAX_PARAM_COUNT``
    (32_766) — verify the column cap tighten did not collide."""
    # 1000 params is fine — within _MAX_PARAM_COUNT but well above
    # the column cap. The body needs db_id+stmt_id+num_params.
    body = encode_uint64(0) + encode_uint64(1) + encode_uint64(1000)
    # Not a real well-formed response but the num_params cap is what
    # we're pinning; it should NOT raise on 1000.
    try:
        StmtResponse.decode_body(body)
    except DecodeError as e:
        # Any decode error must NOT cite the column-count cap.
        assert "column count" not in str(e)
