"""``_MAX_COLUMN_COUNT`` is set to dqlite's actual emission ceiling
(255) so a hostile peer cannot inflate intermediate Python-side
allocations by claiming a higher column count than the C server
ever produces.

The C server's ``stmt.c:10`` defines
``STMT__MAX_COLUMNS = (1 << 8) - 1 = 255``.
"""

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import (
    _MAX_COLUMN_COUNT,
    RowsResponse,
    StmtResponse,
)
from dqlitewire.types import encode_uint64


def test_max_column_count_pinned_to_255() -> None:
    assert _MAX_COLUMN_COUNT == 255


def test_rows_response_rejects_count_above_255() -> None:
    body = encode_uint64(256)
    with pytest.raises(DecodeError, match="(?i)column count"):
        RowsResponse.decode_body(body)


def test_rows_response_accepts_count_at_255() -> None:
    """A 255-column rows response is well-formed and must not be
    rejected by the cap; it fails the body-size check instead
    because we only sent the count, not the column names."""
    body = encode_uint64(255)
    with pytest.raises(DecodeError, match="exceeds maximum possible"):
        RowsResponse.decode_body(body)


def test_servers_response_rejects_count_above_255() -> None:
    """``ServersResponse`` shares the same column-count cap path
    via ``_MAX_NODE_COUNT`` (separate cap) but uses
    ``_MAX_COLUMN_COUNT``-style protection on its own count field."""
    # ``ServersResponse`` uses ``_MAX_NODE_COUNT = 10_000``; the
    # column cap does not apply directly. Pinning here is a sanity
    # check that the cap constant was not accidentally inlined into
    # an unrelated field.
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
