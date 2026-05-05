"""``ResultResponse.rows_affected`` is decode-capped at ``INT_MAX``;
``last_insert_id_signed`` returns the int64-cast for stdlib parity.

The upstream C server returns ``sqlite3_changes(...)`` which is C
``int`` (``INT_MAX``) before being cast to uint64 on the wire
(``gateway.c:484``). A real cluster never emits above INT_MAX.

The signed accessor mirrors stdlib ``sqlite3.Connection.lastrowid``:
SQLite's ``sqlite3_int64`` rowid can be negative; the unsigned wire
value is ``2**64 - abs(rowid)``.
"""

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import ResultResponse
from dqlitewire.types import encode_uint64

_INT_MAX = (1 << 31) - 1


def test_rows_affected_at_int_max_accepted() -> None:
    body = encode_uint64(0) + encode_uint64(_INT_MAX)
    resp = ResultResponse.decode_body(body)
    assert resp.rows_affected == _INT_MAX


def test_rows_affected_above_int_max_rejected() -> None:
    body = encode_uint64(0) + encode_uint64(_INT_MAX + 1)
    with pytest.raises(DecodeError, match="rows_affected"):
        ResultResponse.decode_body(body)


def test_rows_affected_two_to_the_63_rejected() -> None:
    """Hostile-server scenario: explicit huge value."""
    body = encode_uint64(0) + encode_uint64(1 << 63)
    with pytest.raises(DecodeError, match="rows_affected"):
        ResultResponse.decode_body(body)


def test_last_insert_id_signed_round_trip_negative() -> None:
    """A SQLite rowid of -1 round-trips through uint64 as 2**64 - 1.
    The signed accessor reverses the cast."""
    on_wire = (1 << 64) - 1  # -1 as uint64
    resp = ResultResponse(last_insert_id=on_wire, rows_affected=0)
    assert resp.last_insert_id == on_wire
    assert resp.last_insert_id_signed == -1


def test_last_insert_id_signed_round_trip_positive() -> None:
    resp = ResultResponse(last_insert_id=42, rows_affected=0)
    assert resp.last_insert_id == 42
    assert resp.last_insert_id_signed == 42


def test_last_insert_id_signed_at_int64_max() -> None:
    """The signed boundary: 2**63 - 1 stays positive both
    unsigned and signed."""
    boundary = (1 << 63) - 1
    resp = ResultResponse(last_insert_id=boundary, rows_affected=0)
    assert resp.last_insert_id_signed == boundary


def test_last_insert_id_signed_at_int64_min() -> None:
    """``2**63`` on the wire is the most-negative int64 (=-2**63)."""
    on_wire = 1 << 63
    resp = ResultResponse(last_insert_id=on_wire, rows_affected=0)
    assert resp.last_insert_id_signed == -(1 << 63)
