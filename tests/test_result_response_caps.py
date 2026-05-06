"""``ResultResponse.rows_affected`` is decode-capped at ``INT_MAX``;
``last_insert_id_signed`` returns the int64-cast for stdlib parity.

The upstream C server returns ``sqlite3_changes(...)`` which is C
``int`` (``INT_MAX``) before being cast to uint64 on the wire
(``gateway.c:484``). A real cluster never emits above INT_MAX.

The signed accessor mirrors stdlib ``sqlite3.Connection.lastrowid``:
SQLite's ``sqlite3_int64`` rowid can be negative; the unsigned wire
value is ``2**64 - abs(rowid)``.

Encode-side cap is symmetric with decode (mirrors the
``StmtResponse._MAX_TAIL_OFFSET`` symmetric-cap precedent at the
sibling response): a Python encoder constructing a ``ResultResponse``
with ``rows_affected > _MAX_ROWS_AFFECTED`` is rejected up front, so
a same-process round-trip via Python encoder + decoder cannot break
on the encoder's own bytes.
"""

import pytest

from dqlitewire.exceptions import DecodeError, EncodeError
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


def test_rows_affected_at_int_max_construction_accepted() -> None:
    """Construction at the boundary is accepted (mirrors the decode
    side, which accepts at the boundary too)."""
    resp = ResultResponse(last_insert_id=0, rows_affected=_INT_MAX)
    assert resp.rows_affected == _INT_MAX


def test_rows_affected_above_int_max_construction_rejected() -> None:
    """Construction-time cap rejection — mirrors the decode-time cap
    so that a same-process Python encoder + decoder round-trip cannot
    break on its own bytes. Without this guard, ``ResultResponse(0,
    rows_affected=2**32).encode()`` succeeds and the same Python
    decoder then raises ``DecodeError`` on the produced bytes — a
    confusing self-inflicted asymmetry for mock-server / proxy authors.
    """
    with pytest.raises(EncodeError, match="rows_affected"):
        ResultResponse(last_insert_id=0, rows_affected=_INT_MAX + 1)


def test_rows_affected_two_to_the_32_construction_rejected() -> None:
    """The exact reproducer in the issue file: a `2**32` value is the
    canonical mock-server / proxy author trap."""
    with pytest.raises(EncodeError, match="rows_affected"):
        ResultResponse(last_insert_id=0, rows_affected=1 << 32)


def test_rows_affected_round_trip_at_boundary_succeeds() -> None:
    """Same-process round-trip at the boundary: construction +
    encode + decode all succeed."""
    resp = ResultResponse(last_insert_id=42, rows_affected=_INT_MAX)
    encoded = resp.encode()
    from dqlitewire.constants import HEADER_SIZE

    decoded = ResultResponse.decode_body(encoded[HEADER_SIZE:])
    assert decoded.rows_affected == _INT_MAX
    assert decoded.last_insert_id == 42


def test_last_insert_id_not_capped_at_int_max_at_construction() -> None:
    """Negative-pin: ``last_insert_id`` must NOT be capped at INT_MAX.
    The wire field is uint64 (per the protocol spec); SQLite's
    ``sqlite3_int64`` rowid can be the most-negative int64, which
    appears on the wire as ``2**63``. Capping ``last_insert_id``
    would break the existing ``last_insert_id_signed`` accessor
    contract. A future refactor copying the rows_affected cap to
    both fields would silently pass without this pin.
    """
    on_wire = (1 << 63) | 0xFEDCBA  # safely above INT_MAX
    resp = ResultResponse(last_insert_id=on_wire, rows_affected=0)
    assert resp.last_insert_id == on_wire
