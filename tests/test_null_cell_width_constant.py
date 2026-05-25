"""Pin: NULL wire cell width is centralised in ``NULL_CELL_WIDTH``
rather than scattered as a hard-coded ``8`` across the encode arm, the
decode arm, and the short-read diagnostic.

Anchored to four ``/* TODO: allow null to be encoded with 0 bytes */``
sites in ``dqlite-upstream/src/tuple.c`` (lines 71-73, 146, 262, 298).
If upstream lands the 0-byte NULL change, this constant flips and
both encode_value and decode_value NULL branches pick it up in
lockstep. The single-point-of-change discipline lets a future
maintainer answering "how wide is NULL on the wire?" find one site
instead of three.
"""

from __future__ import annotations

from dqlitewire.constants import ValueType
from dqlitewire.types import NULL_CELL_WIDTH, decode_value, encode_value


def test_null_cell_width_is_eight() -> None:
    """Pinned to the current upstream tuple.c contract: NULL is 8
    bytes wide on the wire. Flips iff upstream lands the long-standing
    TODO at tuple.c:71-73, 146, 262, 298."""
    assert NULL_CELL_WIDTH == 8


def test_encode_value_null_emits_exactly_null_cell_width_zero_bytes() -> None:
    """The encode arm emits ``NULL_CELL_WIDTH`` zero bytes, not a
    hard-coded 8."""
    encoded, vtype = encode_value(None)
    assert vtype == ValueType.NULL
    assert encoded == b"\x00" * NULL_CELL_WIDTH
    assert len(encoded) == NULL_CELL_WIDTH


def test_decode_value_null_consumes_exactly_null_cell_width_bytes() -> None:
    """The decode arm consumes ``NULL_CELL_WIDTH`` bytes."""
    value, consumed = decode_value(b"\x00" * NULL_CELL_WIDTH, ValueType.NULL)
    assert value is None
    assert consumed == NULL_CELL_WIDTH
