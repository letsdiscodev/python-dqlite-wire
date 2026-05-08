"""Pin: at column count n=16, no valid ``ValueType`` packing collides
with the row-marker sentinels (``DONE = 0xFF..FF``, ``PART = 0xEE..EE``).

For n=16 the row-header is exactly one 8-byte word — the same shape as
a marker. The decoder applies a full-uint64 marker check (strictly
tighter than Go's first-byte-only check) before the type-nibble
decode. The safety property: no ``ValueType`` is 14 or 15, so neither
``0xEE`` nor ``0xFF`` can arise from packing two valid type nibbles.
That property holds today by construction; this test fixture pins it
so a future change — adding a ``ValueType`` with code 14 or 15,
changing the marker pattern, or changing the row-header layout — can
not silently violate it.
"""

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.tuples import RowMarker, decode_row_header, encode_row_header

_VALID_TYPES = list(ValueType)


@pytest.mark.parametrize("type_uniform", _VALID_TYPES)
def test_16_column_uniform_type_does_not_collide_with_row_marker(
    type_uniform: ValueType,
) -> None:
    """A 16-column row header with all-same type nibbles must not pack
    to either marker sentinel. Round-trip through ``decode_row_header``
    must yield the original types, never a ``RowMarker``."""
    types = [type_uniform] * 16
    header_bytes = encode_row_header(types)
    assert len(header_bytes) == 8
    assert header_bytes != b"\xff" * 8, (
        f"ValueType {type_uniform!r} packs to the DONE marker — would be "
        "indistinguishable from end-of-rows on the wire."
    )
    assert header_bytes != b"\xee" * 8, (
        f"ValueType {type_uniform!r} packs to the PART marker — would be "
        "indistinguishable from end-of-batch on the wire."
    )

    decoded, consumed = decode_row_header(header_bytes, column_count=16)
    assert consumed == 8
    assert decoded == types


def test_16_column_no_valid_type_pair_packs_to_marker_byte() -> None:
    """Stronger property: no pair of valid ``ValueType`` codes packs
    to ``0xEE`` or ``0xFF``. Pinning the property exhaustively across
    every (low, high) pair, not just the uniform-type slice. A new
    ``ValueType`` with code 14 or 15 would break this immediately."""
    for low in _VALID_TYPES:
        for high in _VALID_TYPES:
            packed = (int(high) << 4) | int(low)
            assert packed != 0xFF, (
                f"({low!r}, {high!r}) packs to 0xFF — would collide with the "
                "DONE marker if repeated 8 times."
            )
            assert packed != 0xEE, (
                f"({low!r}, {high!r}) packs to 0xEE — would collide with the "
                "PART marker if repeated 8 times."
            )


def test_16_column_done_marker_bytes_decode_as_marker_not_null_row() -> None:
    """A raw 8-byte ``0xFF * 8`` payload must classify as
    ``RowMarker.DONE``, not as a 16-column row of repeated type
    nibbles. Pin the marker check runs before the type-nibble decode
    so the strict-validation contract is preserved at the n=16
    boundary where header_size == marker_size."""
    payload = b"\xff" * 8
    decoded, consumed = decode_row_header(payload, column_count=16)
    assert decoded is RowMarker.DONE
    assert consumed == 8


def test_16_column_part_marker_bytes_decode_as_marker_not_null_row() -> None:
    """Symmetric pin for the PART marker."""
    payload = b"\xee" * 8
    decoded, consumed = decode_row_header(payload, column_count=16)
    assert decoded is RowMarker.PART
    assert consumed == 8
