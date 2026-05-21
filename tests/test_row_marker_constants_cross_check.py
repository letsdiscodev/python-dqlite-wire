"""Pin: the four row-marker constants form a consistent pair.

The wire format identifies the end of a ``RowsResponse`` body with one
of two 8-byte sentinels: ``0xFF * 8`` (DONE) or ``0xEE * 8`` (PART).
The package spells the sentinel in two representations:

- ``ROW_DONE_BYTE`` / ``ROW_PART_BYTE`` — single-byte ints (source of
  truth).
- ``ROW_DONE_MARKER`` / ``ROW_PART_MARKER`` — uint64 ints used by the
  encode path.
- ``_ROW_DONE_MARKER`` / ``_ROW_PART_MARKER`` — 8-byte ``bytes``
  constants (in ``tuples.py``) used by the decode path.

These four must agree byte-for-byte. Without a cross-check pin, a typo
in any single-byte constant propagates only to the bytes form (which
is derived from it) and not to the uint64 int form (or vice versa),
silently breaking the encode/decode round-trip.
"""

from __future__ import annotations

from dqlitewire.constants import (
    ROW_DONE_BYTE,
    ROW_DONE_MARKER,
    ROW_PART_BYTE,
    ROW_PART_MARKER,
)
from dqlitewire.tuples import _ROW_DONE_MARKER, _ROW_PART_MARKER


def test_row_done_marker_uint64_matches_bytes_form() -> None:
    assert ROW_DONE_MARKER.to_bytes(8, "little") == _ROW_DONE_MARKER


def test_row_part_marker_uint64_matches_bytes_form() -> None:
    assert ROW_PART_MARKER.to_bytes(8, "little") == _ROW_PART_MARKER


def test_row_done_marker_bytes_derived_from_byte_constant() -> None:
    assert bytes([ROW_DONE_BYTE]) * 8 == _ROW_DONE_MARKER


def test_row_part_marker_bytes_derived_from_byte_constant() -> None:
    assert bytes([ROW_PART_BYTE]) * 8 == _ROW_PART_MARKER


def test_row_done_marker_canonical_hex_value() -> None:
    """Pin the wire byte sequence to the C upstream macro value."""
    assert ROW_DONE_MARKER == 0xFFFFFFFFFFFFFFFF
    assert _ROW_DONE_MARKER == b"\xff" * 8


def test_row_part_marker_canonical_hex_value() -> None:
    assert ROW_PART_MARKER == 0xEEEEEEEEEEEEEEEE
    assert _ROW_PART_MARKER == b"\xee" * 8
