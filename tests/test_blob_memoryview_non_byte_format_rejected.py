"""Pin: ``memoryview`` over a non-byte format (e.g. ``memoryview(array.
array('i', ...))``) must be rejected with the same rationale that
``array.array`` itself is rejected.

The inference discriminator at ``types.py`` accepts ``memoryview``
because the common idiom is zero-copy binding of byte buffers
(``memoryview(b"...")``). But ``memoryview(array.array('i', [...]))``
is also a ``memoryview`` instance and would otherwise sail through —
emitting bytes in HOST endianness and HOST int width, which is
exactly the cross-platform-corruption hazard the bare ``array.array``
rejection (ISSUE-765) is meant to prevent.

Both the inference branch (no explicit ``value_type``) and the
explicit-BLOB branch (caller passes ``ValueType.BLOB``) must reject
non-byte-format memoryviews. The byte-format twin
(``memoryview(b"...")``) must continue to work.
"""

from __future__ import annotations

import array

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import decode_value, encode_value


def test_encode_value_infer_rejects_non_byte_format_memoryview() -> None:
    """``memoryview(array.array('i', ...))`` carries host-endian
    int32 bytes; inference must reject it identically to bare
    ``array.array`` (which is rejected per ISSUE-765)."""
    mv = memoryview(array.array("i", [1, 2, 3]))
    assert mv.format == "i" and mv.itemsize == 4
    with pytest.raises(EncodeError, match="format"):
        encode_value(mv)


def test_encode_value_blob_explicit_rejects_non_byte_format_memoryview() -> None:
    """Symmetric pin on the explicit-BLOB branch: the caller-supplied
    ``ValueType.BLOB`` path must also reject non-byte-format
    memoryviews so the inference/explicit branches stay in lockstep
    (mirrors the ``mmap.mmap`` parity established in
    ``test_blob_inference_mmap.py``)."""
    mv = memoryview(array.array("d", [1.5, 2.5, 3.5]))
    assert mv.format == "d" and mv.itemsize == 8
    with pytest.raises(EncodeError, match="format"):
        encode_value(mv, ValueType.BLOB)


def test_encode_value_infer_rejects_non_byte_format_memoryview_int64() -> None:
    """Coverage for the ``'q'`` (signed int64) format — host-endian
    8-byte items, distinct hazard from ``'i'``."""
    mv = memoryview(array.array("q", [1, 2, 3]))
    assert mv.format in ("q", "l") and mv.itemsize == 8
    with pytest.raises(EncodeError, match="format"):
        encode_value(mv)


def test_encode_value_accepts_byte_format_memoryview_infer() -> None:
    """Negative twin: ``memoryview`` over the byte format ('B') is the
    documented zero-copy BLOB binding shape and must continue to work
    via inference."""
    mv = memoryview(b"\x01\x02\x03")
    assert mv.format == "B"
    encoded, vtype = encode_value(mv)
    assert vtype == ValueType.BLOB
    decoded, _ = decode_value(encoded, ValueType.BLOB)
    assert decoded == b"\x01\x02\x03"


def test_encode_value_accepts_byte_format_memoryview_explicit_blob() -> None:
    """Negative twin on the explicit-BLOB branch."""
    mv = memoryview(bytearray(b"explicit"))
    assert mv.format == "B"
    encoded, vtype = encode_value(mv, ValueType.BLOB)
    assert vtype == ValueType.BLOB
    decoded, _ = decode_value(encoded, ValueType.BLOB)
    assert decoded == b"explicit"


def test_encode_value_accepts_signed_byte_format_memoryview() -> None:
    """``'b'`` (signed char) is also a single-byte format and must
    continue to work; the rejection list targets multi-byte typed
    formats only."""
    mv = memoryview(array.array("b", [1, -2, 3]))
    assert mv.format == "b" and mv.itemsize == 1
    encoded, vtype = encode_value(mv)
    assert vtype == ValueType.BLOB
    decoded, _ = decode_value(encoded, ValueType.BLOB)
    assert decoded == bytes(mv)
