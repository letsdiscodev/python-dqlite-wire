"""A non-byte-format ``memoryview`` (e.g. over ``array.array('i', ...)``)
is rejected: its bytes are host-endian/host-width, the same corruption
hazard that bars bare ``array.array``. Byte-format views still work."""

from __future__ import annotations

import array

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import decode_value, encode_value


def test_encode_value_infer_rejects_non_byte_format_memoryview() -> None:
    mv = memoryview(array.array("i", [1, 2, 3]))
    assert mv.format == "i" and mv.itemsize == 4
    with pytest.raises(EncodeError, match="format"):
        encode_value(mv)


def test_encode_value_blob_explicit_rejects_non_byte_format_memoryview() -> None:
    mv = memoryview(array.array("d", [1.5, 2.5, 3.5]))
    assert mv.format == "d" and mv.itemsize == 8
    with pytest.raises(EncodeError, match="format"):
        encode_value(mv, ValueType.BLOB)


def test_encode_value_infer_rejects_non_byte_format_memoryview_int64() -> None:
    mv = memoryview(array.array("q", [1, 2, 3]))
    assert mv.format in ("q", "l") and mv.itemsize == 8
    with pytest.raises(EncodeError, match="format"):
        encode_value(mv)


def test_encode_value_accepts_byte_format_memoryview_infer() -> None:
    mv = memoryview(b"\x01\x02\x03")
    assert mv.format == "B"
    encoded, vtype = encode_value(mv)
    assert vtype == ValueType.BLOB
    decoded, _ = decode_value(encoded, ValueType.BLOB)
    assert decoded == b"\x01\x02\x03"


def test_encode_value_accepts_byte_format_memoryview_explicit_blob() -> None:
    mv = memoryview(bytearray(b"explicit"))
    assert mv.format == "B"
    encoded, vtype = encode_value(mv, ValueType.BLOB)
    assert vtype == ValueType.BLOB
    decoded, _ = decode_value(encoded, ValueType.BLOB)
    assert decoded == b"explicit"


def test_encode_value_accepts_signed_byte_format_memoryview() -> None:
    """``'b'`` (signed char) is single-byte and must work; rejection targets multi-byte formats."""
    mv = memoryview(array.array("b", [1, -2, 3]))
    assert mv.format == "b" and mv.itemsize == 1
    encoded, vtype = encode_value(mv)
    assert vtype == ValueType.BLOB
    decoded, _ = decode_value(encoded, ValueType.BLOB)
    assert decoded == bytes(mv)
