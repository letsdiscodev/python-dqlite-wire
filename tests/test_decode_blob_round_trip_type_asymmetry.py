"""``decode_blob`` always returns ``bytes`` regardless of bind input type
(bytes/bytearray/memoryview), matching stdlib ``sqlite3``. If the contract
is ever tightened to re-wrap as ``memoryview``, delete this pin and update
the ``decode_blob`` docstring."""

from __future__ import annotations

from dqlitewire.constants import ValueType
from dqlitewire.types import decode_blob, decode_value, encode_blob, encode_value


def test_decode_blob_returns_bytes_for_bytes_input() -> None:
    encoded = encode_blob(b"hello")
    decoded, _ = decode_blob(encoded)
    assert type(decoded) is bytes
    assert decoded == b"hello"


def test_decode_blob_returns_bytes_for_bytearray_input() -> None:
    """A ``bytearray`` bind round-trips to ``bytes``, NOT ``bytearray``."""
    encoded, vt = encode_value(bytearray(b"hello"))
    assert vt == ValueType.BLOB
    decoded, _ = decode_value(encoded, vt)
    assert type(decoded) is bytes
    assert decoded == b"hello"


def test_decode_blob_returns_bytes_for_memoryview_input() -> None:
    """A ``memoryview`` bind round-trips to ``bytes``, NOT ``memoryview``."""
    encoded, vt = encode_value(memoryview(b"hello"))
    assert vt == ValueType.BLOB
    decoded, _ = decode_value(encoded, vt)
    assert type(decoded) is bytes
    assert decoded == b"hello"


def test_decode_blob_returns_bytes_when_input_is_memoryview_buffer() -> None:
    """The ``data`` argument shape (bytes vs memoryview) does not affect the
    return type either — always bytes."""
    encoded = encode_blob(b"world")
    via_bytes, _ = decode_blob(encoded)
    via_memoryview, _ = decode_blob(memoryview(encoded))
    assert type(via_bytes) is bytes
    assert type(via_memoryview) is bytes
    assert via_bytes == via_memoryview == b"world"
