"""``decode_blob`` always returns ``bytes`` regardless of whether the
encode-side input was ``bytes``, ``bytearray``, or ``memoryview``.

This pins the round-trip type asymmetry the ``decode_blob`` docstring
calls out. Stdlib ``sqlite3`` has the same behaviour (``sqlite3.Binary
= memoryview`` on bind; ``bytes`` on fetch). The wire-layer contract is
"owned ``bytes`` across the deserialisation boundary" — operators
relying on memoryview-only APIs (``.nbytes``, ``.cast()``) on the
readback must wrap explicitly.

If the contract is ever tightened to re-wrap as ``memoryview`` on
decode, delete this pin AND update the ``decode_blob`` docstring.
"""

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
    """The ``data`` argument shape to ``decode_blob`` (``bytes`` vs
    ``memoryview``) also does not affect the return type — always
    ``bytes``."""
    encoded = encode_blob(b"world")
    via_bytes, _ = decode_blob(encoded)
    via_memoryview, _ = decode_blob(memoryview(encoded))
    assert type(via_bytes) is bytes
    assert type(via_memoryview) is bytes
    assert via_bytes == via_memoryview == b"world"


def test_decode_blob_docstring_documents_type_asymmetry() -> None:
    """The docstring must surface the round-trip type asymmetry so
    callers binding memoryview/bytearray are not surprised by the
    ``bytes`` readback."""
    doc = decode_blob.__doc__ or ""
    assert "memoryview" in doc
    assert "bytes" in doc
    # The asymmetry must be called out as such, not merely mentioned in
    # passing as an accepted input shape.
    assert "asymmetry" in doc.lower() or "always" in doc.lower()
