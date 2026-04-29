"""Pin: ``mmap.mmap`` must infer to ``ValueType.BLOB``.

Stdlib ``sqlite3`` accepts ``mmap.mmap`` as a BLOB bind via the
buffer protocol (``PyObject_CheckBuffer`` →
``SQLITE_BLOB``). Without the explicit fallthrough, large-blob /
memory-mapped-file workflows that pass mmap regions to avoid an
extra copy hit ``EncodeError`` in dqlite while round-tripping
fine in stdlib sqlite3.

The inference discriminator at ``types.py`` previously listed
``(bytes, bytearray, memoryview)`` only — narrowed in ISSUE-347 from
"bytes only" but stopping short of buffer-protocol generality.
``array.array`` rejection (per ISSUE-765) is intentional and
preserved by adding ``mmap.mmap`` explicitly rather than widening to
``collections.abc.Buffer``.
"""

from __future__ import annotations

import mmap

from dqlitewire.constants import ValueType
from dqlitewire.types import decode_value, encode_value


def test_encode_value_infers_blob_from_mmap() -> None:
    payload = b"hello world!"
    mm = mmap.mmap(-1, len(payload))
    try:
        mm.write(payload)
        mm.seek(0)
        encoded, vtype = encode_value(mm)
        assert vtype == ValueType.BLOB
        decoded, _ = decode_value(encoded, ValueType.BLOB)
        assert decoded == payload
    finally:
        mm.close()


def test_encode_value_blob_explicit_accepts_mmap() -> None:
    """The explicit-BLOB branch (caller-supplied ``ValueType.BLOB``)
    must also accept ``mmap.mmap`` — symmetric with the inference
    branch so callers can use either entry point."""
    payload = b"explicit blob payload"
    mm = mmap.mmap(-1, len(payload))
    try:
        mm.write(payload)
        mm.seek(0)
        encoded, vtype = encode_value(mm, ValueType.BLOB)
        assert vtype == ValueType.BLOB
        decoded, _ = decode_value(encoded, ValueType.BLOB)
        assert decoded == payload
    finally:
        mm.close()


def test_encode_value_blob_mmap_decodes_to_bytes() -> None:
    """The decoder yields ``bytes`` (not ``mmap.mmap``); pin the
    decoded type so a future change that returned a memoryview-style
    view would break here rather than only at the ORM layer."""
    raw = b"mmap-payload-content"
    mm = mmap.mmap(-1, len(raw))
    try:
        mm.write(raw)
        mm.seek(0)
        encoded, _ = encode_value(mm, ValueType.BLOB)
        decoded, _ = decode_value(encoded, ValueType.BLOB)
        assert isinstance(decoded, bytes)
        assert decoded == raw
    finally:
        mm.close()
