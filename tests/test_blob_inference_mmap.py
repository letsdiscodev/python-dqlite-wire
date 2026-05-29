"""``mmap.mmap`` infers to ``ValueType.BLOB`` (parity with stdlib sqlite3)."""

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
    """The decoder yields ``bytes``, not a view over ``mmap.mmap``."""
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
