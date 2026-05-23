"""Pin: ``encode_blob`` accepts any bytes-like input (``bytes``,
``bytearray``, ``memoryview``), matching the project's documented
``WireInput`` contract and the discipline already followed by sibling
encoders (``decode_uint64``, ``WriteBuffer.write``, etc.).

Before this fix the public surface annotated ``value: bytes`` and the
runtime body rejected ``memoryview`` at the concatenation step. The
parent ``encode_value(BLOB)`` always materialised via ``bytes(value)``
before delegating, so production callers were safe — but external
callers using ``dqlitewire.types.encode_blob`` directly (mock servers,
golden-byte harnesses, fuzzers) had no signal that they needed to
materialise themselves.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_blob


@pytest.mark.parametrize(
    "value",
    [
        b"hello",
        bytearray(b"hello"),
        memoryview(b"hello"),
        memoryview(bytearray(b"hello")),
    ],
    ids=["bytes", "bytearray", "memoryview-of-bytes", "memoryview-of-bytearray"],
)
def test_encode_blob_accepts_bytes_like_inputs(value: object) -> None:
    encoded = encode_blob(value)  # type: ignore[arg-type]
    # All bytes-like inputs with the same content must produce the
    # same wire bytes — symmetry pinned at golden-byte level.
    expected = encode_blob(b"hello")
    assert encoded == expected


def test_encode_blob_rejects_non_bytes_like() -> None:
    # str is the canonical footgun — Python's "str is bytes-like in some
    # APIs" inconsistency means a caller could accidentally pass a str.
    with pytest.raises(EncodeError, match="Blob value must be bytes"):
        encode_blob("hello")  # type: ignore[arg-type]


def test_encode_blob_rejects_object_type() -> None:
    with pytest.raises(EncodeError, match="Blob value must be bytes"):
        encode_blob(object())  # type: ignore[arg-type]
