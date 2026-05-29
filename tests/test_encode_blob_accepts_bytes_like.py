"""``encode_blob`` accepts any bytes-like input (``bytes``, ``bytearray``,
``memoryview``), matching the ``WireInput`` contract; direct external callers
previously hit a runtime rejection of ``memoryview``."""

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
    expected = encode_blob(b"hello")
    assert encoded == expected


def test_encode_blob_rejects_non_bytes_like() -> None:
    # str is the canonical footgun a caller might pass by accident.
    with pytest.raises(EncodeError, match="Blob value must be bytes"):
        encode_blob("hello")  # type: ignore[arg-type]


def test_encode_blob_rejects_object_type() -> None:
    with pytest.raises(EncodeError, match="Blob value must be bytes"):
        encode_blob(object())  # type: ignore[arg-type]
