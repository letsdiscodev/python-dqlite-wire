"""``encode_blob``/``decode_blob`` accept a ``max_blob_size`` kwarg overriding
the 16 MiB default."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.types import _MAX_BLOB_SIZE, decode_blob, encode_blob


def test_default_max_blob_size_unchanged() -> None:
    oversize = b"\x00" * (_MAX_BLOB_SIZE + 1)
    with pytest.raises(EncodeError, match="exceeds maximum"):
        encode_blob(oversize)


def test_caller_can_lower_encode_cap() -> None:
    with pytest.raises(EncodeError, match="exceeds maximum"):
        encode_blob(b"\x00" * 100, max_blob_size=99)


def test_caller_can_raise_encode_cap() -> None:
    big = b"\x00" * (_MAX_BLOB_SIZE + 16)
    encoded = encode_blob(big, max_blob_size=_MAX_BLOB_SIZE * 2)
    decoded, _ = decode_blob(encoded, max_blob_size=_MAX_BLOB_SIZE * 2)
    assert decoded == big


def test_caller_can_lower_decode_cap() -> None:
    payload = encode_blob(b"\x00" * 100)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        decode_blob(payload, max_blob_size=50)
