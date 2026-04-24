"""``encode_blob`` and ``decode_blob`` accept an optional
``max_blob_size`` kwarg so callers handling deployments with larger
``raft.log.entry_size_max`` can override the defense-in-depth
default (16 MiB) without monkey-patching the module constant.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.types import _MAX_BLOB_SIZE, decode_blob, encode_blob


def test_default_max_blob_size_unchanged() -> None:
    """Default cap is the module-level ``_MAX_BLOB_SIZE`` so existing
    callers see no behavior change."""
    oversize = b"\x00" * (_MAX_BLOB_SIZE + 1)
    with pytest.raises(EncodeError, match="exceeds maximum"):
        encode_blob(oversize)


def test_caller_can_lower_encode_cap() -> None:
    """A caller can tighten the cap below the default to reject blobs
    earlier on a constrained deployment."""
    with pytest.raises(EncodeError, match="exceeds maximum"):
        encode_blob(b"\x00" * 100, max_blob_size=99)


def test_caller_can_raise_encode_cap() -> None:
    """A caller with a larger ``raft.log.entry_size_max`` can raise the
    cap to send blobs above the default 16 MiB."""
    big = b"\x00" * (_MAX_BLOB_SIZE + 16)
    encoded = encode_blob(big, max_blob_size=_MAX_BLOB_SIZE * 2)
    decoded, _ = decode_blob(encoded, max_blob_size=_MAX_BLOB_SIZE * 2)
    assert decoded == big


def test_caller_can_lower_decode_cap() -> None:
    """A caller decoding from an untrusted peer can tighten the cap to
    reject anything above their own per-deployment ceiling."""
    payload = encode_blob(b"\x00" * 100)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        decode_blob(payload, max_blob_size=50)
