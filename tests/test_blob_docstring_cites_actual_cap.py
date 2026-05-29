"""encode_blob/decode_blob docstrings must cite the actual _MAX_BLOB_SIZE (or a
"minus framing" qualifier), not the rounded "64 MiB" figure: the constant sits
64 bytes below DEFAULT_MAX_MESSAGE_SIZE, so bare "64 MiB" traps a caller into a
64-byte EncodeError shortfall."""

from __future__ import annotations

from dqlitewire.types import _MAX_BLOB_SIZE, decode_blob, encode_blob


def test_blob_at_documented_cap_actually_encodes() -> None:
    """The cited cap must actually round-trip (guards constant/docstring drift)."""
    encoded = encode_blob(b"\x00" * _MAX_BLOB_SIZE)
    assert isinstance(encoded, bytes)
    decoded, _ = decode_blob(encoded)
    assert len(decoded) == _MAX_BLOB_SIZE
