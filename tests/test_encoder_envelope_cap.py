"""MessageEncoder must cap total frame size against max_message_size
symmetrically with the decoder: per-field caps alone let a composite frame
encode to bytes the matching default-cap decoder then rejects."""

from __future__ import annotations

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.codec import MessageDecoder, MessageEncoder, encode_message
from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.responses import FilesResponse


def _compose_files(n: int, size: int) -> dict[str, bytes]:
    return {f"f{i:03d}.dat": b"\x00" * size for i in range(n)}


def test_message_encoder_rejects_over_cap_frame() -> None:
    # 80 files at 1 MiB each = ~80 MiB; default 64 MiB cap rejects.
    files = _compose_files(80, 1 << 20)
    enc = MessageEncoder(max_message_size=64 * 1024 * 1024)
    with pytest.raises(EncodeError, match="exceeds maximum"):
        enc.encode(FilesResponse(files=files))


def test_message_encoder_default_cap_matches_default_read_buffer_cap() -> None:
    enc = MessageEncoder()
    frame = enc.encode(FilesResponse(files={"a": b"\x00" * 16}))
    assert len(frame) <= ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE


def test_message_encoder_explicit_loose_cap_accepts_legacy_oversized() -> None:
    enc = MessageEncoder(max_message_size=1 << 30)  # 1 GiB
    files = _compose_files(10, 1 << 20)  # ~10 MiB total
    frame = enc.encode(FilesResponse(files=files))
    dec = MessageDecoder(max_message_size=1 << 30)
    dec.feed(frame)
    decoded = dec.decode()
    assert isinstance(decoded, FilesResponse)


def test_message_encoder_round_trip_at_default_cap_succeeds() -> None:
    enc = MessageEncoder()
    # FilesResponse content must be 8-byte aligned per the dqlite file-entry spec.
    frame = enc.encode(FilesResponse(files={"small.dat": b"\x01\x02\x03\x04\x05\x06\x07\x08"}))
    dec = MessageDecoder()
    dec.feed(frame)
    decoded = dec.decode()
    assert isinstance(decoded, FilesResponse)
    assert decoded.files == {"small.dat": b"\x01\x02\x03\x04\x05\x06\x07\x08"}


def test_message_encoder_rejects_invalid_max_message_size() -> None:
    with pytest.raises(ValueError, match="max_message_size"):
        MessageEncoder(max_message_size=0)
    with pytest.raises(ValueError, match="max_message_size"):
        MessageEncoder(max_message_size=-1)


def test_encode_message_helper_threads_max_message_size() -> None:
    files = _compose_files(80, 1 << 20)
    with pytest.raises(EncodeError, match="exceeds maximum"):
        encode_message(FilesResponse(files=files))
    frame = encode_message(FilesResponse(files=files), max_message_size=1 << 30)
    assert isinstance(frame, bytes)


def test_message_encoder_overflow_diagnostic_shape() -> None:
    files = _compose_files(80, 1 << 20)
    enc = MessageEncoder(max_message_size=64 * 1024 * 1024)
    with pytest.raises(EncodeError) as exc_info:
        enc.encode(FilesResponse(files=files))
    msg = str(exc_info.value)
    assert "exceeds maximum" in msg
    assert str(64 * 1024 * 1024) in msg
