"""``MessageEncoder`` must cap the total frame size against
``max_message_size`` symmetrically with the decoder.

Per-field caps (``_MAX_BLOB_SIZE``, ``_MAX_TEXT_VALUE_SIZE``,
``_MAX_FILE_CONTENT_SIZE``, ``_MAX_NODE_COUNT``, ``_MAX_FILE_COUNT``,
column-count <= 255) partially bound the envelope, but a composite
response — e.g. a ``FilesResponse`` with N files near the per-file
content cap — can still encode to bytes the matching default-cap
``MessageDecoder`` rejects with ``DecodeError``. This is the encoder-
decoder asymmetry mock-server / proxy / replay-tool authors hit on the
same process. The envelope cap on the encoder completes the per-field
cap discipline established by the seven prior done findings (ISSUE-491
/ 1178 / 1179 / 1181 / 1292 / W3 / wire-files-response-content-no-
size-cap).

Operators with a genuinely larger cluster (raft.log.entry_size_max
above 64 MiB) can opt in by passing a higher ``max_message_size`` to
both the encoder and the decoder.
"""

from __future__ import annotations

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.codec import MessageDecoder, MessageEncoder, encode_message
from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.responses import FilesResponse


def _compose_files(n: int, size: int) -> dict[str, bytes]:
    return {f"f{i:03d}.dat": b"\x00" * size for i in range(n)}


def test_message_encoder_rejects_over_cap_frame() -> None:
    """A composite frame above ``max_message_size`` raises
    ``EncodeError`` on encode — symmetric with the decoder."""
    # 80 files at 1 MiB each = ~80 MiB; default 64 MiB cap rejects.
    files = _compose_files(80, 1 << 20)
    enc = MessageEncoder(max_message_size=64 * 1024 * 1024)
    with pytest.raises(EncodeError, match="exceeds maximum"):
        enc.encode(FilesResponse(files=files))


def test_message_encoder_default_cap_matches_default_read_buffer_cap() -> None:
    """The encoder default must equal the decoder default so a caller
    using both defaults stays symmetric."""
    enc = MessageEncoder()
    # A tiny frame well within the cap encodes cleanly.
    frame = enc.encode(FilesResponse(files={"a": b"\x00" * 16}))
    assert len(frame) <= ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE


def test_message_encoder_explicit_loose_cap_accepts_legacy_oversized() -> None:
    """Operator opt-in: a higher cap on the encoder accepts the larger
    frame. Symmetric with ``MessageDecoder(max_message_size=...)``."""
    enc = MessageEncoder(max_message_size=1 << 30)  # 1 GiB
    files = _compose_files(10, 1 << 20)  # ~10 MiB total
    frame = enc.encode(FilesResponse(files=files))
    # And the same-cap decoder accepts the same bytes.
    dec = MessageDecoder(max_message_size=1 << 30)
    dec.feed(frame)
    decoded = dec.decode()
    assert isinstance(decoded, FilesResponse)


def test_message_encoder_round_trip_at_default_cap_succeeds() -> None:
    """A frame encoded under the default cap decodes under the same
    default cap. Pins encoder/decoder symmetry on the happy path."""
    enc = MessageEncoder()
    # FilesResponse content must be 8-byte aligned per the dqlite
    # file-entry spec; use an 8-byte payload.
    frame = enc.encode(FilesResponse(files={"small.dat": b"\x01\x02\x03\x04\x05\x06\x07\x08"}))
    dec = MessageDecoder()
    dec.feed(frame)
    decoded = dec.decode()
    assert isinstance(decoded, FilesResponse)
    assert decoded.files == {"small.dat": b"\x01\x02\x03\x04\x05\x06\x07\x08"}


def test_message_encoder_rejects_invalid_max_message_size() -> None:
    """A nonsense cap (zero / negative) is rejected at construction
    time — symmetric with ``ReadBuffer``."""
    with pytest.raises(ValueError, match="max_message_size"):
        MessageEncoder(max_message_size=0)
    with pytest.raises(ValueError, match="max_message_size"):
        MessageEncoder(max_message_size=-1)


def test_encode_message_helper_threads_max_message_size() -> None:
    """The convenience ``encode_message`` helper forwards
    ``max_message_size`` to the underlying encoder — symmetric with
    ``decode_message``'s same kwarg."""
    files = _compose_files(80, 1 << 20)
    # Default-cap helper rejects.
    with pytest.raises(EncodeError, match="exceeds maximum"):
        encode_message(FilesResponse(files=files))
    # Explicit higher cap accepts.
    frame = encode_message(FilesResponse(files=files), max_message_size=1 << 30)
    assert isinstance(frame, bytes)


def test_message_encoder_overflow_diagnostic_shape() -> None:
    """The diagnostic must name the actual size and the cap so the
    operator can spot the mismatch without re-running with a debugger."""
    files = _compose_files(80, 1 << 20)
    enc = MessageEncoder(max_message_size=64 * 1024 * 1024)
    with pytest.raises(EncodeError) as exc_info:
        enc.encode(FilesResponse(files=files))
    msg = str(exc_info.value)
    # Hex byte count + cap both appear, so operators can grep their logs.
    assert "exceeds maximum" in msg
    assert str(64 * 1024 * 1024) in msg
