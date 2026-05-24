"""Pin: ``MessageDecoder.decode_bytes`` (reached via the
``decode_message`` convenience helper) rejects envelope trailing
bytes beyond ``HEADER_SIZE + body_size``.

Every per-message decoder (FailureResponse, LeaderResponse,
RowsResponse, ServersResponse, FilesResponse, all request decoders)
maintains a strict ``if offset != len(data): raise DecodeError
("trailing bytes")`` posture for its own body. The envelope-level
strip in ``MessageDecoder.decode_bytes`` previously sliced silently
on any input length >= ``HEADER_SIZE + body_size`` — so a caller
passing ``data = header + body + garbage`` got a successful decode
with no indication that the input was malformed.

The streaming path (``ReadBuffer.read_message``) returns exactly
``HEADER_SIZE + body_size`` bytes so the envelope-level strict check
is unreachable through streaming. But ``decode_bytes`` /
``decode_message`` exposes the laxness to direct callers (mock
servers, golden-byte harnesses, fuzzers, packet replay tools).
Strict-decode parity with the per-message decoders is the
documented invariant.
"""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder, decode_message, encode_message
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import LeaderResponse


def test_decode_message_rejects_envelope_trailing_bytes() -> None:
    """A valid frame + trailing garbage is rejected as malformed."""
    msg = LeaderResponse(node_id=5, address="host:9001")
    valid_bytes = encode_message(msg)
    with pytest.raises(DecodeError, match="trailing"):
        decode_message(valid_bytes + b"\x00\x00\x00\x00\x00\x00\x00\x00")


def test_decode_message_accepts_exact_length() -> None:
    """A valid frame with no trailing bytes still decodes cleanly."""
    msg = LeaderResponse(node_id=5, address="host:9001")
    valid_bytes = encode_message(msg)
    decoded = decode_message(valid_bytes)
    assert isinstance(decoded, LeaderResponse)
    assert decoded.address == "host:9001"


def test_decode_bytes_rejects_envelope_trailing_bytes_via_class() -> None:
    """Same reject via the ``MessageDecoder.decode_bytes`` entry
    point (not just the convenience helper)."""
    msg = LeaderResponse(node_id=5, address="host:9001")
    valid_bytes = encode_message(msg)
    decoder = MessageDecoder(is_request=False)
    with pytest.raises(DecodeError, match="trailing"):
        decoder.decode_bytes(valid_bytes + b"GARBAGE!")


def test_decode_message_short_input_still_diagnoses_short_body() -> None:
    """Regression: the existing 'body too short' diagnostic still
    fires for the under-size case — the new check covers only
    over-size."""
    from dqlitewire.constants import ResponseType
    from dqlitewire.messages.base import Header

    # Header claims 2 words (16 bytes) but only 8 bytes follow.
    header = Header(size_words=2, msg_type=ResponseType.LEADER, schema=0)
    short_data = header.encode() + b"\x00" * 8
    with pytest.raises(DecodeError, match="[Bb]ody.*short"):
        decode_message(short_data, is_request=False)


def test_decode_message_zero_trailing_bytes_is_canonical() -> None:
    """A frame with exactly ``HEADER_SIZE + body_size`` bytes is the
    canonical wire shape and must decode cleanly. Pin against a
    future regression that incorrectly rejects exact-length frames."""
    from dqlitewire.constants import HEADER_SIZE
    from dqlitewire.messages.responses import FailureResponse

    msg = FailureResponse(code=1, message="brief")
    valid_bytes = encode_message(msg)
    decoded = decode_message(valid_bytes)
    assert isinstance(decoded, FailureResponse)
    assert len(valid_bytes) >= HEADER_SIZE
