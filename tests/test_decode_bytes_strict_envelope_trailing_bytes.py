"""Pin: ``MessageDecoder.decode_bytes`` (and the ``decode_message`` helper)
rejects envelope trailing bytes beyond ``HEADER_SIZE + body_size``, for
strict-decode parity with the per-message body decoders. The streaming path
returns exact-length frames, so only direct callers hit the previously-lax
envelope strip."""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder, decode_message, encode_message
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import LeaderResponse


def test_decode_message_rejects_envelope_trailing_bytes() -> None:
    msg = LeaderResponse(node_id=5, address="host:9001")
    valid_bytes = encode_message(msg)
    with pytest.raises(DecodeError, match="trailing"):
        decode_message(valid_bytes + b"\x00\x00\x00\x00\x00\x00\x00\x00")


def test_decode_message_accepts_exact_length() -> None:
    msg = LeaderResponse(node_id=5, address="host:9001")
    valid_bytes = encode_message(msg)
    decoded = decode_message(valid_bytes)
    assert isinstance(decoded, LeaderResponse)
    assert decoded.address == "host:9001"


def test_decode_bytes_rejects_envelope_trailing_bytes_via_class() -> None:
    """Same reject via the ``MessageDecoder.decode_bytes`` entry point."""
    msg = LeaderResponse(node_id=5, address="host:9001")
    valid_bytes = encode_message(msg)
    decoder = MessageDecoder(is_request=False)
    with pytest.raises(DecodeError, match="trailing"):
        decoder.decode_bytes(valid_bytes + b"GARBAGE!")


def test_decode_message_short_input_still_diagnoses_short_body() -> None:
    """The 'body too short' diagnostic still fires for under-size input — the
    trailing-bytes check covers only over-size."""
    from dqlitewire.constants import ResponseType
    from dqlitewire.messages.base import Header

    # Header claims 2 words (16 bytes) but only 8 bytes follow.
    header = Header(size_words=2, msg_type=ResponseType.LEADER, schema=0)
    short_data = header.encode() + b"\x00" * 8
    with pytest.raises(DecodeError, match="[Bb]ody.*short"):
        decode_message(short_data, is_request=False)


def test_decode_message_zero_trailing_bytes_is_canonical() -> None:
    """Exact-length frames are canonical and must decode cleanly (guard
    against a regression that rejects them)."""
    from dqlitewire.constants import HEADER_SIZE
    from dqlitewire.messages.responses import FailureResponse

    msg = FailureResponse(code=1, message="brief")
    valid_bytes = encode_message(msg)
    decoded = decode_message(valid_bytes)
    assert isinstance(decoded, FailureResponse)
    assert len(valid_bytes) >= HEADER_SIZE
