"""Pin: malformed EMPTY-body mid-continuation does NOT poison the
buffer — mirrors the FAILURE / StreamError precedents that re-raise
DecodeError without poison when ``read_message`` has already advanced
past a coherent frame.

A peer that emits a single EMPTY-typed frame with a non-8-byte body
should raise DecodeError but leave the buffer wire-coherent: the
caller can resync (e.g. by reading the next frame) without dropping
the connection.
"""

from __future__ import annotations

import struct

import pytest

from dqlitewire.codec import MessageDecoder
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.base import Header
from dqlitewire.messages.responses import EmptyResponse


def _build_empty_frame_with_garbage_length(body_words: int) -> bytes:
    body = struct.pack(f"<{body_words}Q", *([0] * body_words))
    header = Header(size_words=body_words, msg_type=8, schema=0)
    return header.encode() + body


def test_malformed_empty_in_continuation_raises_decode_error_without_poison() -> None:
    """A 16-byte EMPTY body mid-continuation: raise DecodeError, do
    NOT poison the buffer. ``read_message`` already advanced past the
    frame; the connection is wire-coherent and a subsequent frame
    should decode cleanly."""
    malformed = _build_empty_frame_with_garbage_length(2)

    decoder = MessageDecoder(is_request=False)
    decoder._continuation_expected = True
    decoder.feed(malformed)

    with pytest.raises(DecodeError, match="EmptyResponse"):
        decoder.decode_continuation()
    assert decoder.is_poisoned is False
    # The continuation state should also be finalised so the decoder
    # is ready for the next request on the connection.
    assert decoder._continuation_expected is False


def test_decoder_recovers_after_malformed_empty_mid_continuation() -> None:
    """After the malformed EMPTY raises, a well-formed EmptyResponse
    (8-byte body) decoded as a top-level message via decode_bytes
    succeeds — the decoder is not stuck."""
    malformed = _build_empty_frame_with_garbage_length(0)
    decoder = MessageDecoder(is_request=False)
    decoder._continuation_expected = True
    decoder.feed(malformed)
    with pytest.raises(DecodeError):
        decoder.decode_continuation()
    assert decoder.is_poisoned is False

    # Hand-craft a well-formed EmptyResponse and decode it via the
    # stateless path to confirm the decoder is not poisoned.
    body = struct.pack("<Q", 0)
    header = Header(size_words=1, msg_type=8, schema=0)
    wire = header.encode() + body
    result = decoder.decode_bytes(wire)
    assert isinstance(result, EmptyResponse)
