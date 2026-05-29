"""Pin: ``ReadBuffer.feed``'s oversize-reject self-poisons when the
buffer holds *any* unconsumed pre-header bytes (not just a full 8-byte
header) — the gate is ``available() > 0``, covering the 1-7-stray-bytes
case where a peer chunked a header and the second chunk overflowed."""

from __future__ import annotations

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.exceptions import DecodeError


def test_projection_side_oversize_with_partial_header_self_poisons() -> None:
    buf = ReadBuffer(max_message_size=64)
    # Prime with 5 bytes (< HEADER_SIZE=8), then overflow the cap.
    buf.feed(b"\x01\x02\x03\x04\x05")
    assert buf.available() == 5
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(b"\xaa" * 80)
    assert buf.is_poisoned is True


def test_entry_side_oversize_with_partial_header_self_poisons() -> None:
    buf = ReadBuffer(max_message_size=64)
    buf.feed(b"\x01\x02\x03\x04\x05")
    # Exceeds 2x cap: the entry-side reject fires before the projection check.
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(b"\xaa" * (2 * 64 + 1))
    assert buf.is_poisoned is True


def test_oversize_with_empty_buffer_still_does_not_poison() -> None:
    """Safe-reset case (empty buffer, no skip): DecodeError but no poison."""
    buf = ReadBuffer(max_message_size=64)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(b"\xaa" * 80)
    assert buf.is_poisoned is False
    buf.reset()
    assert buf.is_poisoned is False
