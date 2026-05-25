"""Pin: ``ReadBuffer.feed``'s oversize-reject self-poisons when the
buffer already holds *any* unconsumed pre-header bytes (not just a
full 8-byte header).

The previous gate used ``available() >= HEADER_SIZE``, missing the
1-7-stray-bytes case in which a peer chunked a header in two sends
and the second chunk overflowed the cap. A caller following the
``reset()`` recovery path would clobber the buffered header prefix
and silently desync on the next ``feed()``. Widen the predicate to
``available() > 0`` and pin the regression both on the entry-side
(``len(data) > 2 * max_message_size``) and the projection-side
(``projected > max_message_size``) gates.
"""

from __future__ import annotations

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.exceptions import DecodeError


def test_projection_side_oversize_with_partial_header_self_poisons() -> None:
    buf = ReadBuffer(max_message_size=64)
    # Prime buffer with 5 bytes (less than HEADER_SIZE=8).
    buf.feed(b"\x01\x02\x03\x04\x05")
    assert buf.available() == 5
    # Push a chunk that overflows the cap (5 + 80 = 85 > 64).
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(b"\xaa" * 80)
    assert buf.is_poisoned is True


def test_entry_side_oversize_with_partial_header_self_poisons() -> None:
    buf = ReadBuffer(max_message_size=64)
    buf.feed(b"\x01\x02\x03\x04\x05")
    # Push a chunk that exceeds 2x max_message_size; the entry-side
    # reject fires before the projection check.
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(b"\xaa" * (2 * 64 + 1))
    assert buf.is_poisoned is True


def test_oversize_with_empty_buffer_still_does_not_poison() -> None:
    """Negative regression: the genuinely safe-reset case (empty buffer,
    no skip in flight) is preserved — DecodeError but no poison."""
    buf = ReadBuffer(max_message_size=64)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(b"\xaa" * 80)
    assert buf.is_poisoned is False
    # And reset() is safe to call.
    buf.reset()
    assert buf.is_poisoned is False
