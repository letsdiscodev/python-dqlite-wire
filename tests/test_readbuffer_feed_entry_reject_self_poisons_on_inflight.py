"""Pin: ``ReadBuffer.feed``'s entry-side oversize reject (chunk >
2 * max_message_size) self-poisons when an in-flight body is present,
so the caller cannot misapply the "safe reset" recovery and desync."""

from __future__ import annotations

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.constants import HEADER_SIZE
from dqlitewire.exceptions import DecodeError, PoisonedError


def _make_buffer(cap: int = 64 * 1024) -> ReadBuffer:
    return ReadBuffer(max_message_size=cap)


def test_entry_side_oversize_reject_on_empty_buffer_does_not_poison() -> None:
    """Safe-reset case: rejection with no in-flight body must NOT poison."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    huge = b"\x00" * (2 * cap + 1)
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(huge)
    assert buf.is_poisoned is False, (
        "rejection on an empty buffer must not poison — the documented "
        "safe-reset case still has to hold"
    )
    buf.reset()
    buf.feed(b"\x00" * 16)


def test_entry_side_oversize_reject_with_partial_header_in_buffer_poisons() -> None:
    """In-flight frame present (>= HEADER_SIZE unconsumed): the entry-side
    reject MUST self-poison so safe-reset recovery cannot be misapplied."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    primer = b"\x00" * HEADER_SIZE
    buf.feed(primer)
    assert len(buf._data) - buf._pos >= HEADER_SIZE

    huge = b"\x00" * (2 * cap + 1)
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(huge)

    # A subsequent feed must raise rather than silently accept desynced bytes.
    assert buf.is_poisoned is True
    with pytest.raises(PoisonedError):
        buf.feed(b"\x00" * 16)


def test_entry_side_oversize_reject_with_skip_in_flight_poisons() -> None:
    """Same discipline when ``_skip_remaining > 0`` (skip in flight)."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    buf._skip_remaining = 1024

    huge = b"\x00" * (2 * cap + 1)
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(huge)

    assert buf.is_poisoned is True
    assert buf._skip_remaining == 0


def test_projection_side_reject_with_partial_header_in_buffer_poisons() -> None:
    """Projection-side: a chunk passing the entry-side 2x check but pushing
    projected > cap also self-poisons when an in-flight body is present."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    primer = b"\x00" * (cap - HEADER_SIZE)
    buf.feed(primer)
    assert len(buf._data) - buf._pos >= HEADER_SIZE

    chunk = b"\x00" * (HEADER_SIZE + 1)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(chunk)

    assert buf.is_poisoned is True
