"""Pin: ``ReadBuffer.feed``'s entry-side oversize reject (chunk >
2 * max_message_size) self-poisons the buffer when an in-flight body
is present, mirroring the projection-side reject's discipline.

Without this, the docstring's documented "safe reset" recovery
could be misapplied — the caller would call ``reset()`` thinking
the buffer was empty, but the wire is mid-body and the next feed
silently continues on a desynchronised offset.
"""

from __future__ import annotations

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.constants import HEADER_SIZE
from dqlitewire.exceptions import DecodeError, PoisonedError


def _make_buffer(cap: int = 64 * 1024) -> ReadBuffer:
    return ReadBuffer(max_message_size=cap)


def test_entry_side_oversize_reject_on_empty_buffer_does_not_poison() -> None:
    """The safe-reset case: rejection with no in-flight body must
    NOT poison. The caller can ``reset()`` and continue."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    huge = b"\x00" * (2 * cap + 1)
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(huge)
    assert buf.is_poisoned is False, (
        "rejection on an empty buffer must not poison — the documented "
        "safe-reset case still has to hold"
    )
    # And the caller can recover via reset() + feed of a legal-size
    # chunk.
    buf.reset()
    buf.feed(b"\x00" * 16)


def test_entry_side_oversize_reject_with_partial_header_in_buffer_poisons() -> None:
    """When the buffer already holds an in-flight frame (≥ HEADER_SIZE
    unconsumed bytes), the entry-side reject MUST self-poison so the
    caller cannot misapply the safe-reset recovery."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    # Prime the buffer with one complete header — represents an
    # in-flight frame whose body is being awaited.
    primer = b"\x00" * HEADER_SIZE
    buf.feed(primer)
    assert len(buf._data) - buf._pos >= HEADER_SIZE

    huge = b"\x00" * (2 * cap + 1)
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(huge)

    # Load-bearing: the buffer self-poisoned. A subsequent feed /
    # read_message must raise PoisonedError rather than silently
    # accept (offset-by-unknown) bytes.
    assert buf.is_poisoned is True
    with pytest.raises(PoisonedError):
        buf.feed(b"\x00" * 16)


def test_entry_side_oversize_reject_with_skip_in_flight_poisons() -> None:
    """Same discipline when ``_skip_remaining > 0`` — symmetric
    with the projection-side reject's existing arm."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    # Simulate a skip in flight (oversized message being drained).
    buf._skip_remaining = 1024

    huge = b"\x00" * (2 * cap + 1)
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(huge)

    assert buf.is_poisoned is True
    # Skip-tracking fields cleared for diagnostic consistency.
    assert buf._skip_remaining == 0


def test_projection_side_reject_with_partial_header_in_buffer_poisons() -> None:
    """Symmetric pin on the projection-side: even with a chunk that
    passes the entry-side 2x check, the projection check now also
    self-poisons when an in-flight body is present."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    # Prime the buffer with the maximum legal payload minus one header.
    primer = b"\x00" * (cap - HEADER_SIZE)
    buf.feed(primer)
    assert len(buf._data) - buf._pos >= HEADER_SIZE

    # A second chunk under 2x cap but pushing projected > cap.
    chunk = b"\x00" * (HEADER_SIZE + 1)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(chunk)

    assert buf.is_poisoned is True
