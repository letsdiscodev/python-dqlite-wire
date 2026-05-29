"""Pin: a feed() size-projection rejection while a skip is in flight
self-poisons, so the desync'd wire surfaces as PoisonedError rather
than silently continuing."""

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.constants import WORD_SIZE
from dqlitewire.exceptions import DecodeError


def _build_header(declared_words: int) -> bytes:
    size_words_le = declared_words.to_bytes(4, "little")
    return size_words_le + b"\x01" + b"\x00" + b"\x00\x00"


def test_feed_rejection_while_skipping_self_poisons() -> None:
    cap = 64
    buf = ReadBuffer(max_message_size=cap)
    # Feed an oversize header + under-cap body so skip_message arms the
    # deferred-poison path.
    declared_words = 100
    body_bytes_under_cap = WORD_SIZE  # one body word, well under cap
    payload = _build_header(declared_words) + b"\x00" * body_bytes_under_cap
    buf.feed(payload)
    assert buf.skip_message() is False
    assert buf._skip_remaining > 0
    assert not buf.is_poisoned

    # Feed a chunk whose post-skip-discard remainder still exceeds the
    # cap, triggering the projection check; keep it under the early
    # >2*cap gate so that gate doesn't fire first.
    bogus = b"\x00" * (cap + buf._skip_remaining + 1)
    if len(bogus) > 2 * cap:
        bogus = b"\x00" * (2 * cap)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(bogus)

    assert buf.is_poisoned, (
        "feed() must self-poison when its projection check fires while "
        "_skip_remaining > 0; the wire is desync'd and the buffer must "
        "advertise that fact rather than silently continue."
    )


def test_feed_rejection_while_skipping_clears_skip_tracking_fields() -> None:
    """After self-poison, _skip_remaining and _poison_after_skip must be
    cleared so post-poison introspection sees consistent state."""
    cap = 64
    buf = ReadBuffer(max_message_size=cap)
    declared_words = 100
    body_bytes_under_cap = WORD_SIZE
    payload = _build_header(declared_words) + b"\x00" * body_bytes_under_cap
    buf.feed(payload)
    assert buf.skip_message() is False
    assert buf._skip_remaining > 0

    bogus = b"\x00" * (cap + buf._skip_remaining + 1)
    if len(bogus) > 2 * cap:
        bogus = b"\x00" * (2 * cap)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(bogus)

    assert buf.is_poisoned
    assert buf._skip_remaining == 0, (
        "Self-poison branch must clear _skip_remaining; stale value "
        "would mislead any post-poison introspection."
    )
    assert buf._poison_after_skip is None, (
        "Self-poison branch must clear _poison_after_skip; the "
        "deferred-poison path is no longer reachable post-poison."
    )
