"""Pin: when ``ReadBuffer.feed()``'s size-projection check fires
while ``_skip_remaining > 0`` (a skip is in flight), the buffer
self-poisons.

Before this fix, a feed-time DecodeError in skip mode left
``_skip_remaining`` stale and ``_poisoned`` clear: the caller would
have to read the docstring's recovery semantics carefully to learn
that the wire is desynchronised and the connection must be dropped.
The architect-preferred contract is to advertise that fact via
``_poisoned`` so the caller's natural ``except DecodeError`` plus
subsequent feed/decode call surfaces ``PoisonedError`` instead of
quietly continuing on a desync'd buffer.
"""

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
    # Step 1: feed an oversize header + body (just enough body to fit
    # under the cap) and let skip_message arm the deferred-poison path.
    declared_words = 100
    body_bytes_under_cap = WORD_SIZE  # one body word, well under cap
    payload = _build_header(declared_words) + b"\x00" * body_bytes_under_cap
    buf.feed(payload)
    assert buf.skip_message() is False
    assert buf._skip_remaining > 0
    assert not buf.is_poisoned

    # Step 2: feed a chunk whose post-skip-discard remainder still
    # exceeds the cap. With cap=64, _skip_remaining=cap-feed_so_far
    # of the in-flight oversize body, the cleanest way to trigger the
    # projection check is to push a chunk such that
    # ``len(data) - _skip_remaining + len(_data) - _pos`` > cap.
    # We craft data of length cap + skip_remaining + 1 so the residual
    # past skip is cap + 1, which projects above cap.
    bogus = b"\x00" * (cap + buf._skip_remaining + 1)
    # The early >2*cap gate may fire first if bogus is too large.
    # Cap=64, so 2*cap=128. Keep bogus under that gate.
    if len(bogus) > 2 * cap:
        # Tighter cap so the >2*cap gate doesn't fire first.
        bogus = b"\x00" * (2 * cap)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(bogus)

    # The architect-preferred contract: the buffer self-poisons because
    # the wire is now desync'd (the rejected bytes contained body for
    # the in-flight skip, which is lost; subsequent peer bytes are
    # mis-attributed).
    assert buf.is_poisoned, (
        "feed() must self-poison when its projection check fires while "
        "_skip_remaining > 0; the wire is desync'd and the buffer must "
        "advertise that fact rather than silently continue."
    )


def test_feed_rejection_while_skipping_clears_skip_tracking_fields() -> None:
    """After self-poison, ``_skip_remaining`` and ``_poison_after_skip``
    must be cleared so any post-poison introspection (or a future
    ``recover()`` helper) sees consistent unrecoverable state. The
    buffer is unrecoverable until ``reset()`` regardless, so this is
    diagnostic-consistency defence; pre-fix the fields stayed at their
    pre-rejection values, contradicting the poisoned-and-cleared
    semantics that ``reset()`` enforces."""
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
