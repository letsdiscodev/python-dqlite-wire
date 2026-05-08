"""Pin: ``ReadBuffer.skip_message`` returns ``False`` in the
immediate-poison sub-branch (the entire oversize message body is
already in the buffer when we call skip_message), even though
``_skip_remaining`` reaches zero synchronously.

Pre-fix the method returned ``self._skip_remaining == 0`` which
evaluated to ``True`` after the synchronous ``self.poison(...)``
call, contradicting the docstring contract: a caller doing
``if buf.skip_message(): continue_decoding()`` would proceed and
hit ``PoisonedError`` on the next decode with no warning.

The contract is now: ``True`` means "skip succeeded AND the buffer
remains usable"; ``False`` means "more data needed OR the buffer is
now poisoned (caller must ``reset()``)".
"""

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.constants import HEADER_SIZE, WORD_SIZE
from dqlitewire.exceptions import ProtocolError


def _build_header(declared_words: int) -> bytes:
    """Build an 8-byte header declaring ``declared_words`` words of body."""
    size_words_le = declared_words.to_bytes(4, "little")
    type_byte = b"\x01"
    schema_byte = b"\x00"
    extra = b"\x00\x00"
    header = size_words_le + type_byte + schema_byte + extra
    assert len(header) == HEADER_SIZE
    return header


def test_skip_message_immediate_poison_returns_false() -> None:
    """The "immediate-poison" sub-branch fires when the *declared*
    total_size exceeds max_message_size, the body is only partially
    present (bounded by the cap), and ``effective_total == available``.
    In that branch _skip_remaining drops to zero synchronously and
    poison fires inline; pre-fix the method then returned True (the
    raw ``_skip_remaining == 0`` value) which contradicts the
    "next decode can proceed" docstring contract.
    """
    cap = 64
    buf = ReadBuffer(max_message_size=cap)
    # Header declares far more bytes than the cap. We feed exactly
    # ``cap`` bytes (header + body slice that fits under the projection
    # check). At skip_message() time:
    #   total_size = HEADER_SIZE + 100*WORD_SIZE = 808 (>> cap)
    #   effective_total = min(808, 64) = 64
    #   available = 64
    #   skip_now = 64; _skip_remaining = 0; effective_total < total_size
    declared_words = 100
    body_bytes_to_feed = cap - HEADER_SIZE
    payload = _build_header(declared_words) + b"\x00" * body_bytes_to_feed
    assert len(payload) == cap
    buf.feed(payload)

    result = buf.skip_message()
    assert result is False, (
        "skip_message must return False when it synchronously poisons "
        "the buffer; True would falsely advertise that the next decode "
        "can proceed."
    )
    # Confirm the buffer is in fact poisoned now.
    assert buf.is_poisoned
    with pytest.raises(ProtocolError):
        buf.feed(b"\x00" * HEADER_SIZE)


def test_skip_message_returns_false_for_deferred_poison_path() -> None:
    """When the body is only partially in the buffer (deferred poison),
    skip_message also returns False — _skip_remaining > 0 here. Pin
    both branches so the contract is symmetric."""
    cap = 64
    buf = ReadBuffer(max_message_size=cap)
    declared_words = 100
    # Feed header + a few body bytes (less than the cap so deferred path).
    payload = _build_header(declared_words) + b"\x00" * (WORD_SIZE * 2)
    buf.feed(payload)
    assert buf.skip_message() is False
    assert not buf.is_poisoned  # deferred — poison fires when feed completes the skip
