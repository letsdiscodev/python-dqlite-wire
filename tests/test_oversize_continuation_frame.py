"""Oversize continuation frames must be rejected before a body read.

``ReadBuffer`` enforces ``max_message_size`` uniformly on every
``feed()`` / ``read_message()`` call regardless of frame ordinal —
there is no "first vs. continuation" distinction in the buffer
layer. This module is a structural regression fence: a future
refactor that split the cap check into a "first-frame only" branch,
added a per-frame counter with an off-by-one, or weakened the cap
during continuation mode would silently re-open an amplification
channel. The tests below pin the contract from the MessageDecoder
call-site that an in-progress continuation respects the cap.
"""

from __future__ import annotations

from typing import cast

import pytest

from dqlitewire.buffer import HEADER_SIZE, WORD_SIZE
from dqlitewire.codec import MessageDecoder
from dqlitewire.constants import ResponseType, ValueType
from dqlitewire.exceptions import DecodeError, PoisonedError
from dqlitewire.messages.responses import RowsResponse


def _fabricate_oversize_rows_header(body_words: int) -> bytes:
    """Return a valid header that claims a body of ``body_words`` WORDs
    plus enough trailing garbage bytes so the buffer would accept the
    frame if no cap check fired. The body itself is never read — the
    cap check on ``feed()`` triggers first.
    """
    # Header layout: uint32 size_in_words | u8 type | u8 schema | u16 extra.
    header = (
        body_words.to_bytes(4, "little")
        + int(ResponseType.ROWS).to_bytes(1, "little")
        + (0).to_bytes(1, "little")
        + (0).to_bytes(2, "little")
    )
    assert len(header) == HEADER_SIZE
    # A tiny body tail — the buffer rejects before reading this; we
    # only need enough bytes so the buffer sees a complete-ish frame
    # arriving. ``feed()`` raises on size projection, not on content.
    return header + b"\x00" * 8


class TestContinuationFrameOverCap:
    """Cap check must apply to continuation frames identically to
    the initial frame."""

    def test_continuation_header_over_cap_rejected(self) -> None:
        """After a valid first ROWS frame with has_more=True, a
        continuation header declaring a body over the buffer cap must
        raise DecodeError once the decoder attempts to consume it.
        The continuation path does not have ``skip_message()`` recovery,
        so the decoder poisons and forces the caller to reset().
        """
        small_cap = 4096
        decoder = MessageDecoder(max_message_size=small_cap)

        # First frame: valid, has_more=True.
        first = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[1]],
            has_more=True,
        )
        first_bytes = first.encode()
        assert len(first_bytes) < small_cap

        decoder.feed(first_bytes)
        result = cast(RowsResponse, decoder.decode())
        assert result is not None
        assert result.has_more is True

        # Continuation header that claims a body size well over the cap.
        # The header+tail is only a handful of bytes so ``feed()`` accepts
        # it (total projected size is tiny) — the cap check fires when
        # ``decode_continuation()`` attempts to read the declared frame.
        oversize_body_words = (small_cap // WORD_SIZE) + 32
        oversize_frame = _fabricate_oversize_rows_header(oversize_body_words)
        decoder.feed(oversize_frame)

        with pytest.raises(DecodeError, match=r"exceeds maximum"):
            decoder.decode_continuation()

    def test_continuation_within_cap_decodes(self) -> None:
        """A legitimate continuation frame under the cap must decode
        cleanly — boundary fence against an over-eager cap that
        rejects anything in continuation mode.
        """
        small_cap = 64 * 1024
        decoder = MessageDecoder(max_message_size=small_cap)

        first = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[1]],
            has_more=True,
        )
        decoder.feed(first.encode())
        assert decoder.decode() is not None

        # Second frame: small, has_more=False to terminate the stream.
        second = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[2]],
            has_more=False,
        )
        decoder.feed(second.encode())
        cont = decoder.decode_continuation()
        assert cont is not None
        assert cont.has_more is False
        assert cont.rows == [[2]]


class TestContinuationPoisonedAfterOversize:
    """After an oversize continuation rejection, further decode calls
    must raise PoisonedError — the buffer has no safe recovery path
    during continuation mode."""

    def test_poisoned_after_oversize_continuation(self) -> None:
        small_cap = 4096
        decoder = MessageDecoder(max_message_size=small_cap)

        first = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[1]],
            has_more=True,
        )
        decoder.feed(first.encode())
        decoder.decode()

        oversize_body_words = (small_cap // WORD_SIZE) + 32
        oversize_frame = _fabricate_oversize_rows_header(oversize_body_words)
        decoder.feed(oversize_frame)

        with pytest.raises(DecodeError):
            decoder.decode_continuation()

        # Further operations must see the poisoned state — the
        # continuation path cannot recover in-place via skip_message().
        with pytest.raises(PoisonedError):
            decoder.decode_continuation()
