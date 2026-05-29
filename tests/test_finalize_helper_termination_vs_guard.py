"""Pin: guard arms (skip_message/decode raising ContinuationError
mid-continuation) must NOT call _finalize_continuation_state, since
that would clear _continuation_expected and lose the in-progress stream
invariant the guard enforces."""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder, MessageEncoder
from dqlitewire.exceptions import ContinuationError
from dqlitewire.messages.responses import RowsResponse


def _drive_to_continuation(decoder: MessageDecoder) -> None:
    initial = RowsResponse(
        column_names=["c0"],
        rows=[[1]],
        has_more=True,
    )
    decoder.feed(MessageEncoder().encode(initial))
    decoder.decode()
    assert decoder._continuation_expected is True


def test_skip_message_guard_preserves_continuation_flag() -> None:
    """skip_message mid-continuation raises but leaves the continuation
    flag True, so recovery is decode_continuation, not decode."""
    decoder = MessageDecoder(is_request=False)
    _drive_to_continuation(decoder)
    with pytest.raises(ContinuationError):
        decoder.skip_message()
    assert decoder._continuation_expected is True


def test_decode_mid_continuation_guard_preserves_continuation_flag() -> None:
    decoder = MessageDecoder(is_request=False)
    _drive_to_continuation(decoder)
    with pytest.raises(ContinuationError):
        decoder.decode()
    assert decoder._continuation_expected is True
