"""Pin: ``_finalize_continuation_state``'s docstring distinguishes
*termination* arms from *guard* arms.

The helper is called from every termination arm of
``decode_continuation`` and from ``reset()`` to centralise per-stream
counter clearing. The guard arms in ``skip_message`` /
``decode`` that raise ``ContinuationError`` mid-continuation must NOT
call the helper because doing so would clear
``_continuation_expected`` and silently lose the invariant the guard
exists to enforce (the next ``decode_continuation`` call would then
fail to recognise the in-progress stream).
"""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder, MessageEncoder
from dqlitewire.exceptions import ContinuationError
from dqlitewire.messages.responses import RowsResponse


def test_finalize_helper_docstring_distinguishes_termination_from_guard() -> None:
    doc = MessageDecoder._finalize_continuation_state.__doc__ or ""
    assert "termination" in doc.lower()
    assert "guard" in doc.lower()
    assert "skip_message" in doc


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
    """Behaviour pin: ``skip_message`` mid-continuation raises but the
    continuation flag stays True so the caller's correct recovery is
    ``decode_continuation``, not a silently-broken ``decode``."""
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
