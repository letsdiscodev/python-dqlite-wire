"""Pin: ROWS continuation frames must have the same column_count as
the initial frame.

A peer that drops to ``column_count=0`` mid-stream silently
truncates the result; a peer that grows ``column_count`` produces
phantom NULLs. Either is a corrupt-stream signal that the decoder
must surface as ``DecodeError``.
"""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import RowsResponse


def _build_initial_frame(column_count: int, has_more: bool, decoder: MessageDecoder) -> bytes:
    rows = RowsResponse(
        column_names=[f"c{i}" for i in range(column_count)],
        rows=[[i for i in range(column_count)]],
        has_more=has_more,
    )
    from dqlitewire.codec import MessageEncoder

    enc = MessageEncoder()
    return enc.encode(rows)


def test_continuation_column_count_drift_raises() -> None:
    """A continuation frame with a different column_count from the
    initial frame must raise DecodeError."""
    decoder = MessageDecoder(is_request=False)
    # Bypass handshake for response decoder (handshake_done=True per
    # is_request=False)
    initial_bytes = _build_initial_frame(column_count=3, has_more=True, decoder=decoder)
    decoder.feed(initial_bytes)
    initial = decoder.decode()
    assert isinstance(initial, RowsResponse)
    assert decoder._continuation_column_count == 3

    # Continuation frame with a different (smaller) column_count.
    cont_bytes = _build_initial_frame(column_count=2, has_more=False, decoder=decoder)
    decoder.feed(cont_bytes)
    with pytest.raises(DecodeError, match="column count drift"):
        decoder.decode_continuation()


def test_continuation_column_count_match_passes() -> None:
    """A continuation frame with the same column_count as the initial
    frame is accepted normally."""
    decoder = MessageDecoder(is_request=False)
    initial_bytes = _build_initial_frame(column_count=3, has_more=True, decoder=decoder)
    decoder.feed(initial_bytes)
    decoder.decode()

    # Continuation frame with same column_count, no more.
    cont_bytes = _build_initial_frame(column_count=3, has_more=False, decoder=decoder)
    decoder.feed(cont_bytes)
    cont = decoder.decode_continuation()
    assert cont is not None
    # State cleared after final frame.
    assert decoder._continuation_column_count is None
