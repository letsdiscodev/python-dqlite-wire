"""Pin: ROWS continuation frames must keep the initial frame's column_count;
drift silently truncates results or fabricates NULLs, so the decoder raises
DecodeError."""

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
    decoder = MessageDecoder(is_request=False)
    initial_bytes = _build_initial_frame(column_count=3, has_more=True, decoder=decoder)
    decoder.feed(initial_bytes)
    initial = decoder.decode()
    assert isinstance(initial, RowsResponse)
    assert decoder._continuation_column_count == 3

    cont_bytes = _build_initial_frame(column_count=2, has_more=False, decoder=decoder)
    decoder.feed(cont_bytes)
    with pytest.raises(DecodeError, match="column count drift"):
        decoder.decode_continuation()


def test_continuation_column_count_match_passes() -> None:
    decoder = MessageDecoder(is_request=False)
    initial_bytes = _build_initial_frame(column_count=3, has_more=True, decoder=decoder)
    decoder.feed(initial_bytes)
    decoder.decode()

    cont_bytes = _build_initial_frame(column_count=3, has_more=False, decoder=decoder)
    decoder.feed(cont_bytes)
    cont = decoder.decode_continuation()
    assert cont is not None
    # State cleared after final frame.
    assert decoder._continuation_column_count is None
