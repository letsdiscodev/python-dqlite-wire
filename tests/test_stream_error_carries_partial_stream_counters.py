"""Pin: when ``decode_continuation`` raises ``StreamError`` because the
mid-stream frame had a wrong message type, the exception carries the
in-flight partial-stream counters (``frame_count``, ``total_rows``)
so the caller can coordinate a "retry the query" recovery.

Without these counters, the no-poison precedent on the wrong-type arm
gives the caller a resync option but leaves them blind to how much of
the stream was already consumed — they would have to track running
totals externally even though the decoder already knows.
"""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder, MessageEncoder
from dqlitewire.exceptions import StreamError
from dqlitewire.messages.responses import LeaderResponse, RowsResponse


def test_stream_error_default_counters_are_zero() -> None:
    """Backwards-compat: positional-only callers continue to work and
    see zero values for the new fields."""
    err = StreamError("plain message")
    assert err.frame_count == 0
    assert err.total_rows == 0


def test_wrong_type_stream_error_carries_observed_frame_and_row_counts() -> None:
    """Prime 1 initial + 2 continuation frames (10 rows each), then a
    LEADER frame mid-stream. ``StreamError.frame_count`` reports 3
    (one initial + two continuations), ``total_rows`` reports 30.
    The buffer is not poisoned."""
    encoder = MessageEncoder()
    decoder = MessageDecoder(is_request=False)

    initial = RowsResponse(
        column_names=["c0"],
        rows=[[i] for i in range(10)],
        has_more=True,
    )
    cont1 = RowsResponse(column_names=["c0"], rows=[[i] for i in range(10)], has_more=True)
    cont2 = RowsResponse(column_names=["c0"], rows=[[i] for i in range(10)], has_more=True)
    wrong_type = LeaderResponse(node_id=1, address="127.0.0.1:9001")

    decoder.feed(
        encoder.encode(initial)
        + encoder.encode(cont1)
        + encoder.encode(cont2)
        + encoder.encode(wrong_type)
    )
    msg = decoder.decode()
    assert isinstance(msg, RowsResponse)
    # Frame 1 (initial) is recorded inside decode() so the count is 1.
    assert isinstance(decoder.decode_continuation(), RowsResponse)
    assert isinstance(decoder.decode_continuation(), RowsResponse)

    with pytest.raises(StreamError) as excinfo:
        decoder.decode_continuation()
    assert excinfo.value.frame_count == 3
    assert excinfo.value.total_rows == 30
    assert decoder.is_poisoned is False
