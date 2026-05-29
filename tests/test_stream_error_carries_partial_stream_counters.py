"""A wrong-type StreamError from decode_continuation carries the in-flight
partial-stream counters (frame_count, total_rows) so the caller can recover."""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder, MessageEncoder
from dqlitewire.exceptions import StreamError
from dqlitewire.messages.responses import LeaderResponse, RowsResponse


def test_stream_error_default_counters_are_zero() -> None:
    """Backwards-compat: plain callers see zero for the new fields."""
    err = StreamError("plain message")
    assert err.frame_count == 0
    assert err.total_rows == 0


def test_wrong_type_stream_error_carries_observed_frame_and_row_counts() -> None:
    """3 frames x 10 rows then a mid-stream LEADER: StreamError reports
    frame_count=3, total_rows=30, and the buffer is not poisoned."""
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
    # The initial frame is counted inside decode().
    assert isinstance(decoder.decode_continuation(), RowsResponse)
    assert isinstance(decoder.decode_continuation(), RowsResponse)

    with pytest.raises(StreamError) as excinfo:
        decoder.decode_continuation()
    assert excinfo.value.frame_count == 3
    assert excinfo.value.total_rows == 30
    assert decoder.is_poisoned is False
