"""A blob at the _MAX_BLOB_SIZE cap (and text at _MAX_TEXT_VALUE_SIZE) must
round-trip through a default MessageDecoder: the cap sits below
DEFAULT_MAX_MESSAGE_SIZE by enough to cover in-row framing overhead, so the
encoder's output is not rejected by its own same-default decoder."""

from __future__ import annotations

from dqlitewire.buffer import ReadBuffer
from dqlitewire.codec import MessageDecoder, decode_message, encode_message
from dqlitewire.constants import ValueType
from dqlitewire.messages.responses import RowsResponse
from dqlitewire.types import _MAX_BLOB_SIZE, _MAX_TEXT_VALUE_SIZE


def test_max_blob_size_leaves_room_for_in_row_framing_overhead() -> None:
    """_MAX_BLOB_SIZE must leave >= 48 bytes (worst-case in-row framing) below
    DEFAULT_MAX_MESSAGE_SIZE, else an at-cap encode produces bytes the
    same-default decoder rejects."""
    overhead = 48  # 8 hdr + 8 cnt + 8 col + 8 row_hdr + 8 len + 8 marker
    margin = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE - _MAX_BLOB_SIZE
    assert margin >= overhead, (
        f"_MAX_BLOB_SIZE={_MAX_BLOB_SIZE} must leave >= {overhead} bytes "
        f"under DEFAULT_MAX_MESSAGE_SIZE={ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE} "
        f"so encode→decode round-trips at the cap (margin={margin})"
    )


def test_max_text_value_size_leaves_room_for_in_row_framing_overhead() -> None:
    """Mirror for TEXT cells."""
    overhead = 48
    margin = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE - _MAX_TEXT_VALUE_SIZE
    assert margin >= overhead


def test_blob_at_documented_cap_round_trips_in_default_decoder() -> None:
    """End-to-end: an at-cap blob in a single-row RowsResponse round-trips
    through a default MessageDecoder."""
    blob = b"x" * _MAX_BLOB_SIZE
    resp = RowsResponse(
        column_names=["data"],
        column_types=[ValueType.BLOB],
        row_types=[[ValueType.BLOB]],
        rows=[[blob]],
        has_more=False,
    )
    wire = encode_message(resp)
    assert len(wire) <= ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE, (
        f"encoded message {len(wire)} bytes overflows default envelope "
        f"{ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE}; cap math is off"
    )
    decoded = decode_message(wire)
    assert isinstance(decoded, RowsResponse)
    assert decoded.rows[0][0] == blob


def test_text_at_documented_cap_round_trips_in_default_decoder() -> None:
    """Mirror end-to-end for TEXT cells."""
    text = "x" * _MAX_TEXT_VALUE_SIZE
    resp = RowsResponse(
        column_names=["data"],
        column_types=[ValueType.TEXT],
        row_types=[[ValueType.TEXT]],
        rows=[[text]],
        has_more=False,
    )
    wire = encode_message(resp)
    assert len(wire) <= ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE
    # Stateful feed/decode path (production usage), not decode_message.
    dec = MessageDecoder()
    dec.feed(wire)
    decoded = dec.decode()
    assert isinstance(decoded, RowsResponse)
    assert decoded.rows[0][0] == text
