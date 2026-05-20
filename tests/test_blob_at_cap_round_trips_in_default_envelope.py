"""A blob at the documented ``_MAX_BLOB_SIZE`` cap must round-trip
through a default ``MessageDecoder`` (no ``max_message_size``
override). Mirror for TEXT cells at ``_MAX_TEXT_VALUE_SIZE``.

The previous fix raised both inner caps to 64 MiB ``"to match"``
``DEFAULT_MAX_MESSAGE_SIZE`` — but the encoder produced 64 MiB +
framing-overhead bytes for a 64 MiB payload, and the same-default
decoder then rejected its own encoder's output with
``DecodeError``. The cap-vs-envelope arithmetic must leave enough
margin for the in-row framing (8 message header + 8 col_count + 8
col_name + 8 row_header + 8 length prefix + 8 row marker = 48
bytes; rounded to 64 for slop) so a payload at the documented cap
round-trips.
"""

from __future__ import annotations

from dqlitewire.buffer import ReadBuffer
from dqlitewire.codec import MessageDecoder, decode_message, encode_message
from dqlitewire.constants import ValueType
from dqlitewire.messages.responses import RowsResponse
from dqlitewire.types import _MAX_BLOB_SIZE, _MAX_TEXT_VALUE_SIZE


def test_max_blob_size_leaves_room_for_in_row_framing_overhead() -> None:
    """Arithmetic pin: ``_MAX_BLOB_SIZE`` must leave at least 48 bytes
    (worst-case in-row framing) below ``DEFAULT_MAX_MESSAGE_SIZE``. If
    the constants drift back to exact equality, an encoder at the cap
    produces wire bytes the same-default decoder rejects."""
    overhead = 48  # 8 hdr + 8 cnt + 8 col + 8 row_hdr + 8 len + 8 marker
    margin = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE - _MAX_BLOB_SIZE
    assert margin >= overhead, (
        f"_MAX_BLOB_SIZE={_MAX_BLOB_SIZE} must leave >= {overhead} bytes "
        f"under DEFAULT_MAX_MESSAGE_SIZE={ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE} "
        f"so encode→decode round-trips at the cap (margin={margin})"
    )


def test_max_text_value_size_leaves_room_for_in_row_framing_overhead() -> None:
    """Mirror for TEXT cells. Same overhead budget applies."""
    overhead = 48
    margin = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE - _MAX_TEXT_VALUE_SIZE
    assert margin >= overhead


def test_blob_at_documented_cap_round_trips_in_default_decoder() -> None:
    """End-to-end: a blob of exactly ``_MAX_BLOB_SIZE`` bytes in a
    single-row ``RowsResponse`` must round-trip through a default
    ``MessageDecoder``. Previously failed with ``DecodeError`` from
    the buffer envelope check."""
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
    """Mirror end-to-end for TEXT cells. Sanity check on the symmetric
    cap; if either cap regresses, both this and the blob twin trip."""
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
    # Use a stateful MessageDecoder.feed/decode path symmetric with
    # production usage, not the convenience decode_message wrapper.
    dec = MessageDecoder()
    dec.feed(wire)
    decoded = dec.decode()
    assert isinstance(decoded, RowsResponse)
    assert decoded.rows[0][0] == text
