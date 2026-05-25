"""Pin: ``MessageDecoder`` accepts a ``text_errors`` knob and threads
it through to per-row-cell TEXT / ISO8601 decoding.

A single non-UTF-8 TEXT cell (e.g. a legacy SQLite store with latin-1
columns) used to fail the entire ``RowsResponse`` decode under the
strict default. The knob lets a caller opt into permissive decoding
(``"replace"``, ``"backslashreplace"``, etc.) and matches Go's
``getString`` shape that just returns raw bytes.
"""

from __future__ import annotations

import struct

import pytest

from dqlitewire.codec import MessageDecoder, MessageEncoder
from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import RowsResponse
from dqlitewire.tuples import decode_row_values
from dqlitewire.types import decode_value, encode_text, encode_uint64


def _bad_utf8_text_block() -> bytes:
    # latin-1 "café" with stray 0xff prefix; not valid UTF-8.
    raw = b"\xffcaf\xe9\x00"
    pad = (-len(raw)) % 8
    return raw + b"\x00" * pad


def test_decode_value_default_strict_rejects_bad_utf8() -> None:
    with pytest.raises(DecodeError, match="Invalid UTF-8"):
        decode_value(_bad_utf8_text_block(), ValueType.TEXT)


def test_decode_value_text_errors_replace_admits_bad_utf8() -> None:
    value, _ = decode_value(_bad_utf8_text_block(), ValueType.TEXT, text_errors="replace")
    assert isinstance(value, str)
    assert "caf" in value
    # Replacement char must appear at the bad bytes.
    assert "�" in value


def test_decode_row_values_text_errors_threaded() -> None:
    values, _ = decode_row_values(_bad_utf8_text_block(), [ValueType.TEXT], text_errors="replace")
    assert values == [values[0]]
    assert "�" in str(values[0])


def _rows_response_body_with_bad_utf8_text() -> bytes:
    """Build a single-column, single-row RowsResponse body whose only
    TEXT cell carries bytes that are not valid UTF-8."""
    body = encode_uint64(1)  # column_count
    body += encode_text("col0")
    # Row header: single TEXT cell — one 4-bit nibble per cell,
    # padded to word boundary (8 bytes).
    body += bytes([int(ValueType.TEXT)]) + b"\x00" * 7
    body += _bad_utf8_text_block()
    # Row DONE marker.
    body += struct.pack("<Q", 0xFFFFFFFFFFFFFFFF)
    return body


def test_rows_response_strict_default_rejects_bad_utf8() -> None:
    with pytest.raises(DecodeError, match="Invalid UTF-8"):
        RowsResponse.decode_body(_rows_response_body_with_bad_utf8_text())


def test_rows_response_text_errors_replace_admits_bad_utf8() -> None:
    rsp = RowsResponse.decode_body(_rows_response_body_with_bad_utf8_text(), text_errors="replace")
    assert len(rsp.rows) == 1
    assert "�" in str(rsp.rows[0][0])


def test_message_decoder_text_errors_propagates_to_row_cells() -> None:
    decoder = MessageDecoder(is_request=False, text_errors="replace")
    # Build a Rows wire frame containing the bad-utf8 row.
    encoder = MessageEncoder()
    valid = RowsResponse(
        column_names=["c0"],
        column_types=[ValueType.TEXT],
        rows=[["ok"]],
        has_more=False,
    )
    wire = encoder.encode(valid)
    # Splice the bad-utf8 body in place of the valid one by re-using the
    # full wire pipeline: feed and decode.
    bad_body = _rows_response_body_with_bad_utf8_text()
    # Construct the header manually so we don't need a custom encoder.
    from dqlitewire.constants import ResponseType
    from dqlitewire.messages.base import Header

    body_size_words = len(bad_body) // 8
    header = Header(size_words=body_size_words, msg_type=ResponseType.ROWS, schema=0)
    bad_wire = header.encode() + bad_body

    decoder.feed(bad_wire)
    decoded = decoder.decode()
    assert isinstance(decoded, RowsResponse)
    assert "�" in str(decoded.rows[0][0])

    # Also keep the valid round-trip working through the same decoder.
    decoder2 = MessageDecoder(is_request=False, text_errors="replace")
    decoder2.feed(wire)
    assert isinstance(decoder2.decode(), RowsResponse)


def test_message_decoder_rejects_unknown_text_errors_value() -> None:
    with pytest.raises(ValueError, match="text_errors"):
        MessageDecoder(is_request=False, text_errors="bogus-value")
