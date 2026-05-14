"""Pin: ``RowsResponse.decode_body`` rejects trailing bytes after the
DONE / PART marker on the non-zero-column path, in parity with the
zero-column fast path and with sibling decoders
(``LeaderResponse``, ``FailureResponse``, ``ServersResponse``).

A prior alignment cycle established this parity: previously the
zero-column path raised on trailing bytes but the non-zero-column
path returned immediately on the marker, silently consuming any
trailing bytes via the body slice. A byte-replay against the same
stray-trailing pattern produced different results between the two
paths — a strict-decode posture inconsistency now closed.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import ROW_DONE_MARKER, ValueType
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import RowsResponse
from dqlitewire.types import encode_text, encode_uint64, encode_value


def _build_one_row_one_column_body_with_trailer(trailer: bytes) -> bytes:
    """Construct a one-column / one-row RowsResponse body whose marker
    is followed by ``trailer`` bytes."""
    column_count = 1
    body = encode_uint64(column_count)
    # Column name (text + 8-byte alignment built into encode_text).
    body += encode_text("c0")
    # Row header: a single 8-byte word holding the type-nibble for one
    # INTEGER column. encode_value(1) returns (data, type_code).
    value_bytes, type_code = encode_value(1)
    assert type_code == ValueType.INTEGER.value
    # Row type header: byte 0 holds the type for column 0; word-padded.
    row_header = bytes([type_code]) + b"\x00" * 7
    body += row_header
    body += value_bytes
    # End-of-rows marker.
    body += encode_uint64(ROW_DONE_MARKER)
    body += trailer
    return body


def test_rows_response_rejects_trailing_byte_after_done_marker() -> None:
    body = _build_one_row_one_column_body_with_trailer(b"\x00")
    with pytest.raises(DecodeError, match=r"trailing bytes after DONE"):
        RowsResponse.decode_body(body)


def test_rows_response_rejects_trailing_word_after_done_marker() -> None:
    body = _build_one_row_one_column_body_with_trailer(b"\x00" * 8)
    with pytest.raises(DecodeError, match=r"trailing bytes after DONE"):
        RowsResponse.decode_body(body)


def test_rows_response_clean_done_still_decodes() -> None:
    """Regression guard: the strict check does not break the
    no-trailing-bytes happy path."""
    body = _build_one_row_one_column_body_with_trailer(b"")
    response = RowsResponse.decode_body(body)
    assert response.column_names == ["c0"]
    assert response.rows == [[1]]
    assert response.has_more is False
