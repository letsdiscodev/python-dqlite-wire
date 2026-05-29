"""``FailureResponse.decode_body`` recovers the genuine error message when
the server appends the failure record after an un-rewound partial rows
header.

When a row-returning statement raises *during stepping* (e.g. integer
overflow on ``SELECT abs(-9223372036854775808)``), the dqlite server has
already written the rows-response header (column count + column names)
into its send buffer; it then appends the real failure record without
rewinding, and frames the whole thing as one ``RESPONSE_FAILURE`` body:

    [ col_count ][ col_name_1 .. col_name_N ][ real code ][ real message ]

The genuine diagnostic is the trailing ``(code, message)`` record; the
leading region is a stale rows header. Previously the decoder rejected
any trailing bytes outright, surfacing a misleading "trailing bytes"
``DecodeError`` instead of the real reason (e.g. "integer overflow").
The decoder now recovers the trailing record. Bodies that do not match
this shape fall back to the first record (matching the reference Go
client, which reads one record and ignores the rest); genuinely
truncated/too-short bodies still raise.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import FailureResponse
from dqlitewire.types import encode_text, encode_uint64


def _stacked_body(column_names: list[str], code: int, message: str) -> bytes:
    """Build the server's un-rewound body: a partial rows header (column
    count + names) followed by the appended failure record."""
    body = encode_uint64(len(column_names))
    for name in column_names:
        body += encode_text(name)
    body += encode_uint64(code)
    body += encode_text(message)
    return body


def test_single_column_partial_header_recovers_real_message() -> None:
    # The captured live-cluster shape for SELECT abs(-9223372036854775808):
    # col_count=1, column name = the offending expression, then the real
    # failure record (code=1, "integer overflow").
    body = _stacked_body(["abs(-9223372036854775808)"], 1, "integer overflow")
    decoded = FailureResponse.decode_body(body)
    assert decoded.code == 1
    assert decoded.message == "integer overflow"


def test_multi_column_partial_header_recovers_real_message() -> None:
    body = _stacked_body(["one", "two", "bad"], 19, "integer overflow")
    decoded = FailureResponse.decode_body(body)
    assert decoded.code == 19
    assert decoded.message == "integer overflow"


def test_clean_single_record_body_unchanged() -> None:
    body = encode_uint64(5) + encode_text("checkpoint in progress")
    decoded = FailureResponse.decode_body(body)
    assert decoded.code == 5
    assert decoded.message == "checkpoint in progress"


def test_too_short_body_still_raises() -> None:
    with pytest.raises(DecodeError):
        FailureResponse.decode_body(b"\x00" * 8)
