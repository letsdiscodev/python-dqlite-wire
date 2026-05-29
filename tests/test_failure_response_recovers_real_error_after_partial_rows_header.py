"""FailureResponse.decode_body recovers the real (code, message) when the
server frames a failure after an un-rewound partial rows header:

    [ col_count ][ col_name_1 .. col_name_N ][ real code ][ real message ]

Non-matching bodies fall back to the first record (matching the Go client);
truncated bodies still raise."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import FailureResponse
from dqlitewire.types import encode_text, encode_uint64


def _stacked_body(column_names: list[str], code: int, message: str) -> bytes:
    """Un-rewound body: partial rows header (count + names) then failure record."""
    body = encode_uint64(len(column_names))
    for name in column_names:
        body += encode_text(name)
    body += encode_uint64(code)
    body += encode_text(message)
    return body


def test_single_column_partial_header_recovers_real_message() -> None:
    # Captured live-cluster shape for SELECT abs(-9223372036854775808).
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
