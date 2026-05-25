"""Pin: ``DescribeRequest.decode_body`` admits non-zero format values
when the caller passes ``strict=False``, mirroring Go's permissive
``DecodeDescribe`` (a bare ``getUint64``).

The default-strict path is preserved: a fresh decode of a non-zero
format frame still raises ``DecodeError`` so production code paths
fail fast against a malformed payload the C gateway would reject
anyway. Proxy / replay / fuzz tools that need to inspect captured
traffic from a buggy peer can opt into the permissive shape.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.messages.requests import DescribeRequest
from dqlitewire.types import encode_uint64


def test_describe_request_default_strict_rejects_non_zero_format() -> None:
    with pytest.raises(DecodeError, match="format must be 0"):
        DescribeRequest.decode_body(encode_uint64(1))


def test_describe_request_strict_false_admits_non_zero_format() -> None:
    msg = DescribeRequest.decode_body(encode_uint64(1), strict=False)
    assert msg.format == 1


def test_describe_request_fresh_construct_non_zero_still_rejected() -> None:
    """Construction-time gate unchanged: the strict=False escape is
    decode-only; outbound emission still fails fast."""
    with pytest.raises(EncodeError, match="format must be 0"):
        DescribeRequest(format=1)


def test_describe_request_zero_format_round_trip_unchanged() -> None:
    msg = DescribeRequest.decode_body(encode_uint64(0))
    assert msg.format == 0
    assert msg.encode_body() == encode_uint64(0)
