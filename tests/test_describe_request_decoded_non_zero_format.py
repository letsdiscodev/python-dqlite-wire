"""``DescribeRequest.decode_body`` admits non-zero format only under
``strict=False`` (for proxy/replay/fuzz tools); the default stays strict."""

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
    """The strict=False escape is decode-only; outbound emission still fails."""
    with pytest.raises(EncodeError, match="format must be 0"):
        DescribeRequest(format=1)


def test_describe_request_zero_format_round_trip_unchanged() -> None:
    msg = DescribeRequest.decode_body(encode_uint64(0))
    assert msg.format == 0
    assert msg.encode_body() == encode_uint64(0)
