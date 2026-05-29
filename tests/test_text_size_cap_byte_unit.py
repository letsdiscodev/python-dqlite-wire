"""encode_text and decode_text cap on UTF-8 BYTES, not codepoints, so
round-trip identity holds for content where byte length != codepoint length."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.responses import (
    _MAX_ADDRESS_SIZE,
    _MAX_FAILURE_MESSAGE_SIZE,
    FailureResponse,
    LeaderResponse,
)
from dqlitewire.types import decode_text, encode_text


class TestEncodeTextMaxSizeBytes:
    def test_max_size_caps_on_bytes_not_codepoints(self) -> None:
        # 😀 is 4 bytes; 32 codepoints = 128 bytes. cap=130 admits, cap=120 rejects.
        text = "😀" * 32
        encode_text(text, max_size=130)
        with pytest.raises(EncodeError, match="length 128 exceeds maximum"):
            encode_text(text, max_size=120)

    def test_default_no_cap(self) -> None:
        # Without max_size, there is no per-call cap.
        encode_text("a" * 100_000)

    def test_label_in_error(self) -> None:
        with pytest.raises(EncodeError, match="leader address length"):
            encode_text("a" * 16, max_size=8, label="leader address")


class TestRoundTripIdentityAtCap:
    def test_failure_message_round_trip_at_byte_cap(self) -> None:
        text = "😀" * (_MAX_FAILURE_MESSAGE_SIZE // 4)  # 4 bytes each
        body = FailureResponse(code=1, message=text).encode_body()
        decoded = FailureResponse.decode_body(body)
        assert decoded.message == text

    def test_failure_message_rejected_above_byte_cap(self) -> None:
        text = "😀" * ((_MAX_FAILURE_MESSAGE_SIZE // 4) + 1)
        with pytest.raises(EncodeError, match="(?i)failure message"):
            FailureResponse(code=1, message=text).encode_body()

    def test_leader_address_round_trip_at_byte_cap(self) -> None:
        text = "ä" * (_MAX_ADDRESS_SIZE // 2)  # 2-byte UTF-8 codepoint
        body = LeaderResponse(node_id=1, address=text).encode_body()
        decoded = LeaderResponse.decode_body(body)
        assert decoded.address == text


class TestDecodeTextCapBeforeAllocate:
    def test_cap_bound_materialisation_window(self) -> None:
        # NUL well past max_size: decoder must cap-error without materialising
        # the whole buffer (we can only observe the error shape, not allocation).
        too_long = b"a" * 100 + b"\x00"
        view = memoryview(too_long)
        with pytest.raises(Exception, match="exceeds maximum"):
            decode_text(view, max_size=10)
