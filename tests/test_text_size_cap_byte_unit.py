"""Pin: ``encode_text`` and ``decode_text`` cap on UTF-8 BYTES, not
codepoints.

Decode-side caps have always counted bytes (``decode_text`` walks the
buffer until NUL and compares ``null_pos`` to ``max_size``). Encode-
side caps used to count codepoints (``len(s)``) at every call site,
so the encoder admitted non-ASCII payloads near the boundary that
the decoder rejected — round-trip identity broke for any content
where ``len(s.encode("utf-8")) != len(s)``.

The fix routes the cap through ``encode_text(value, max_size=...)``
so the units match. This also closes the cap-after-allocate ordering
in ``decode_text`` (the materialisation now bounds itself to
``max_size + 1`` bytes) and in ``encode_value`` BLOB
(``len(value)`` is checked before ``bytes(value)`` materialises).
"""

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
        # 4-byte UTF-8 sequence (😀 = U+1F600) takes 4 bytes per
        # codepoint. 32 codepoints = 128 bytes. cap=130 admits it.
        # cap=120 must reject it because byte length exceeds cap,
        # even though codepoint length (32) is well below.
        text = "😀" * 32
        encode_text(text, max_size=130)  # ok
        with pytest.raises(EncodeError, match="length 128 exceeds maximum"):
            encode_text(text, max_size=120)

    def test_default_no_cap(self) -> None:
        # Without max_size, encode_text matches its historical
        # contract — no per-call cap.
        encode_text("a" * 100_000)

    def test_label_in_error(self) -> None:
        with pytest.raises(EncodeError, match="leader address length"):
            encode_text("a" * 16, max_size=8, label="leader address")


class TestRoundTripIdentityAtCap:
    def test_failure_message_round_trip_at_byte_cap(self) -> None:
        # 4-byte UTF-8 codepoints at the byte boundary.
        text = "😀" * (_MAX_FAILURE_MESSAGE_SIZE // 4)
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
        # Build a buffer where the NUL is well past max_size. The
        # decoder must surface a cap-exceeded error WITHOUT
        # materialising the entire buffer — but we cannot directly
        # observe allocation, so we just confirm the error shape.
        too_long = b"a" * 100 + b"\x00"
        view = memoryview(too_long)
        with pytest.raises(Exception, match="exceeds maximum"):
            decode_text(view, max_size=10)
