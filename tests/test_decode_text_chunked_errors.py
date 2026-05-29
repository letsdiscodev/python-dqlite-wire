"""Error-path tests for the memoryview chunked branch of ``decode_text``
(the one-shot branch already has null-terminator / invalid-UTF-8 tests).
"""

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.types import decode_text


def _align(n: int) -> int:
    """Bytes needed to pad n up to a multiple of 8 (word alignment)."""
    rem = n % 8
    return 0 if rem == 0 else 8 - rem


class TestDecodeTextChunkedErrors:
    def test_chunked_no_null_within_cap_raises_exceeds_maximum(self) -> None:
        """A 70 KiB non-NUL-terminated payload at ``max_size=4 KiB`` raises
        "exceeds maximum" (NOT "not null-terminated") and stops scanning at
        the cap rather than allocating megabytes of chunks first."""
        payload = b"a" * (70 * 1024) + b"\x00" * 8
        buf = memoryview(payload)
        with pytest.raises(DecodeError, match="(?i)exceeds maximum"):
            decode_text(buf, max_size=4 * 1024)

    def test_chunked_null_just_past_cap_raises_exceeds_maximum(self) -> None:
        """A NUL just past the cap surfaces as cap-exceeded, not accepted."""
        # NUL at byte 4097, exactly past max_size=4096.
        payload = b"a" * 4097 + b"\x00" + b"b" * (70 * 1024) + b"\x00" * 8
        buf = memoryview(payload)
        with pytest.raises(DecodeError, match="(?i)exceeds maximum"):
            decode_text(buf, max_size=4096)

    def test_chunked_missing_null_terminator_raises(self) -> None:
        # 70 KiB > 64 KiB threshold, forcing the chunked branch.
        payload = b"a" * (70 * 1024)
        buf = memoryview(payload + b"\x00" * _align(len(payload)))
        with pytest.raises(DecodeError, match="not null-terminated"):
            decode_text(buf)

    def test_chunked_invalid_utf8_before_null_raises(self) -> None:
        # 70 KiB of ASCII + a truncated multi-byte UTF-8 sequence, then null.
        prefix = b"a" * (70 * 1024)
        payload = prefix + b"\xc3" + b"\x00"
        buf = memoryview(payload + b"\x00" * _align(len(payload)))
        with pytest.raises(DecodeError, match="Invalid UTF-8"):
            decode_text(buf)

    def test_chunked_multibyte_codepoint_across_boundary(self) -> None:
        """A 3-byte codepoint straddling the ``_TEXT_SCAN_CHUNK`` boundary
        decodes correctly (the accumulator joins chunks before decoding)."""
        # Place the codepoint one byte before the chunk boundary; total must
        # exceed _TEXT_ONE_SHOT_MAX to force the chunked branch.
        from dqlitewire.types import _TEXT_ONE_SHOT_MAX, _TEXT_SCAN_CHUNK

        filler_len = _TEXT_SCAN_CHUNK - 1
        codepoint = "€".encode()  # 3 bytes: e2 82 ac
        assert len(codepoint) == 3
        tail_padding = b"z" * (_TEXT_ONE_SHOT_MAX + 1 - filler_len - len(codepoint))
        payload = b"a" * filler_len + codepoint + tail_padding + b"\x00"
        buf = memoryview(payload + b"\x00" * _align(len(payload)))

        text, _consumed = decode_text(buf)
        assert text == "a" * filler_len + "€" + "z" * len(tail_padding)
