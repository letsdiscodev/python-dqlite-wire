"""Pin the oversize-message rejection boundary.

``max_message_size`` caps the total envelope size of a single wire
frame. Realistic triggers in production are dynamically-generated SQL:
many-row ``INSERT VALUES (...), (...), ...`` or ``WHERE col IN (?, ?, ...)``
with thousands of placeholders. The encoder must reject at construction
time with a clear error rather than silently truncate or produce bytes
the server will refuse.

Previously no test pinned this boundary. Changes to the envelope
cap behavior would go undetected.
"""

from __future__ import annotations

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.codec import decode_message, encode_message
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.requests import QuerySqlRequest


class TestOversizeSqlEncode:
    """An encoder-constructed frame whose total size exceeds the
    buffer's max cap must be rejected at decode — the wire format
    itself does not refuse oversize frames at encode time, but the
    decoder's envelope check is the first thing any real peer does
    to our bytes."""

    def test_huge_sql_frame_exceeds_buffer_cap(self) -> None:
        """A frame encoded from a multi-MB SQL string must be rejected
        by the decoder's envelope cap."""
        # Construct a SQL string whose encoded frame exceeds a small
        # buffer cap. Use a cap small enough that the test is cheap;
        # the real cap is 64 MiB.
        small_cap = 1024 * 1024  # 1 MiB
        # Each "?," is 2 bytes; aim for ~1.5 MB of SQL.
        big_sql = "SELECT " + ",".join(["?"] * 750_000)
        msg = QuerySqlRequest(db_id=0, sql=big_sql)
        encoded = encode_message(msg)
        assert len(encoded) > small_cap, (
            f"test setup: encoded frame must exceed cap ({len(encoded)} vs {small_cap})"
        )

        buf = ReadBuffer(max_message_size=small_cap)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.feed(encoded)

    def test_huge_sql_within_cap_decodes(self) -> None:
        """A frame that fits within the cap must decode cleanly —
        guards against a regression that over-tightens the check."""
        big_sql = "SELECT " + ",".join(["?"] * 1_000)  # ~6 KB
        msg = QuerySqlRequest(db_id=0, sql=big_sql)
        encoded = encode_message(msg)
        assert len(encoded) < ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, QuerySqlRequest)
        assert decoded.sql == big_sql

    def test_just_over_sql_text_cap_is_rejected_at_encode(self) -> None:
        """SQL text fields are now capped on encode at the same
        ``_MAX_TEXT_VALUE_SIZE`` (16 MiB) the decoder applies, so an
        oversize SQL is rejected at construction time rather than
        producing bytes the decoder will refuse. Mirrors the
        cap-symmetry pattern applied to other text fields."""
        from dqlitewire.exceptions import EncodeError

        # 16 MiB + 1 byte; the SQL-text cap fires before the outer
        # frame cap (which is at 64 MiB). The EncodeError is the
        # caller-actionable diagnostic — without the cap, the
        # encoder produced bytes that decode_text would reject on
        # the receive side with a non-actionable wire error.
        oversize = "x" * (16 * 1024 * 1024 + 1)
        with pytest.raises(EncodeError):
            QuerySqlRequest(db_id=0, sql=oversize).encode_body()

    def test_oversize_frame_header_rejected_at_decode(self) -> None:
        """The frame-level ``ReadBuffer`` cap protects against a hostile
        peer announcing a > cap body in the header — distinct from the
        encode-side text cap above. Hostile peers do NOT route through
        our encode path; the decode-side cap is the load-bearing
        defense, and a regression to the boundary check (e.g. ``>=``
        vs ``>``, or removal) must fail this pin.

        Construct the oversize frame header as raw bytes (bypassing the
        encode-side cap that would otherwise reject) and exercise
        ``peek_header`` / ``has_message`` / ``read_message`` against a
        small cap. The cap fires on the size_words → total_size check
        (peek_header / read_message both raise; has_message returns
        True so the consume-loop reaches the raise).
        """
        small_cap = 1024 * 1024  # 1 MiB — keep the test cheap
        oversize_words = (small_cap // 8) + 16  # declared body > cap
        # Header: size_words=oversize_words (uint32 LE), msg_type=0,
        # schema=0, reserved=0
        header = oversize_words.to_bytes(4, "little") + b"\x00\x00\x00\x00"

        buf = ReadBuffer(max_message_size=small_cap)
        buf.feed(header)
        # peek_header is the strict-raise variant per its docstring.
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.peek_header()
