"""Pin the oversize-message rejection boundary (ISSUE-105).

``max_message_size`` caps the total envelope size of a single wire
frame. Realistic triggers in production are dynamically-generated SQL:
many-row ``INSERT VALUES (...), (...), ...`` or ``WHERE col IN (?, ?, ...)``
with thousands of placeholders. The encoder must reject at construction
time with a clear error rather than silently truncate or produce bytes
the server will refuse.

Pre-ISSUE-105, no test pinned this boundary. Changes to the envelope
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

    def test_just_over_default_cap_is_rejected(self) -> None:
        """Boundary test against the real 64 MiB default. Construct a
        SQL string whose frame size is slightly over the default cap.
        The test uses a custom buffer at the default size so the
        assertion is explicit."""
        # Each '?' + comma = 2 bytes; padding overhead is small. Aim
        # for ~64 MiB + a margin.
        target_bytes = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE + 1024 * 1024  # 65 MiB
        n_placeholders = target_bytes // 2
        # Build the string in one go to avoid quadratic concatenation.
        big_sql = "SELECT " + ",".join(["?"] * n_placeholders)
        msg = QuerySqlRequest(db_id=0, sql=big_sql)
        encoded = encode_message(msg)
        assert len(encoded) > ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE

        buf = ReadBuffer()  # default cap
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.feed(encoded)
