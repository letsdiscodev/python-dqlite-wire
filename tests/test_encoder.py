"""Tests for MessageEncoder envelope caps and oversize-message rejection."""

from __future__ import annotations

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.codec import MessageDecoder, MessageEncoder, decode_message, encode_message
from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.limits import MAX_BLOB_SIZE, MAX_TEXT_VALUE_SIZE
from dqlitewire.messages.requests import QuerySqlRequest
from dqlitewire.messages.responses import FilesResponse, RowsResponse

# ---- merged from test_encoder_envelope_cap.py ----
# MessageEncoder must cap total frame size against max_message_size
# symmetrically with the decoder: per-field caps alone let a composite frame
# encode to bytes the matching default-cap decoder then rejects.


def _compose_files(n: int, size: int) -> dict[str, bytes]:
    return {f"f{i:03d}.dat": b"\x00" * size for i in range(n)}


def test_message_encoder_rejects_over_cap_frame() -> None:
    # 80 files at 1 MiB each = ~80 MiB; default 64 MiB cap rejects.
    files = _compose_files(80, 1 << 20)
    enc = MessageEncoder(max_message_size=64 * 1024 * 1024)
    with pytest.raises(EncodeError, match="exceeds maximum"):
        enc.encode(FilesResponse(files=files))


def test_message_encoder_default_cap_matches_default_read_buffer_cap() -> None:
    enc = MessageEncoder()
    frame = enc.encode(FilesResponse(files={"a": b"\x00" * 16}))
    assert len(frame) <= ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE


def test_message_encoder_explicit_loose_cap_accepts_legacy_oversized() -> None:
    enc = MessageEncoder(max_message_size=1 << 30)  # 1 GiB
    files = _compose_files(10, 1 << 20)  # ~10 MiB total
    frame = enc.encode(FilesResponse(files=files))
    dec = MessageDecoder(max_message_size=1 << 30)
    dec.feed(frame)
    decoded = dec.decode()
    assert isinstance(decoded, FilesResponse)


def test_message_encoder_round_trip_at_default_cap_succeeds() -> None:
    enc = MessageEncoder()
    # FilesResponse content must be 8-byte aligned per the dqlite file-entry spec.
    frame = enc.encode(FilesResponse(files={"small.dat": b"\x01\x02\x03\x04\x05\x06\x07\x08"}))
    dec = MessageDecoder()
    dec.feed(frame)
    decoded = dec.decode()
    assert isinstance(decoded, FilesResponse)
    assert decoded.files == {"small.dat": b"\x01\x02\x03\x04\x05\x06\x07\x08"}


def test_message_encoder_rejects_invalid_max_message_size() -> None:
    with pytest.raises(ValueError, match="max_message_size"):
        MessageEncoder(max_message_size=0)
    with pytest.raises(ValueError, match="max_message_size"):
        MessageEncoder(max_message_size=-1)


def test_encode_message_helper_threads_max_message_size() -> None:
    files = _compose_files(80, 1 << 20)
    with pytest.raises(EncodeError, match="exceeds maximum"):
        encode_message(FilesResponse(files=files))
    frame = encode_message(FilesResponse(files=files), max_message_size=1 << 30)
    assert isinstance(frame, bytes)


def test_message_encoder_overflow_diagnostic_shape() -> None:
    files = _compose_files(80, 1 << 20)
    enc = MessageEncoder(max_message_size=64 * 1024 * 1024)
    with pytest.raises(EncodeError) as exc_info:
        enc.encode(FilesResponse(files=files))
    msg = str(exc_info.value)
    assert "exceeds maximum" in msg
    assert str(64 * 1024 * 1024) in msg


# ---- merged from test_blob_at_cap_round_trips_in_default_envelope.py ----
# A blob at the MAX_BLOB_SIZE cap (and text at MAX_TEXT_VALUE_SIZE) must
# round-trip through a default MessageDecoder: the cap sits below
# DEFAULT_MAX_MESSAGE_SIZE by enough to cover in-row framing overhead, so the
# encoder's output is not rejected by its own same-default decoder.


def test_max_blob_size_leaves_room_for_in_row_framing_overhead() -> None:
    """MAX_BLOB_SIZE must leave >= 48 bytes (worst-case in-row framing) below
    DEFAULT_MAX_MESSAGE_SIZE, else an at-cap encode produces bytes the
    same-default decoder rejects."""
    overhead = 48  # 8 hdr + 8 cnt + 8 col + 8 row_hdr + 8 len + 8 marker
    margin = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE - MAX_BLOB_SIZE
    assert margin >= overhead, (
        f"MAX_BLOB_SIZE={MAX_BLOB_SIZE} must leave >= {overhead} bytes "
        f"under DEFAULT_MAX_MESSAGE_SIZE={ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE} "
        f"so encode→decode round-trips at the cap (margin={margin})"
    )


def test_max_text_value_size_leaves_room_for_in_row_framing_overhead() -> None:
    """Mirror for TEXT cells."""
    overhead = 48
    margin = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE - MAX_TEXT_VALUE_SIZE
    assert margin >= overhead


def test_blob_at_documented_cap_round_trips_in_default_decoder() -> None:
    """End-to-end: an at-cap blob in a single-row RowsResponse round-trips
    through a default MessageDecoder."""
    blob = b"x" * MAX_BLOB_SIZE
    resp = RowsResponse(
        column_names=["data"],
        column_types=[ValueType.BLOB],
        row_types=[[ValueType.BLOB]],
        rows=[[blob]],
        has_more=False,
    )
    wire = encode_message(resp)
    assert len(wire) <= ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE, (
        f"encoded message {len(wire)} bytes overflows default envelope "
        f"{ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE}; cap math is off"
    )
    decoded = decode_message(wire)
    assert isinstance(decoded, RowsResponse)
    assert decoded.rows[0][0] == blob


def test_text_at_documented_cap_round_trips_in_default_decoder() -> None:
    """Mirror end-to-end for TEXT cells."""
    text = "x" * MAX_TEXT_VALUE_SIZE
    resp = RowsResponse(
        column_names=["data"],
        column_types=[ValueType.TEXT],
        row_types=[[ValueType.TEXT]],
        rows=[[text]],
        has_more=False,
    )
    wire = encode_message(resp)
    assert len(wire) <= ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE
    # Stateful feed/decode path (production usage), not decode_message.
    dec = MessageDecoder()
    dec.feed(wire)
    decoded = dec.decode()
    assert isinstance(decoded, RowsResponse)
    assert decoded.rows[0][0] == text


# ---- merged from test_oversize_message.py ----
# Pin the oversize-message rejection boundary.
#
# ``max_message_size`` caps the total envelope size of a single wire
# frame. Realistic triggers in production are dynamically-generated SQL:
# many-row ``INSERT VALUES (...), (...), ...`` or ``WHERE col IN (?, ?, ...)``
# with thousands of placeholders. The encoder must reject at construction
# time with a clear error rather than silently truncate or produce bytes
# the server will refuse.
#
# Previously no test pinned this boundary. Changes to the envelope
# cap behavior would go undetected.


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
        """SQL text fields are capped on encode at
        ``MAX_TEXT_VALUE_SIZE`` (the same cap the decoder applies),
        so an oversize SQL is rejected at construction time rather
        than producing bytes the decoder will refuse. Mirrors the
        cap-symmetry pattern applied to other text fields."""
        from dqlitewire.exceptions import EncodeError
        from dqlitewire.limits import MAX_TEXT_VALUE_SIZE

        oversize = "x" * (MAX_TEXT_VALUE_SIZE + 1)
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
