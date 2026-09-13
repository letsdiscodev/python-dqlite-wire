"""Tests for response message encoding/decoding."""

from __future__ import annotations

import logging
import struct
from typing import Any
from unittest.mock import patch

import pytest

from dqlitewire.codec import RESPONSE_TYPES, MessageDecoder
from dqlitewire.constants import (
    DQLITE_NOTFOUND,
    DQLITE_PARSE,
    HEADER_SIZE,
    ROW_DONE_MARKER,
    SQLITE_IOERR_LEADERSHIP_LOST,
    SQLITE_IOERR_NOT_LEADER,
    NodeRole,
    ResponseType,
    ValueType,
)
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.limits import (
    MAX_ADDRESS_SIZE,
    MAX_COLUMN_COUNT,
    MAX_COLUMN_NAME_SIZE,
    MAX_FAILURE_MESSAGE_SIZE,
    MAX_FILENAME_SIZE,
    MAX_NODE_COUNT,
    MAX_PARAM_COUNT,
    MAX_TAIL_OFFSET,
)
from dqlitewire.messages.base import Header
from dqlitewire.messages.responses import (
    DbResponse,
    EmptyResponse,
    FailureResponse,
    FilesResponse,
    LeaderResponse,
    MetadataResponse,
    NodeInfo,
    ResultResponse,
    RowsResponse,
    ServersResponse,
    StmtResponse,
    WelcomeResponse,
)
from dqlitewire.types import WireValue, encode_text, encode_uint64, encode_value


class TestFailureResponse:
    def test_encode(self) -> None:
        msg = FailureResponse(code=1, message="error")
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == ResponseType.FAILURE

    def test_roundtrip(self) -> None:
        msg = FailureResponse(code=42, message="Something went wrong")
        encoded = msg.encode()
        decoded = FailureResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.code == 42
        assert decoded.message == "Something went wrong"

    def test_empty_message(self) -> None:
        msg = FailureResponse(code=1, message="")
        encoded = msg.encode()
        decoded = FailureResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.code == 1
        assert decoded.message == ""

    def test_construct_with_code_zero_accepted(self) -> None:
        """Upstream's gateway emits ``failure(req, 0, "empty statement")``
        from ``handle_prepare_done_cb`` (gateway.c:372) and
        ``handle_query_sql_done_cb`` (gateway.c:890) when the SQL parses
        to no statement (empty / comment-only). The C source carries an
        explicit ``/* FIXME Should we use a code other than 0 here? */``
        next to the emit. Pin acceptance so a conforming server's
        empty-statement reply does not poison the decoder."""
        msg = FailureResponse(code=0, message="empty statement")
        assert msg.code == 0
        assert msg.message == "empty statement"

    def test_decode_accepts_code_zero(self) -> None:
        """Symmetric: bytes with code=0 round-trip cleanly through the
        decoder, surfacing as a normal ``OperationalError(0, "empty
        statement")`` to the user rather than a poisoned-buffer
        ProtocolError."""
        body = encode_uint64(0) + encode_text("empty statement")
        decoded = FailureResponse.decode_body(body)
        assert decoded.code == 0
        assert decoded.message == "empty statement"

    def test_round_trip_code_zero_through_message_decoder(self) -> None:
        """End-to-end pin: ``encode_message`` → ``MessageDecoder.feed`` →
        ``decode`` for ``FailureResponse(code=0)``. This is the path a
        real client takes when the server replies to an empty SQL
        prepare; the prior contract-level rejection broke this path."""
        from dqlitewire.codec import MessageDecoder

        msg = FailureResponse(code=0, message="empty statement")
        encoded = msg.encode()
        decoder = MessageDecoder(is_request=False)
        decoder.feed(encoded)
        decoded = decoder.decode()
        assert isinstance(decoded, FailureResponse)
        assert decoded.code == 0
        assert decoded.message == "empty statement"
        assert not decoder.is_poisoned

    @pytest.mark.parametrize("code", [4, 9, 10, 11, 13])
    def test_round_trip_auto_rollback_codes(self, code: int) -> None:
        """Pin round-trip for the primary SQLite codes that the downstream
        client interprets as server-side auto-rollback. A future codec
        refactor that mishandled one of these would otherwise surface only
        in an integration test."""
        msg = FailureResponse(code=code, message=f"primary code {code}")
        encoded = msg.encode_body()
        decoded = FailureResponse.decode_body(encoded)
        assert decoded.code == code
        assert decoded.message == f"primary code {code}"

    def test_decode_rejects_oversize_message(self) -> None:
        """A peer claiming a multi-megabyte error message can force a large
        allocation and full-string scan through sanitize_server_text. Cap
        the decoded message at MAX_FAILURE_MESSAGE_SIZE so the decoder
        fails fast before the sanitize scan runs."""
        oversize = "a" * (MAX_FAILURE_MESSAGE_SIZE + 1)
        body = encode_uint64(1) + encode_text(oversize)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            FailureResponse.decode_body(body)

    def test_decode_accepts_message_at_cap(self) -> None:
        """Exactly-cap message must still decode."""
        at_cap = "a" * MAX_FAILURE_MESSAGE_SIZE
        body = encode_uint64(1) + encode_text(at_cap)
        decoded = FailureResponse.decode_body(body)
        assert decoded.code == 1
        assert len(decoded.message) == MAX_FAILURE_MESSAGE_SIZE

    def test_encode_rejects_oversize_message(self) -> None:
        """Encoder mirrors the decode cap so callers fail fast on an
        accidentally-huge message string."""
        msg = FailureResponse(code=1, message="a" * (MAX_FAILURE_MESSAGE_SIZE + 1))
        with pytest.raises(EncodeError, match="exceeds maximum"):
            msg.encode_body()

    def test_decode_too_short_diagnostic_mentions_conforming_minimum(self) -> None:
        """The 9-byte gate's diagnostic must reference the realistic
        16-byte conforming-peer minimum (8 code + 8 padded empty text
        block per BytePad64), not just the gate's own 9-byte
        threshold. A maintainer reading the message should immediately
        understand what a real peer would emit and why this gate
        catches only the degenerate truncated case."""
        with pytest.raises(DecodeError) as exc_info:
            FailureResponse.decode_body(b"\x00" * 8)
        msg = str(exc_info.value)
        assert "16" in msg
        assert "padded" in msg


class TestLeaderResponse:
    def test_roundtrip(self) -> None:
        msg = LeaderResponse(node_id=1, address="192.168.1.1:9001")
        encoded = msg.encode()
        decoded = LeaderResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.node_id == 1
        assert decoded.address == "192.168.1.1:9001"

    def test_empty_address(self) -> None:
        msg = LeaderResponse(node_id=0, address="")
        encoded = msg.encode()
        decoded = LeaderResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.node_id == 0
        assert decoded.address == ""

    def test_decode_body_legacy(self) -> None:
        """Pre-1.0 servers send only a text address (no node_id) for type code 1.

        Go has DecodeNodeLegacy for this. decode_body_legacy() handles the
        legacy format, returning node_id=0.
        """
        from dqlitewire.types import encode_text

        legacy_body = encode_text("192.168.1.1:9001")
        decoded = LeaderResponse.decode_body_legacy(legacy_body)
        assert decoded.node_id == 0
        assert decoded.address == "192.168.1.1:9001"

    def test_legacy_decode_encode_byte_identical(self) -> None:
        """Pin: a captured legacy body round-trips byte-identically
        through ``decode_body_legacy → encode_body_legacy``. The
        encoder is a niche compatibility primitive whose entire
        purpose is reproducing the legacy wire shape exactly; without
        a captured-fixture pin, a future refactor (e.g. sharing
        helper code with the modern encoder, changing pad
        calculation) could silently break legacy-server compat and
        no test would catch it.
        """
        from dqlitewire.types import encode_text

        captured_legacy = encode_text("10.0.0.1:9001")
        decoded = LeaderResponse.decode_body_legacy(captured_legacy)
        assert decoded.encode_body_legacy() == captured_legacy


class TestLeaderResponseAddressSize:
    """Per-address length cap applies to modern and legacy decoders.

    Legitimate addresses are short (hostname + port, or IPv6 literal
    in brackets + port). Cap at ``MAX_ADDRESS_SIZE`` so an oversize
    peer-supplied string cannot amplify through logs / exception
    messages even after sanitization.
    """

    def test_decode_rejects_oversize_address(self) -> None:
        oversize = "a" * (MAX_ADDRESS_SIZE + 1)
        body = encode_uint64(1) + encode_text(oversize)
        with pytest.raises(DecodeError, match="leader address"):
            LeaderResponse.decode_body(body)

    def test_decode_accepts_address_at_cap(self) -> None:
        at_cap = "a" * MAX_ADDRESS_SIZE
        body = encode_uint64(1) + encode_text(at_cap)
        decoded = LeaderResponse.decode_body(body)
        assert decoded.address == at_cap

    def test_decode_body_legacy_rejects_oversize_address(self) -> None:
        oversize = "a" * (MAX_ADDRESS_SIZE + 1)
        body = encode_text(oversize)
        with pytest.raises(DecodeError, match="leader address"):
            LeaderResponse.decode_body_legacy(body)


class TestWelcomeResponse:
    def test_roundtrip(self) -> None:
        msg = WelcomeResponse(heartbeat_timeout=15000)
        encoded = msg.encode()
        decoded = WelcomeResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.heartbeat_timeout == 15000

    @pytest.mark.parametrize("heartbeat", [0, 1, 2**64 - 1])
    def test_roundtrip_heartbeat_boundaries(self, heartbeat: int) -> None:
        """heartbeat_timeout is uint64 on the wire. Existing tests
        already cover 10_000 / 15_000 / 15_000_000_000 (larger than
        uint32); this pins the uint64 extremes so a narrowing refactor
        would surface here, not in a heartbeat-tied latency regression
        far downstream."""
        msg = WelcomeResponse(heartbeat_timeout=heartbeat)
        encoded = msg.encode()
        decoded = WelcomeResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.heartbeat_timeout == heartbeat

    def test_heartbeat_timeout_seconds_property(self) -> None:
        """``heartbeat_timeout_seconds`` must expose the server's value
        as seconds (milliseconds / 1000). Pin the conversion so a caller
        wiring the field directly into ``asyncio.wait_for`` picks up the
        seconds-based property rather than the raw milliseconds."""
        assert WelcomeResponse(heartbeat_timeout=15000).heartbeat_timeout_seconds == 15.0
        assert WelcomeResponse(heartbeat_timeout=0).heartbeat_timeout_seconds == 0.0
        assert WelcomeResponse(heartbeat_timeout=250).heartbeat_timeout_seconds == 0.25


class TestDbResponse:
    def test_roundtrip(self) -> None:
        msg = DbResponse(db_id=1)
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == ResponseType.DB
        decoded = DbResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 1


class TestStmtResponse:
    def test_roundtrip(self) -> None:
        msg = StmtResponse(db_id=1, stmt_id=5, num_params=3)
        encoded = msg.encode()
        decoded = StmtResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 1
        assert decoded.stmt_id == 5
        assert decoded.num_params == 3

    def test_body_size(self) -> None:
        """StmtResponse body should be exactly 16 bytes per Go reference.

        Format: uint32 db_id + uint32 stmt_id + uint64 num_params = 16 bytes.
        """
        msg = StmtResponse(db_id=1, stmt_id=2, num_params=3)
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.body_size == 16

    def test_roundtrip_v1_with_tail_offset(self) -> None:
        """V1 STMT response includes tail_offset for multi-statement prepare.

        Note: tail_offset is not present in the canonical Go client
        (go-dqlite). It may be supported by the C server but the Go
        client's EncodePrepare always uses schema=0 and DecodeStmt
        never reads tail_offset.
        """
        msg = StmtResponse(db_id=1, stmt_id=5, num_params=3, tail_offset=42)
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.body_size == 24  # 16 + 8 for tail_offset
        decoded = StmtResponse.decode_body(encoded[HEADER_SIZE:], schema=header.schema)
        assert decoded.db_id == 1
        assert decoded.stmt_id == 5
        assert decoded.num_params == 3
        assert decoded.tail_offset == 42

    def test_v0_has_no_tail_offset(self) -> None:
        """V0 STMT response has no tail_offset."""
        msg = StmtResponse(db_id=1, stmt_id=5, num_params=3)
        encoded = msg.encode()
        decoded = StmtResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.tail_offset is None

    def test_schema_0_rejects_short_body(self) -> None:
        """235: Schema=0 body must be exactly 16 bytes."""
        with pytest.raises(DecodeError, match=r"schema=0 body must be exactly 16 bytes"):
            StmtResponse.decode_body(b"\x00" * 8, schema=0)

    def test_schema_1_rejects_v0_length_body(self) -> None:
        """235: Schema=1 demands 24 bytes.

        A 16-byte body under schema=1 was previously accepted silently and
        returned ``tail_offset=None``, indistinguishable from a real V0
        body. That collapsed two different protocol states into one.
        """
        v0_body = b"\x01\x00\x00\x00" + b"\x02\x00\x00\x00" + b"\x03" + b"\x00" * 7
        assert len(v0_body) == 16
        with pytest.raises(DecodeError, match=r"schema=1 body must be exactly 24 bytes"):
            StmtResponse.decode_body(v0_body, schema=1)

    def test_decode_preserves_schema_v1_zero_tail(self) -> None:
        """A V1 body with ``tail_offset=0`` must round-trip as V1 (24-byte
        body), not V0 (16-byte body). The decoder therefore stores the
        incoming header schema byte on the dataclass so ``_get_schema``
        prefers it over the ``tail_offset is not None`` auto-select."""
        import struct

        body = (
            struct.pack("<I", 1)  # db_id
            + struct.pack("<I", 2)  # stmt_id
            + struct.pack("<Q", 3)  # num_params
            + struct.pack("<Q", 0)  # tail_offset = 0 (valid V1)
        )
        assert len(body) == 24
        decoded = StmtResponse.decode_body(body, schema=1)
        assert decoded.tail_offset == 0
        assert decoded.schema == 1
        assert decoded._get_schema() == 1
        # Re-encoding produces a V1 body (24 B), not V0 (16 B).
        reencoded = decoded.encode_body()
        assert len(reencoded) == 24

    def test_decode_schema_0_leaves_schema_attribute_set_to_zero(self) -> None:
        """V0 path stores ``schema=0`` on the dataclass so the field is
        always meaningful after decode (no ``None`` ambiguity)."""
        import struct

        body = struct.pack("<I", 1) + struct.pack("<I", 2) + struct.pack("<Q", 3)
        assert len(body) == 16
        decoded = StmtResponse.decode_body(body, schema=0)
        assert decoded.schema == 0
        assert decoded.tail_offset is None
        assert len(decoded.encode_body()) == 16

    def test_rejects_oversized_num_params(self) -> None:
        """Defense-in-depth: cap server-declared num_params to match the
        encoder-side MAX_PARAM_COUNT. A malicious or corrupt server
        returning num_params=2**63-1 produces a clean DecodeError, not
        an unchecked value that a cautious caller could trust.
        """
        # Build a schema=0 body with num_params = 2**63 - 1.
        import struct

        body = (
            struct.pack("<I", 1)  # db_id
            + struct.pack("<I", 2)  # stmt_id
            + struct.pack("<Q", 2**63 - 1)  # num_params (bogus)
        )
        with pytest.raises(DecodeError, match="num_params"):
            StmtResponse.decode_body(body, schema=0)

    def test_rejects_oversized_tail_offset(self) -> None:
        """Defense-in-depth: a schema=1 server emitting ``tail_offset``
        above ``MAX_TAIL_OFFSET`` could make Python's ``sql[offset:]``
        silently return ``""`` and drop trailing statements. Mirror the
        encoder-side cap with a decode-side DecodeError so the attack
        surface is closed at both boundaries.
        """
        import struct

        from dqlitewire.limits import MAX_TAIL_OFFSET

        body = (
            struct.pack("<I", 1)  # db_id
            + struct.pack("<I", 2)  # stmt_id
            + struct.pack("<Q", 3)  # num_params
            + struct.pack("<Q", MAX_TAIL_OFFSET + 1)  # tail_offset (bogus)
        )
        assert len(body) == 24
        with pytest.raises(DecodeError, match="tail_offset"):
            StmtResponse.decode_body(body, schema=1)

    def test_accepts_multi_mib_tail_offset(self) -> None:
        """A 4 MiB ``tail_offset`` from a multi-statement prepare must
        decode cleanly. The cap is now aligned with the message envelope
        (64 MiB) so any tail offset that could legitimately appear in a
        prepare response, bounded by the server's message envelope per
        ``gateway.c:410``, decodes without false-rejecting on the
        Python side. Pre-bump the cap was 1 MiB and this test would
        have raised."""
        import struct

        offset = 4 * 1024 * 1024
        body = (
            struct.pack("<I", 1)  # db_id
            + struct.pack("<I", 2)  # stmt_id
            + struct.pack("<Q", 3)  # num_params
            + struct.pack("<Q", offset)  # tail_offset (4 MiB)
        )
        decoded = StmtResponse.decode_body(body, schema=1)
        assert decoded.tail_offset == offset


class TestResultResponse:
    def test_roundtrip(self) -> None:
        msg = ResultResponse(last_insert_id=42, rows_affected=5)
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == ResponseType.RESULT
        decoded = ResultResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.last_insert_id == 42
        assert decoded.rows_affected == 5

    def test_zero_values(self) -> None:
        msg = ResultResponse(last_insert_id=0, rows_affected=0)
        encoded = msg.encode()
        decoded = ResultResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.last_insert_id == 0
        assert decoded.rows_affected == 0

    def test_decodes_uint64_max_last_insert_id(self) -> None:
        """``last_insert_id`` is wire uint64 and round-trips at the
        unsigned-max boundary without overflow or sign coercion.
        ``rows_affected`` has its own INT_MAX cap (the C server's
        ``sqlite3_changes`` cannot exceed it) and is tested at the
        boundary in ``test_result_response_caps.py``."""
        import struct

        # ``rows_affected`` capped at INT_MAX; pair it with the
        # boundary value rather than uint64-max.
        int_max = (1 << 31) - 1
        body = struct.pack("<QQ", 0xFFFFFFFFFFFFFFFF, int_max)
        decoded = ResultResponse.decode_body(body)
        assert decoded.last_insert_id == 0xFFFFFFFFFFFFFFFF
        assert decoded.rows_affected == int_max

    def test_decodes_signed_negative_rowid_as_unsigned(self) -> None:
        """The C server gateway casts sqlite3_last_insert_rowid()
        (signed int64) through (uint64_t). A negative SQLite rowid
        ``-5`` arrives as ``0xFFFFFFFFFFFFFFFB``. Pin the asymmetry —
        the decoder MUST surface the unsigned form, not coerce to
        Python's signed int."""
        import struct

        body = struct.pack("<QQ", 0xFFFFFFFFFFFFFFFB, 0)
        decoded = ResultResponse.decode_body(body)
        assert decoded.last_insert_id == 0xFFFFFFFFFFFFFFFB
        # The unsigned form is NOT -5 — that conversion belongs to a
        # downstream layer that wants stdlib sqlite3 parity, not to the
        # wire decoder.
        assert decoded.last_insert_id != -5

    def test_decodes_signed_min_int64_boundary(self) -> None:
        """The signed-int64 ``MIN`` value (``-2**63``) round-trips
        through ``(uint64_t)`` as ``0x8000000000000000``."""
        import struct

        body = struct.pack("<QQ", 0x8000000000000000, 0)
        decoded = ResultResponse.decode_body(body)
        assert decoded.last_insert_id == 0x8000000000000000


class TestRowsResponseAliasing:
    """Regression tests for caller-list aliasing in RowsResponse.

    ``RowsResponse`` used to store caller-supplied ``column_names`` and
    ``column_types`` lists by reference, creating silent aliasing:

    ``column_types`` was stored as ``all_row_types[0]`` inside
    ``decode_body`` via ``if not column_types: column_types = types``.
    The returned object then had ``r.column_types is r.row_types[0]``,
    so mutating one list silently rewrote the other.

    The fix copies both lists on construction via ``__post_init__``.
    """

    def test_constructor_copies_row_types_and_rows(self) -> None:
        """Caller-supplied lists must be independent from the message
        after construction. Applies uniformly to column_names,
        column_types, row_types (outer + inner), and rows (outer + inner).
        """
        supplied_row_types = [[ValueType.INTEGER, ValueType.TEXT]]
        supplied_rows: list[list[WireValue]] = [[1, "alice"]]
        msg = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            row_types=supplied_row_types,
            rows=supplied_rows,
            has_more=False,
        )

        # Outer-list copies: mutating the supplied list does not affect
        # the message.
        supplied_row_types.append([ValueType.NULL])
        supplied_rows.append([99, "mallory"])
        assert len(msg.row_types) == 1
        assert len(msg.rows) == 1

        # Inner-list copies: mutating the supplied inner list does not
        # affect the message's inner copies.
        supplied_row_types[0][0] = ValueType.NULL
        supplied_rows[0][0] = 999
        assert msg.row_types[0][0] == ValueType.INTEGER
        assert msg.rows[0][0] == 1

    def test_column_types_is_not_aliased_to_row_types_first(self) -> None:
        """After decoding, ``column_types`` must be a distinct list
        object from ``row_types[0]`` so mutation of one does not
        propagate to the other.
        """
        msg = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[[1, "alice"], [2, "bob"]],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])

        assert decoded.row_types  # sanity: rows were decoded with types
        assert decoded.column_types is not decoded.row_types[0], (
            "column_types must not share identity with row_types[0]"
        )

        # Prove independence by mutation.
        decoded.column_types[0] = ValueType.NULL
        assert decoded.row_types[0][0] != ValueType.NULL

    def test_column_names_defensive_copy_on_construct(self) -> None:
        """A caller who supplies ``column_names`` to RowsResponse and
        later mutates the source list must not see the change
        reflected in the returned object.
        """
        names_in = ["a", "b", "c"]
        msg = RowsResponse(
            column_names=names_in,
            column_types=[ValueType.INTEGER, ValueType.TEXT, ValueType.BLOB],
            row_types=[],
            rows=[],
            has_more=False,
        )

        names_in.append("d")

        assert msg.column_names == ["a", "b", "c"], (
            "RowsResponse.column_names must not alias the caller's list"
        )

    def test_column_types_defensive_copy_on_construct(self) -> None:
        """Same for column_types passed into the constructor."""
        types_in = [ValueType.INTEGER, ValueType.TEXT]
        msg = RowsResponse(
            column_names=["a", "b"],
            column_types=types_in,
            row_types=[],
            rows=[],
            has_more=False,
        )

        types_in[0] = ValueType.BLOB

        assert msg.column_types[0] == ValueType.INTEGER, (
            "RowsResponse.column_types must not alias the caller's list"
        )

    def test_get_row_types_does_not_alias_column_types(self) -> None:
        """Regression: _get_row_types must not alias column_types.

        ``__post_init__`` defensively copies ``column_types`` so that
        mutation of a caller-supplied or decoder-supplied list cannot
        rewrite the ``RowsResponse``'s private copy. ``_get_row_types``
        used to return ``self.column_types`` by reference on the
        "no per-row types" fallback path, which silently punched a
        hole in that invariant: a caller who captured the return
        value and mutated it would mutate the message's private
        list.

        This test asserts that the returned list is a distinct object
        from ``self.column_types`` and that mutating the return value
        does not affect the message.
        """
        msg = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            row_types=[],
            rows=[[1]],
            has_more=False,
        )

        row_types = msg._get_row_types(0, [1])

        assert row_types is not msg.column_types, (
            "_get_row_types must return a fresh list, not alias column_types"
        )

        row_types.append(ValueType.BLOB)
        assert msg.column_types == [ValueType.INTEGER], (
            "mutating the _get_row_types return value must not affect "
            "msg.column_types (no-aliasing invariant)"
        )

    def test_get_row_types_does_not_alias_row_types(self) -> None:
        """Regression: _get_row_types must not alias row_types[i].

        The ``row_types`` branch of ``_get_row_types`` used to return
        ``self.row_types[row_idx]`` by reference — the same aliasing
        pattern already fixed for the ``column_types`` fallback.
        A caller who captured the return value and mutated it would
        silently rewrite the message's internal ``row_types[i]`` list.
        """
        msg = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[1]],
            has_more=False,
        )

        row_types = msg._get_row_types(0, [1])

        assert row_types is not msg.row_types[0], (
            "_get_row_types must return a fresh list, not alias row_types[i]"
        )

        row_types.append(ValueType.BLOB)
        assert msg.row_types[0] == [ValueType.INTEGER], (
            "mutating the _get_row_types return value must not affect "
            "msg.row_types[i] (no-aliasing invariant)"
        )

    def test_decode_body_bypasses_post_init_deep_copy(self) -> None:
        """``decode_body`` builds fresh ``rows`` / ``row_types`` lists by
        construction and must NOT pay the constructor's per-row defensive
        deep-copy on top — that cost scales with row count and runs on
        the loop thread for every decoded frame.

        The decoder uses a private ``_from_decoded`` bypass classmethod
        that skips ``__post_init__``. Verify the bypass actually fires
        for both the row-bearing path and the zero-column fast path so
        a future refactor cannot silently re-introduce the duplicate
        copy.
        """
        from unittest import mock

        # Row-bearing decode path.
        msg = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[[1, "alice"], [2, "bob"]],
            has_more=False,
        )
        encoded = msg.encode()

        with mock.patch.object(RowsResponse, "__post_init__", autospec=True) as post_init:
            decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])

        assert post_init.call_count == 0, (
            "decode_body must bypass __post_init__ to skip the per-row "
            f"defensive deep-copy; got {post_init.call_count} call(s)"
        )
        # Behaviour must still be correct: alias break holds, row data
        # decoded as expected.
        assert decoded.column_types is not decoded.row_types[0]
        assert decoded.rows[0][0] == 1
        assert decoded.rows[1][1] == "bob"

        # Zero-column fast path also bypasses.
        empty = RowsResponse(
            column_names=[],
            column_types=[],
            rows=[],
            has_more=False,
        )
        encoded_empty = empty.encode()
        with mock.patch.object(RowsResponse, "__post_init__", autospec=True) as post_init_empty:
            RowsResponse.decode_body(encoded_empty[HEADER_SIZE:])
        assert post_init_empty.call_count == 0, (
            "zero-column decode path must also bypass __post_init__"
        )


class TestRowsResponse:
    def test_empty_result(self) -> None:
        msg = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.column_names == ["id", "name"]
        assert decoded.rows == []
        assert decoded.has_more is False

    def test_single_row(self) -> None:
        msg = RowsResponse(
            column_names=["id"],
            column_types=[ValueType.INTEGER],
            rows=[[42]],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.column_names == ["id"]
        assert len(decoded.rows) == 1
        assert decoded.rows[0][0] == 42

    def test_multiple_rows(self) -> None:
        msg = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[[1, "Alice"], [2, "Bob"], [3, "Charlie"]],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert len(decoded.rows) == 3
        assert decoded.rows[0] == [1, "Alice"]
        assert decoded.rows[1] == [2, "Bob"]
        assert decoded.rows[2] == [3, "Charlie"]

    def test_has_more(self) -> None:
        msg = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            has_more=True,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.has_more is True

    def test_heterogeneous_types_roundtrip(self) -> None:
        """Rows with different types per column must encode/decode correctly.

        SQLite allows different storage types for the same column across rows
        (type affinity). Each row has its own type header in the wire format.
        """
        msg = RowsResponse(
            column_names=["x"],
            row_types=[
                [ValueType.INTEGER],
                [ValueType.TEXT],
                [ValueType.NULL],
            ],
            rows=[[42], ["hello"], [None]],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.rows[0] == [42]
        assert decoded.rows[1] == ["hello"]
        assert decoded.rows[2] == [None]
        assert decoded.row_types[0] == [ValueType.INTEGER]
        assert decoded.row_types[1] == [ValueType.TEXT]
        assert decoded.row_types[2] == [ValueType.NULL]

    def test_column_types_populated_on_decode(self) -> None:
        """column_types should be populated from the first row's types after decode."""
        msg = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[[1, "Alice"], [2, "Bob"]],
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.column_types == [ValueType.INTEGER, ValueType.TEXT]

    def test_column_types_reflects_first_row_only(self) -> None:
        """column_types reflects first row's types; use row_types for per-row types."""
        msg = RowsResponse(
            column_names=["x"],
            row_types=[
                [ValueType.INTEGER],
                [ValueType.TEXT],
                [ValueType.NULL],
            ],
            rows=[[42], ["hello"], [None]],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        # column_types reflects first row only
        assert decoded.column_types == [ValueType.INTEGER]
        # row_types has accurate per-row types
        assert decoded.row_types[0] == [ValueType.INTEGER]
        assert decoded.row_types[1] == [ValueType.TEXT]
        assert decoded.row_types[2] == [ValueType.NULL]

    def test_zero_columns_with_done_marker(self) -> None:
        """Zero-column result with DONE marker should decode to empty rows."""
        from dqlitewire.types import encode_uint64

        # column_count=0, followed by DONE marker
        data = encode_uint64(0) + encode_uint64(0xFFFFFFFFFFFFFFFF)
        decoded = RowsResponse.decode_body(data)
        assert decoded.column_names == []
        assert decoded.rows == []
        assert decoded.has_more is False

    def test_zero_columns_with_part_marker(self) -> None:
        """Zero-column result with PART marker should decode with has_more=True."""
        from dqlitewire.types import encode_uint64

        # column_count=0, followed by PART marker
        data = encode_uint64(0) + encode_uint64(0xEEEEEEEEEEEEEEEE)
        decoded = RowsResponse.decode_body(data)
        assert decoded.column_names == []
        assert decoded.rows == []
        assert decoded.has_more is True

    def test_zero_columns_roundtrip(self) -> None:
        """Zero-column RowsResponse should roundtrip correctly."""
        msg = RowsResponse(column_names=[], rows=[], has_more=False)
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.column_names == []
        assert decoded.rows == []
        assert decoded.has_more is False

    def test_zero_columns_malformed_no_infinite_loop(self) -> None:
        """Zero-column result with non-marker data must raise a clear error."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # column_count=0, followed by non-marker bytes
        data = encode_uint64(0) + b"\x01\x02\x03\x04\x05\x06\x07\x08"
        with pytest.raises(DecodeError, match="Expected DONE or PART marker"):
            RowsResponse.decode_body(data)

    def test_zero_columns_missing_marker_raises(self) -> None:
        """Zero-column result with no end marker should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # column_count=0, but no marker follows
        data = encode_uint64(0)
        with pytest.raises(DecodeError, match="end marker"):
            RowsResponse.decode_body(data)

    def test_zero_columns_rejects_torn_done_marker(self) -> None:
        """Zero-column result with only the first byte of DONE must be
        rejected — the non-zero-column path rejects the same torn shape
        via decode_row_header, and both paths should be symmetric."""
        from dqlitewire.types import encode_uint64

        # column_count=0, followed by 0xff + 7 zero bytes (torn marker).
        data = encode_uint64(0) + b"\xff" + b"\x00" * 7
        with pytest.raises(DecodeError, match="marker"):
            RowsResponse.decode_body(data)

    def test_zero_columns_rejects_torn_part_marker(self) -> None:
        """Zero-column result with only the first byte of PART must be rejected."""
        from dqlitewire.types import encode_uint64

        data = encode_uint64(0) + b"\xee" + b"\x00" * 7
        with pytest.raises(DecodeError, match="marker"):
            RowsResponse.decode_body(data)

    def test_decode_body_rejects_non_list_row_header(self) -> None:
        """decode_body must raise DecodeError if decode_row_header returns unexpected type."""
        from unittest.mock import patch

        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        # Build a valid body
        body = encode_uint64(1)  # column_count=1
        body += encode_text("id")
        body += encode_row_header([ValueType.INTEGER])
        body += encode_row_values([42], [ValueType.INTEGER])
        body += encode_uint64(0xFFFFFFFFFFFFFFFF)  # DONE

        # Patch decode_row_header to return a string (simulating unexpected type)
        with (
            patch("dqlitewire.messages.responses.decode_row_header", return_value=("bad", 8)),
            pytest.raises(DecodeError, match="Expected column types list"),
        ):
            RowsResponse.decode_body(body)

    def test_truncated_body_without_marker_raises(self) -> None:
        """Body exhausted without DONE/PART marker must raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        # Build a body with rows but NO end marker
        body = encode_uint64(1)  # column_count=1
        body += encode_text("id")
        body += encode_row_header([ValueType.INTEGER])
        body += encode_row_values([42], [ValueType.INTEGER])
        # No marker at the end!

        with pytest.raises(DecodeError, match="end marker"):
            RowsResponse.decode_body(body)

    def test_bogus_column_count_raises(self) -> None:
        """A column_count larger than remaining data should raise DecodeError early."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # column_count = 1 billion, but only 8 bytes of data after count
        body = encode_uint64(1_000_000_000) + b"\x00" * 8
        with pytest.raises(DecodeError, match="exceeds maximum"):
            RowsResponse.decode_body(body)

    def test_column_count_exceeds_hard_limit(self) -> None:
        """Column count exceeding the hard limit should raise DecodeError.

        Cap is now SQLite's documented maximum (32767 per
        https://www.sqlite.org/limits.html); anything above is
        provably malformed."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # 32_768 columns — one above SQLite's hard cap.
        # Provide enough data bytes to bypass the early "Not enough
        # data" guard so we reach the cap check.
        body = encode_uint64(32_768) + b"\x00" * (32_768 * 8 + 8)
        with pytest.raises(DecodeError, match="(?i)column count.*exceeds maximum"):
            RowsResponse.decode_body(body)

    def test_max_rows_limit_decode_body(self) -> None:
        """decode_body should reject messages exceeding the max_rows limit."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        # Build a body with 5 rows
        body = encode_uint64(1)  # column_count=1
        body += encode_text("x")
        for i in range(5):
            body += encode_row_header([ValueType.INTEGER])
            body += encode_row_values([i], [ValueType.INTEGER])
        body += encode_uint64(0xFFFFFFFFFFFFFFFF)  # DONE

        # Should succeed with default limit
        decoded = RowsResponse.decode_body(body)
        assert len(decoded.rows) == 5

        # Should fail with max_rows=3
        with pytest.raises(DecodeError, match="Row count.*reached limit"):
            RowsResponse.decode_body(body, max_rows=3)

    def test_max_rows_exact_boundary_rejects_at_limit(self) -> None:
        """max_rows=3 with exactly 3 rows should raise DecodeError.

        The max_rows parameter is a strict upper bound: at most max_rows - 1
        rows should be decoded without error. When the number of rows
        reaches max_rows, the limit has been exceeded and DecodeError
        should fire immediately — without decoding another row first.
        """
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        def build_body(n_rows: int) -> bytes:
            body = encode_uint64(1)  # column_count=1
            body += encode_text("x")
            for i in range(n_rows):
                body += encode_row_header([ValueType.INTEGER])
                body += encode_row_values([i], [ValueType.INTEGER])
            body += encode_uint64(0xFFFFFFFFFFFFFFFF)  # DONE
            return body

        # Exactly max_rows rows should raise — message must say "reached",
        # not "exceeds", because len(rows) == max_rows
        with pytest.raises(DecodeError, match="reached limit"):
            RowsResponse.decode_body(build_body(3), max_rows=3)

        # One fewer than max_rows should succeed
        decoded = RowsResponse.decode_body(build_body(2), max_rows=3)
        assert len(decoded.rows) == 2


class TestRowsResponseColumnCountBounds:
    """The pre-loop column-count bound must reserve 8 bytes for the
    mandatory row-end marker (DONE/PART). A ``column_count`` that
    consumes every remaining byte would otherwise proceed into the row
    decode loop and only fail there with a less-specific error.
    """

    def test_column_count_leaves_room_for_end_marker(self) -> None:
        """With remaining=16 (two 8-byte words), column_count=2 would
        eat every byte and leave none for the marker. Reject at the
        pre-loop bound with a marker-aware diagnostic."""
        from dqlitewire.types import encode_uint64

        # 8 bytes of column_count + 16 bytes of remaining room = 24 total.
        # Two empty column names would consume those 16 bytes, leaving
        # zero for the 8-byte end marker.
        body = encode_uint64(2) + (b"\x00" * 16)
        with pytest.raises(DecodeError, match="row end marker"):
            RowsResponse.decode_body(body)

    def test_column_count_exactly_fills_names_and_marker(self) -> None:
        """With remaining=24 (three 8-byte words), column_count=2 leaves
        exactly 8 bytes for the marker. Must NOT be rejected by the
        pre-loop bound (the loop body handles the valid path).
        """
        # Build a well-formed frame with 2 empty column names and a
        # DONE marker so the rest of the decode succeeds.
        from dqlitewire.types import encode_uint64

        done_marker = b"\xff" * 8  # ROW_DONE_MARKER as 8 little-endian bytes
        body = encode_uint64(2) + (b"\x00" * 16) + done_marker
        resp = RowsResponse.decode_body(body)
        assert resp.column_names == ["", ""]
        assert not resp.has_more


class TestRowsResponseEncodeInference:
    """Pin the docstring's type-inference contract: when both ``row_types``
    and ``column_types`` are empty, per-cell types are inferred from
    Python types, which cannot distinguish dqlite-specific encodings from
    their primitive siblings. Regression fence — a future 'helpful' encoder
    that quietly promotes int to UNIXTIME would break golden-byte tests
    silently."""

    def test_int_infers_integer_not_unixtime(self) -> None:
        from dqlitewire.constants import ValueType

        resp = RowsResponse(column_names=["ts"], rows=[[1700000000]])
        encoded = resp.encode_body()
        # Decode and inspect the row_types — per-row type nibble is INTEGER, not UNIXTIME.
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.column_types == [ValueType.INTEGER]

    def test_bool_infers_boolean_not_integer(self) -> None:
        """Sanity: ``bool`` is a special case with its own ValueType,
        inferred correctly (no ambiguity with the int case above)."""
        from dqlitewire.constants import ValueType

        resp = RowsResponse(column_names=["flag"], rows=[[True]])
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.column_types == [ValueType.BOOLEAN]


class TestRowsResponseEncodeValidation:
    """Encode-time validation of column count consistency."""

    def test_mismatched_column_types_length(self) -> None:
        resp = RowsResponse(
            column_names=["a", "b", "c"],
            column_types=[ValueType.INTEGER],
            rows=[],
        )
        with pytest.raises(EncodeError, match="column_types length"):
            resp.encode_body()

    def test_mismatched_row_values_length(self) -> None:
        resp = RowsResponse(
            column_names=["a", "b"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            row_types=[[ValueType.INTEGER, ValueType.TEXT]],
            rows=[[1]],  # only 1 value, expected 2
        )
        with pytest.raises(EncodeError, match="Row 0 has 1 values, expected 2"):
            resp.encode_body()

    def test_mismatched_row_types_length(self) -> None:
        resp = RowsResponse(
            column_names=["a", "b"],
            row_types=[[ValueType.INTEGER]],  # only 1 type, expected 2
            rows=[[1, 2]],
        )
        with pytest.raises(EncodeError, match="row_types\\[0\\] has 1 types, expected 2"):
            resp.encode_body()

    def test_partial_row_types_rejected(self) -> None:
        """row_types must be empty or exactly match rows one-to-one."""
        resp = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],  # only one entry, 3 rows
            rows=[[1], [2], [3]],
        )
        with pytest.raises(EncodeError, match="row_types length .* != rows length"):
            resp.encode_body()

    def test_empty_row_types_with_rows_infers_ok(self) -> None:
        """Empty row_types is valid — types are inferred per-row."""
        resp = RowsResponse(
            column_names=["a"],
            rows=[[1], [2], [3]],
        )
        resp.encode_body()

    def test_full_row_types_matches_rows_ok(self) -> None:
        resp = RowsResponse(
            column_names=["a"],
            row_types=[[ValueType.INTEGER], [ValueType.INTEGER], [ValueType.INTEGER]],
            rows=[[1], [2], [3]],
        )
        resp.encode_body()

    def test_empty_column_types_with_rows_ok(self) -> None:
        """Empty column_types is valid — types are inferred from values."""
        resp = RowsResponse(
            column_names=["a"],
            rows=[[42]],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.rows == [[42]]

    def test_zero_columns_with_rows_raises(self) -> None:
        """Zero-column rows are nonsensical: each row emits 0 bytes, so
        the encoded message is indistinguishable from a zero-row result
        set. Reject at encode time so the encoder is symmetric with the
        decoder's zero-column fast path (it returns no rows).
        """
        resp = RowsResponse(
            column_names=[],
            column_types=[],
            row_types=[],
            rows=[[]],
        )
        with pytest.raises(EncodeError, match=r"zero columns.*1 empty row"):
            resp.encode_body()


class TestRowsResponseColumnNameSize:
    """Per-column-name length cap in RowsResponse decode.

    The outer 64 MiB frame cap is the ultimate backstop, but a peer can
    still pack a single giant column name inside a frame-legal response
    and force the client to allocate it as a Python string. Cap each
    column name at ``MAX_COLUMN_NAME_SIZE`` (same policy as
    ``MAX_FAILURE_MESSAGE_SIZE``).
    """

    def _build_body(self, name: str) -> bytes:
        # One-column RowsResponse, no rows, DONE marker.
        from dqlitewire.constants import ROW_DONE_MARKER

        return encode_uint64(1) + encode_text(name) + encode_uint64(ROW_DONE_MARKER)

    def test_decode_rejects_oversize_column_name(self) -> None:
        oversize = "a" * (MAX_COLUMN_NAME_SIZE + 1)
        body = self._build_body(oversize)
        with pytest.raises(DecodeError, match="column name"):
            RowsResponse.decode_body(body)

    def test_decode_accepts_column_name_at_cap(self) -> None:
        at_cap = "a" * MAX_COLUMN_NAME_SIZE
        body = self._build_body(at_cap)
        decoded = RowsResponse.decode_body(body)
        assert decoded.column_names == [at_cap]


class TestRowsResponseNullInTypedColumn:
    """137: None values in rows with explicit column types must encode correctly."""

    def test_none_with_explicit_column_types(self) -> None:
        """None in a TEXT column must encode as NULL, not crash."""
        resp = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            row_types=[[ValueType.INTEGER, ValueType.TEXT]],
            rows=[[1, None]],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.rows == [[1, None]]
        # The type nibble for the None column should be NULL (5), not TEXT (3)
        assert decoded.row_types[0][1] == ValueType.NULL

    def test_none_with_explicit_row_types_integer(self) -> None:
        """None in an INTEGER column with explicit row_types."""
        resp = RowsResponse(
            column_names=["val"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[None]],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.rows == [[None]]
        assert decoded.row_types[0][0] == ValueType.NULL

    def test_mixed_none_and_values(self) -> None:
        """Row with mix of None and real values."""
        resp = RowsResponse(
            column_names=["a", "b", "c"],
            column_types=[ValueType.INTEGER, ValueType.TEXT, ValueType.FLOAT],
            row_types=[[ValueType.INTEGER, ValueType.TEXT, ValueType.FLOAT]],
            rows=[[None, "hello", None]],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.rows == [[None, "hello", None]]
        assert decoded.row_types[0][0] == ValueType.NULL
        assert decoded.row_types[0][1] == ValueType.TEXT
        assert decoded.row_types[0][2] == ValueType.NULL

    def test_round_trip_first_row_null_clobbers_column_type(self) -> None:
        """When row 0 contains NULL in a column declared as INTEGER, the
        decoded ``column_types`` reflects NULL (not the schema-declared
        INTEGER), because ``column_types`` is captured from the first
        decoded row's per-row type tag and the encoder writes NULL for
        None values to match Go-parity. This pins the documented caveat
        — consumers needing the schema type should not rely on
        ``column_types[X]`` when row 0 may carry NULL."""
        resp = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[None]],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        # column_types now reflects NULL, not the originally-declared INTEGER.
        assert decoded.column_types == [ValueType.NULL]


class TestRowsResponseWordBoundary:
    """115: row header padding crosses word boundary at 17+ columns."""

    def test_16_columns_exact_word(self) -> None:
        """16 columns = 8 nibble bytes = exactly 1 word, no padding."""
        n = 16
        types = [ValueType.INTEGER] * n
        names = [f"c{i}" for i in range(n)]
        values: list[WireValue] = list(range(n))
        resp = RowsResponse(
            column_names=names,
            column_types=types,
            row_types=[types],
            rows=[values],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.column_names == names
        assert decoded.rows == [values]
        assert decoded.row_types[0] == types

    def test_17_columns_crosses_word_boundary(self) -> None:
        """17 columns = 9 nibble bytes, padded to 16 = 2 words."""
        n = 17
        types = [ValueType.INTEGER] * n
        names = [f"c{i}" for i in range(n)]
        values: list[WireValue] = list(range(n))
        resp = RowsResponse(
            column_names=names,
            column_types=types,
            row_types=[types],
            rows=[values],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.column_names == names
        assert decoded.rows == [values]
        assert decoded.row_types[0] == types

    def test_33_columns_third_word_boundary(self) -> None:
        """33 columns = 17 nibble bytes, padded to 24 = 3 words."""
        n = 33
        types = [ValueType.INTEGER] * n
        names = [f"c{i}" for i in range(n)]
        values: list[WireValue] = list(range(n))
        resp = RowsResponse(
            column_names=names,
            column_types=types,
            row_types=[types],
            rows=[values],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.column_names == names
        assert decoded.rows == [values]
        assert decoded.row_types[0] == types


class TestRowsResponseValueTypes:
    """Full RowsResponse round-trips with BOOLEAN, UNIXTIME, ISO8601, and BLOB."""

    def test_boolean_column(self) -> None:
        """BOOLEAN (code 11) uses all 4 nibble bits — catches truncation bugs."""
        resp = RowsResponse(
            column_names=["flag"],
            column_types=[ValueType.BOOLEAN],
            rows=[[True], [False]],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        assert decoded.rows == [[True], [False]]
        assert decoded.column_types == [ValueType.BOOLEAN]

    def test_unixtime_column(self) -> None:
        """UNIXTIME (code 9) round-trips as raw int, not datetime."""
        resp = RowsResponse(
            column_names=["created_at"],
            column_types=[ValueType.UNIXTIME],
            row_types=[[ValueType.UNIXTIME], [ValueType.UNIXTIME]],
            rows=[[1700000000], [0]],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        assert decoded.rows == [[1700000000], [0]]
        assert decoded.column_types == [ValueType.UNIXTIME]

    def test_iso8601_column(self) -> None:
        """ISO8601 values round-trip as raw text at the wire level."""
        iso = "2024-06-15 10:30:00+00:00"
        resp = RowsResponse(
            column_names=["ts"],
            column_types=[ValueType.ISO8601],
            rows=[[iso]],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        assert decoded.rows[0][0] == iso

    def test_blob_column(self) -> None:
        """BLOB has variable-length encoding with padding in row context."""
        resp = RowsResponse(
            column_names=["data"],
            column_types=[ValueType.BLOB],
            rows=[[b"\xde\xad\xbe\xef"], [b""]],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        assert decoded.rows == [[b"\xde\xad\xbe\xef"], [b""]]

    def test_mixed_all_types(self) -> None:
        """Every value type in a single row exercises full encoding path."""
        iso = "2024-01-01 00:00:00+00:00"
        resp = RowsResponse(
            column_names=["i", "f", "t", "b", "n", "ut", "iso", "bo"],
            column_types=[
                ValueType.INTEGER,
                ValueType.FLOAT,
                ValueType.TEXT,
                ValueType.BLOB,
                ValueType.NULL,
                ValueType.UNIXTIME,
                ValueType.ISO8601,
                ValueType.BOOLEAN,
            ],
            row_types=[
                [
                    ValueType.INTEGER,
                    ValueType.FLOAT,
                    ValueType.TEXT,
                    ValueType.BLOB,
                    ValueType.NULL,
                    ValueType.UNIXTIME,
                    ValueType.ISO8601,
                    ValueType.BOOLEAN,
                ]
            ],
            rows=[
                [42, 3.14, "hello", b"\x00\x01", None, 1700000000, iso, True],
            ],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        row = decoded.rows[0]
        assert row[0] == 42
        assert row[1] == 3.14
        assert row[2] == "hello"
        assert row[3] == b"\x00\x01"
        assert row[4] is None
        assert row[5] == 1700000000
        assert row[6] == iso
        assert row[7] is True

    def test_blob_multiple_sizes(self) -> None:
        """Multiple BLOB rows with different sizes verify padding/offset interplay."""
        resp = RowsResponse(
            column_names=["data"],
            column_types=[ValueType.BLOB],
            rows=[
                [b""],  # empty
                [b"x"],  # 1 byte, needs 7 pad
                [b"12345678"],  # exactly 8 bytes, no pad
                [b"123456789"],  # 9 bytes, needs 7 pad
            ],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        assert decoded.rows[0] == [b""]
        assert decoded.rows[1] == [b"x"]
        assert decoded.rows[2] == [b"12345678"]
        assert decoded.rows[3] == [b"123456789"]


class TestEmptyResponse:
    def test_encode(self) -> None:
        msg = EmptyResponse()
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == ResponseType.EMPTY
        assert header.size_words == 1  # Reserved uint64 per Go spec

    def test_body_has_reserved_field(self) -> None:
        """EmptyResponse body must contain a reserved uint64 per Go spec."""
        msg = EmptyResponse()
        body = msg.encode_body()
        assert len(body) == 8

    def test_roundtrip(self) -> None:
        msg = EmptyResponse()
        encoded = msg.encode()
        decoded = EmptyResponse.decode_body(encoded[HEADER_SIZE:])
        assert isinstance(decoded, EmptyResponse)


class TestFilesResponse:
    def test_empty(self) -> None:
        msg = FilesResponse(files={})
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files == {}

    def test_empty_wire_format(self) -> None:
        """Empty files should encode as just uint64 count=0."""
        msg = FilesResponse(files={})
        body = msg.encode_body()
        # Should be exactly 8 bytes: uint64 count = 0
        assert len(body) == 8
        assert body == b"\x00" * 8

    def test_wire_format_starts_with_count(self) -> None:
        """Body must start with uint64 file count per Go wire protocol."""
        from dqlitewire.types import decode_uint64

        msg = FilesResponse(files={"test.db": b"datadata"})  # 8 bytes
        body = msg.encode_body()
        count = decode_uint64(body[:8])
        assert count == 1

    def test_wire_format_has_size_field(self) -> None:
        """Each file entry has text name, uint64 size, then raw content bytes."""
        from dqlitewire.types import decode_text, decode_uint64

        content = b"abcdefgh"
        msg = FilesResponse(files={"test.db": content})
        body = msg.encode_body()
        offset = 8  # skip count
        _name, consumed = decode_text(body[offset:])
        offset += consumed
        # Next should be uint64 size of content
        size = decode_uint64(body[offset:])
        assert size == len(content)

    def test_roundtrip(self) -> None:
        # 16 and 8 bytes — word-aligned per upstream C's dumpFile assert.
        msg = FilesResponse(files={"db.sqlite": b"databasecontent!", "wal": b"wal data"})
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files["db.sqlite"] == b"databasecontent!"
        assert decoded.files["wal"] == b"wal data"

    def test_roundtrip_single_file(self) -> None:
        msg = FilesResponse(files={"main.db": b"\x00\x01\x02\x03\x04\x05\x06\x07"})
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files["main.db"] == b"\x00\x01\x02\x03\x04\x05\x06\x07"

    def test_roundtrip_aligned_content(self) -> None:
        """Real dqlite content is always word-aligned (SQLite pages are multiples of 512)."""
        page = b"\x00" * 512  # Realistic SQLite page size
        msg = FilesResponse(
            files={
                "main.db": page,
                "wal.db": page + page,
            }
        )
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files["main.db"] == page
        assert decoded.files["wal.db"] == page + page

    def test_encode_rejects_non_aligned_content(self) -> None:
        """Content whose length is not a multiple of 8 is rejected at encode.

        The upstream C server (gateway.c::dumpFile) asserts
        ``len % 8 == 0``. We enforce the same invariant on the encoder
        side so mock-server frames cannot diverge from real C output.
        """
        from dqlitewire.exceptions import EncodeError

        msg = FilesResponse(files={"file1.db": b"\x01\x02\x03"})  # 3 bytes
        with pytest.raises(EncodeError, match="8-byte aligned"):
            msg.encode_body()

        msg = FilesResponse(files={"file2.db": b"\x04\x05\x06\x07\x08\x09\x0a"})  # 7
        with pytest.raises(EncodeError, match="8-byte aligned"):
            msg.encode_body()

    def test_roundtrip_empty_content(self) -> None:
        """116: zero-length file content must round-trip correctly."""
        msg = FilesResponse(files={"empty.db": b""})
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files == {"empty.db": b""}

    def test_roundtrip_mixed_empty_and_nonempty(self) -> None:
        """116: empty and non-empty files in the same response.

        All non-empty content must be 8-byte aligned.
        """
        msg = FilesResponse(
            files={
                "main.db": b"datadata",
                "empty.db": b"",
                "wal.db": b"moredata",
            }
        )
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files == msg.files

    def test_aligned_content_has_no_padding(self) -> None:
        """Word-aligned content must not produce any extra padding bytes."""
        content = b"\x00" * 16  # exactly 2 words
        msg = FilesResponse(files={"a.db": content})
        body = msg.encode_body()
        # count(8) + name "a.db\0"(8) + size(8) + content(16) = 40
        assert len(body) == 40

    def test_no_padding_between_files_matches_go(self) -> None:
        """Files are written back-to-back with no padding between
        entries (upstream gateway.c::dumpFile). Each file's own content
        must be 8-byte aligned, which matches reality (SQLite pages are
        always multiples of 512).
        """
        from dqlitewire.types import encode_text, encode_uint64

        body = encode_uint64(2)  # count=2
        body += encode_text("f1")  # filename
        body += encode_uint64(8)  # size=8 (one word)
        body += b"\x01\x02\x03\x04\x05\x06\x07\x08"  # aligned content
        body += encode_text("f2")  # next filename immediately after
        body += encode_uint64(8)  # size=8
        body += b"\xff" * 8  # content
        decoded = FilesResponse.decode_body(body)
        assert decoded.files["f1"] == b"\x01\x02\x03\x04\x05\x06\x07\x08"
        assert decoded.files["f2"] == b"\xff" * 8

    def test_decode_rejects_non_aligned_content(self) -> None:
        """Symmetric with the encode-side alignment check: a peer that
        claims a non-multiple-of-8 content size is producing bytes the
        real C server could never emit.
        """
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_text, encode_uint64

        body = encode_uint64(1)  # count=1
        body += encode_text("f1")
        body += encode_uint64(3)  # size=3, non-aligned
        body += b"\x01\x02\x03"
        with pytest.raises(DecodeError, match="8-byte aligned"):
            FilesResponse.decode_body(body)

    def test_bogus_file_count_raises(self) -> None:
        """A file count larger than remaining data should raise DecodeError early."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        body = encode_uint64(1_000_000_000) + b"\x00" * 8
        with pytest.raises(DecodeError, match="exceeds maximum"):
            FilesResponse.decode_body(body)

    def test_file_count_exceeds_hard_limit(self) -> None:
        """File count exceeding the hard limit should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # 200 files, enough data to pass data-size check
        body = encode_uint64(200) + b"\x00" * (200 * 16 + 8)
        with pytest.raises(DecodeError, match="File count.*exceeds maximum"):
            FilesResponse.decode_body(body)

    def test_truncated_file_content_raises(self) -> None:
        """Declared file size larger than available data should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_text, encode_uint64

        body = encode_uint64(1)  # count=1
        body += encode_text("test.db")  # filename
        body += encode_uint64(4096)  # claims 4096 bytes
        body += b"\x00" * 100  # but only 100 bytes available
        with pytest.raises(DecodeError, match="truncated"):
            FilesResponse.decode_body(body)

    def test_rejects_duplicate_filename(self) -> None:
        """The wire format is a positional sequence of N records; a
        dict-style silent overwrite would make ``len(files) < count``
        after decode and break re-encode symmetry. Upstream's
        ``handle_dump`` only ever emits distinct names (``main`` and
        ``main-wal``), so the duplicate must be rejected as a malicious
        or misframed peer rather than silently merged.
        """
        from dqlitewire.types import encode_text, encode_uint64

        body = encode_uint64(2)  # count=2
        body += encode_text("main") + encode_uint64(0)  # first
        body += encode_text("main") + encode_uint64(0)  # duplicate
        with pytest.raises(DecodeError, match="duplicate filename"):
            FilesResponse.decode_body(body)


class TestServersResponseAddressSize:
    """Per-node address length cap in ServersResponse decode."""

    def _build_body(self, address: str) -> bytes:
        # One node: uint64 id, text address, uint64 role.
        return (
            encode_uint64(1)
            + encode_uint64(1)
            + encode_text(address)
            + encode_uint64(2)  # NodeRole.VOTER
        )

    def test_decode_rejects_oversize_address(self) -> None:
        oversize = "a" * (MAX_ADDRESS_SIZE + 1)
        with pytest.raises(DecodeError, match="server address"):
            ServersResponse.decode_body(self._build_body(oversize))

    def test_decode_accepts_address_at_cap(self) -> None:
        at_cap = "a" * MAX_ADDRESS_SIZE
        decoded = ServersResponse.decode_body(self._build_body(at_cap))
        assert decoded.nodes[0].address == at_cap


class TestFilesResponseFilenameSize:
    """Per-filename length cap in FilesResponse decode.

    The outer 64 MiB frame cap bounds total bytes, but a peer can still
    pack a giant filename in a frame-legal FilesResponse. Cap each
    filename at ``MAX_FILENAME_SIZE`` (POSIX PATH_MAX convention).
    """

    def _build_body(self, name: str) -> bytes:
        # Single entry with word-aligned content.
        content = b"\x00" * 8
        return encode_uint64(1) + encode_text(name) + encode_uint64(len(content)) + content

    def test_decode_rejects_oversize_filename(self) -> None:
        oversize = "a" * (MAX_FILENAME_SIZE + 1)
        body = self._build_body(oversize)
        with pytest.raises(DecodeError, match="filename"):
            FilesResponse.decode_body(body)

    def test_decode_accepts_filename_at_cap(self) -> None:
        at_cap = "a" * MAX_FILENAME_SIZE
        body = self._build_body(at_cap)
        decoded = FilesResponse.decode_body(body)
        assert at_cap in decoded.files


class TestServersResponse:
    def test_empty(self) -> None:
        msg = ServersResponse(nodes=[])
        encoded = msg.encode()
        decoded = ServersResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.nodes == []

    def test_roundtrip(self) -> None:
        nodes = [
            NodeInfo(node_id=1, address="node1:9001", role=NodeRole.STANDBY),
            NodeInfo(node_id=2, address="node2:9002", role=NodeRole.SPARE),
            NodeInfo(node_id=3, address="node3:9003", role=NodeRole.SPARE),
        ]
        msg = ServersResponse(nodes=nodes)
        encoded = msg.encode()
        decoded = ServersResponse.decode_body(encoded[HEADER_SIZE:])
        assert len(decoded.nodes) == 3
        for i, expected in enumerate(nodes):
            assert decoded.nodes[i].node_id == expected.node_id
            assert decoded.nodes[i].address == expected.address
            assert decoded.nodes[i].role == expected.role

    def test_v0_format_body_rejected_by_existing_safety_nets(self) -> None:
        """Pin: a hostile peer that emits a V0-format body
        (id + address only, no role) is rejected by the existing
        24-byte-per-node bound and trailing-bytes reject when its
        layout doesn't align with V1's ``id + addr + role``.

        The decoder unconditionally parses V1; ``ClusterRequest``
        rejects ``format=0`` outbound so no in-tree client can
        receive a V0 body. This pin guards the residual hostile-
        peer path against silent-misparse regressions.
        """
        from dqlitewire.types import encode_text, encode_uint64

        # A V0 body for one node: count=1, id=1, address (no role
        # field). The V1 parser would read role from the next 8
        # bytes and either run off the end or report a trailing
        # bytes mismatch.
        body = encode_uint64(1) + encode_uint64(1) + encode_text("a:1")
        with pytest.raises(DecodeError):
            ServersResponse.decode_body(body)

    def test_wire_format_starts_with_count(self) -> None:
        """Body must start with uint64 node count per Go reference."""
        from dqlitewire.types import decode_uint64

        nodes = [
            NodeInfo(node_id=1, address="node1:9001", role=NodeRole.STANDBY),
            NodeInfo(node_id=2, address="node2:9002", role=NodeRole.SPARE),
        ]
        msg = ServersResponse(nodes=nodes)
        body = msg.encode_body()
        count = decode_uint64(body[:8])
        assert count == 2

    def test_empty_wire_format(self) -> None:
        """Empty nodes should encode as just uint64 count=0."""
        msg = ServersResponse(nodes=[])
        body = msg.encode_body()
        assert len(body) == 8
        assert body == b"\x00" * 8

    def test_bogus_node_count_raises(self) -> None:
        """A node count larger than remaining data should raise DecodeError early."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        body = encode_uint64(1_000_000_000) + b"\x00" * 8
        with pytest.raises(DecodeError, match="exceeds maximum"):
            ServersResponse.decode_body(body)

    def test_node_count_exceeds_hard_limit(self) -> None:
        """Node count exceeding the hard limit should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # 20_000 nodes, enough data to pass data-size check
        body = encode_uint64(20_000) + b"\x00" * (20_000 * 24 + 8)
        with pytest.raises(DecodeError, match="Node count.*exceeds maximum"):
            ServersResponse.decode_body(body)

    def test_decode_rejects_unknown_role(self) -> None:
        """An unknown ``role`` value must raise DecodeError at the wire seam.

        Upstream C (``src/roles.c``) only ever emits VOTER/STANDBY/SPARE
        (0/1/2). A server that sends anything else is either buggy or
        hostile; either way we refuse to build a NodeInfo with an
        unvalidated enum value — the failure must surface at the wire
        boundary, not silently propagate.
        """
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_text, encode_uint64

        body = (
            encode_uint64(1)  # count = 1
            + encode_uint64(7)  # node_id
            + encode_text("n1:9001")
            + encode_uint64(999)  # role = invalid
        )
        with pytest.raises(DecodeError, match="Invalid node role 999"):
            ServersResponse.decode_body(body)

    def test_roundtrip_preserves_noderole_type(self) -> None:
        """Decoded role must be a NodeRole member, not a bare int."""
        from dqlitewire.constants import NodeRole

        nodes = [
            NodeInfo(node_id=1, address="n1:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="n2:9002", role=NodeRole.STANDBY),
        ]
        encoded = ServersResponse(nodes=nodes).encode()
        decoded = ServersResponse.decode_body(encoded[HEADER_SIZE:])
        assert [n.role for n in decoded.nodes] == [NodeRole.VOTER, NodeRole.STANDBY]
        assert all(isinstance(n.role, NodeRole) for n in decoded.nodes)

    def test_int_equality_survives_enum_typing(self) -> None:
        """Downstream code compares ``role == 0`` / ``== 1``; IntEnum
        subclassing of int must keep those comparisons true even after
        we tightened the type annotation.
        """
        from dqlitewire.constants import NodeRole

        info = NodeInfo(node_id=1, address="n1:9001", role=NodeRole.VOTER)
        assert info.role == 0
        assert info.role == NodeRole.VOTER


class TestMetadataResponse:
    def test_roundtrip(self) -> None:
        msg = MetadataResponse(failure_domain=1, weight=50)
        encoded = msg.encode()
        decoded = MetadataResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.failure_domain == 1
        assert decoded.weight == 50

    @pytest.mark.parametrize(
        "failure_domain,weight",
        [
            (2**32 - 1, 2**32 - 1),
            (2**63, 2**63 + 1),
            (2**64 - 1, 2**64 - 1),
        ],
    )
    def test_roundtrip_uint64_boundaries(self, failure_domain: int, weight: int) -> None:
        """Both fields are declared uint64 on the wire. Pinning the high
        bits protects against a future refactor that narrows either
        field to int32 / int64 — a single happy-path test would still
        pass with (1, 50)."""
        msg = MetadataResponse(failure_domain=failure_domain, weight=weight)
        encoded = msg.encode()
        decoded = MetadataResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.failure_domain == failure_domain
        assert decoded.weight == weight


class TestShortBodyDecoding:
    """Fixed-width response decoders must raise ``DecodeError`` on a body
    shorter than the declared field layout. The primitive helpers
    (``decode_uint32`` / ``decode_uint64``) already raise on short input —
    these tests pin the contract at the message-class boundary so a
    future refactor that swaps a helper for a looser one surfaces here
    rather than producing silently truncated values.
    """

    @pytest.mark.parametrize(
        "cls,min_body",
        [
            (FailureResponse, 8),  # uint64 code
            (WelcomeResponse, 8),  # uint64 heartbeat_timeout
            (MetadataResponse, 16),  # 2 x uint64
            (ResultResponse, 16),  # 2 x uint64
            (DbResponse, 8),  # uint32 db_id + uint32 reserved
            (EmptyResponse, 8),  # uint64 reserved
        ],
    )
    def test_decode_body_raises_on_short(self, cls: Any, min_body: int) -> None:
        for length in range(min_body):
            with pytest.raises(DecodeError):
                cls.decode_body(b"\x00" * length)

    def test_leader_response_legacy_missing_null_terminator_raises(self) -> None:
        """``decode_body_legacy`` expects NUL-terminated text; bytes without
        a NUL must be rejected rather than produce an unterminated string.
        """
        with pytest.raises(DecodeError):
            LeaderResponse.decode_body_legacy(b"abc")

    def test_leader_response_missing_null_terminator_names_field(self) -> None:
        """When the address text has no NUL terminator, the DecodeError
        must carry the ``leader address`` label so operators can trace the
        failure to the LEADER response's address field rather than see a
        generic ``"Text not null-terminated"``.
        """
        # 8 bytes of node_id then bytes with no NUL.
        body = b"\x01\x00\x00\x00\x00\x00\x00\x00" + b"abc"
        with pytest.raises(DecodeError, match="leader address not null-terminated"):
            LeaderResponse.decode_body(body)

    def test_decode_text_uses_label_on_null_term_error(self) -> None:
        """``decode_text``'s ``label`` kwarg must propagate into the
        null-terminator diagnostic, not just the max-length one."""
        from dqlitewire.types import decode_text

        with pytest.raises(DecodeError, match="zop not null-terminated"):
            decode_text(b"abc", label="zop")


class TestServerTextSanitization:
    """Server-supplied strings that flow into exception messages and
    logs must have C0 control characters and DEL replaced with '?' so a
    malicious server cannot inject ANSI escape sequences, CR/LF
    log-forgery, or NUL bytes into operator-facing output.

    Tab (0x09) and LF (0x0A) are preserved so multi-line diagnostics
    render as real newlines. CR (0x0D) is dropped because it is the
    log-injection vector.
    """

    def test_failure_response_sanitizes_ansi(self) -> None:
        # Encode bytes directly because encode_text(str) stringifies.
        from dqlitewire.types import encode_text, encode_uint64

        payload = encode_uint64(42) + encode_text("\x1b[2J\x1b[Hwiped!")
        decoded = FailureResponse.decode_body(payload)
        assert decoded.message == "?[2J?[Hwiped!"

    def test_failure_response_sanitizes_cr(self) -> None:
        from dqlitewire.types import encode_text, encode_uint64

        payload = encode_uint64(1) + encode_text("ok\r\nFAKE log line")
        decoded = FailureResponse.decode_body(payload)
        # LF preserved; CR replaced with '?' so it cannot forge a log line.
        assert decoded.message == "ok?\nFAKE log line"

    def test_failure_response_sanitizes_raw_nul(self) -> None:
        # Construct a body whose text field contains a raw NUL byte
        # *before* the real NUL terminator. The legitimate encoder
        # rejects this, so hand-build the body to model a malicious
        # peer that bypasses encode_text.
        from dqlitewire.types import encode_uint64

        # uint64 code (8 bytes) + raw "ab\x01cd" + NUL terminator + pad.
        body = encode_uint64(1) + b"ab\x01cd\x00\x00\x00"
        decoded = FailureResponse.decode_body(body)
        assert decoded.message == "ab?cd"

    def test_failure_response_preserves_tab_and_lf(self) -> None:
        from dqlitewire.types import encode_text, encode_uint64

        payload = encode_uint64(1) + encode_text("line1\nline2\tkey=val")
        decoded = FailureResponse.decode_body(payload)
        assert decoded.message == "line1\nline2\tkey=val"

    def test_leader_response_preserves_raw_address_bytes(self) -> None:
        """Address fields decode RAW — they flow into TCP routing and
        allowlist comparisons. Sanitisation happens at log / exception
        format time in the client layer, not at decode. If the decoder
        mangled the address, an operator-configured allowlist string
        would silently fail to match the peer's own canonical form.
        """
        from dqlitewire.types import encode_text, encode_uint64

        payload = encode_uint64(5) + encode_text("evil.com:9001\r\nHost: x")
        decoded = LeaderResponse.decode_body(payload)
        # Decode preserves the input verbatim (control characters
        # included). The client layer applies ``sanitize_server_text``
        # when formatting the address into an exception or log line.
        assert decoded.address == "evil.com:9001\r\nHost: x"

    def test_leader_response_legacy_preserves_raw_address(self) -> None:
        from dqlitewire.types import encode_text

        payload = encode_text("\x1b[31mred\x1b[0m")
        decoded = LeaderResponse.decode_body_legacy(payload)
        assert decoded.address == "\x1b[31mred\x1b[0m"

    def test_servers_response_preserves_raw_node_addresses(self) -> None:
        # Build a SERVERS body with a single node whose address contains
        # an ANSI clear-screen sequence. Decode keeps it raw; client-
        # layer log formatters sanitise at display time.
        from dqlitewire.constants import NodeRole
        from dqlitewire.types import encode_text, encode_uint64

        body = encode_uint64(1)
        body += encode_uint64(1)
        body += encode_text("node1\x1b[2J:9001")
        body += encode_uint64(NodeRole.VOTER)
        decoded = ServersResponse.decode_body(body)
        assert len(decoded.nodes) == 1
        assert decoded.nodes[0].address == "node1\x1b[2J:9001"

    # Trojan-Source / log-injection primitives. Each row here matches
    # one codepoint class inside ``_CONTROL_CHARS_RE``:
    #  - U+2028/U+2029 : line- and paragraph-separators (log-line forging)
    #  - U+202A-U+202E : LRE/RLE/PDF/LRO/RLO bidi-overrides
    #  - U+2066-U+2069 : LRI/RLI/FSI/PDI isolation formatting
    #  - U+061C        : Arabic letter mark (bidi)
    #  - U+200B-U+200F : zero-width and directional markers
    #  - U+FEFF        : zero-width no-break space / byte-order mark
    #  - U+1680        : Ogham space mark (visible-but-different blank)
    #  - U+180E        : Mongolian vowel separator (historically zero-width)
    #  - U+202F        : narrow no-break space (invisible whitespace)
    #  - U+2060        : word joiner (same family as ZWSP)
    # A regex regression narrowing to ``[\x00-\x1f]`` would pass the
    # ANSI/CR/NUL/Tab cases above while silently re-opening these.
    _BIDI_AND_ZERO_WIDTH: list[tuple[str, str]] = [
        (" ", "LINE SEPARATOR"),
        (" ", "PARAGRAPH SEPARATOR"),
        ("‪", "LRE"),
        ("‫", "RLE"),
        ("‬", "PDF"),
        ("‭", "LRO"),
        ("‮", "RLO"),
        ("⁦", "LRI"),
        ("⁧", "RLI"),
        ("⁨", "FSI"),
        ("⁩", "PDI"),
        ("؜", "ALM"),
        ("​", "ZWSP"),
        ("‌", "ZWNJ"),
        ("‍", "ZWJ"),
        ("‎", "LRM"),
        ("‏", "RLM"),
        ("﻿", "ZWNBSP/BOM"),
        (" ", "OGHAM SPACE MARK"),
        ("᠎", "MONGOLIAN VOWEL SEPARATOR"),
        (" ", "NARROW NO-BREAK SPACE"),
        ("⁠", "WORD JOINER"),
    ]

    @pytest.mark.parametrize(("bad_char", "label"), _BIDI_AND_ZERO_WIDTH)
    def test_failure_response_sanitizes_bidi_and_zero_width(
        self, bad_char: str, label: str
    ) -> None:
        from dqlitewire.types import encode_text, encode_uint64

        payload = encode_uint64(1) + encode_text(f"hello{bad_char}world")
        decoded = FailureResponse.decode_body(payload)
        assert decoded.message == "hello?world", f"failed on {label}"

    @pytest.mark.parametrize(("bad_char", "label"), _BIDI_AND_ZERO_WIDTH)
    def test_leader_response_preserves_bidi_and_zero_width(self, bad_char: str, label: str) -> None:
        """Address decode is raw: the client layer sanitises at log-
        format time so routing and allowlist comparisons see the
        peer's authentic bytes. A future ``sanitize_server_text``
        applied at decode would split an operator-configured address
        set when the peer advertised its own canonical form.
        """
        from dqlitewire.types import encode_text, encode_uint64

        payload = encode_uint64(5) + encode_text(f"host{bad_char}:9001")
        decoded = LeaderResponse.decode_body(payload)
        assert decoded.address == f"host{bad_char}:9001", f"failed on {label}"

    @pytest.mark.parametrize(("bad_char", "label"), _BIDI_AND_ZERO_WIDTH)
    def test_servers_response_preserves_bidi_and_zero_width(
        self, bad_char: str, label: str
    ) -> None:
        """Node addresses decode raw — same rationale as the leader
        address decoder."""
        from dqlitewire.constants import NodeRole
        from dqlitewire.types import encode_text, encode_uint64

        body = encode_uint64(1)
        body += encode_uint64(1)
        body += encode_text(f"node{bad_char}1:9001")
        body += encode_uint64(NodeRole.VOTER)
        decoded = ServersResponse.decode_body(body)
        assert len(decoded.nodes) == 1
        assert decoded.nodes[0].address == f"node{bad_char}1:9001", f"failed on {label}"


class TestReservedFieldDiscard:
    """``DbResponse`` and ``EmptyResponse`` reserved fields are read-and-
    discarded on decode, matching Go's
    ``response.getUint32()`` / ``getUint64()`` discard at
    ``internal/protocol/response.go:140`` and ``:186``. The C field is
    documented as ``__pad__`` in ``response.h`` and is reserved for
    future protocol extensions; rejecting non-zero values would create
    a forward-compat hazard with no defensive value (the field is not
    a checksum or invariant).

    NOTE: ``LeaderRequest`` and other request-side fields keep strict
    rejection — the asymmetry is intentional, mirroring Go's split
    between request encoders (we must produce zero) and response
    decoders (we must accept whatever upstream emits).
    """

    def test_db_response_accepts_nonzero_reserved(self) -> None:
        body = b"\x01\x00\x00\x00" + b"\xff\xff\xff\xff"
        assert DbResponse.decode_body(body).db_id == 1

    def test_db_response_accepts_zero_reserved(self) -> None:
        # Round-trip form stays accepted.
        body = b"\x01\x00\x00\x00" + b"\x00\x00\x00\x00"
        assert DbResponse.decode_body(body).db_id == 1

    def test_empty_response_accepts_nonzero_reserved(self) -> None:
        assert isinstance(EmptyResponse.decode_body(b"\xff" * 8), EmptyResponse)

    def test_empty_response_accepts_zero_reserved(self) -> None:
        assert isinstance(EmptyResponse.decode_body(b"\x00" * 8), EmptyResponse)


class TestEncodeSideCaps:
    """Encode-side caps mirror the decode-side defense-in-depth bounds.

    The wire decoders already reject oversized addresses, node counts,
    column counts, column-name sizes, param counts, tail offsets, and
    filenames. The encoders must refuse the same values so that a mock
    server / proxy cannot produce bytes a conforming peer would reject.
    """

    def test_leader_response_rejects_oversize_address(self) -> None:
        oversize = "a" * (MAX_ADDRESS_SIZE + 1)
        with pytest.raises(EncodeError, match="leader address"):
            LeaderResponse(node_id=1, address=oversize).encode_body()

    def test_leader_response_accepts_address_at_cap(self) -> None:
        at_cap = "a" * MAX_ADDRESS_SIZE
        body = LeaderResponse(node_id=1, address=at_cap).encode_body()
        decoded = LeaderResponse.decode_body(body)
        assert decoded.address == at_cap

    def test_leader_response_legacy_rejects_oversize_address(self) -> None:
        oversize = "a" * (MAX_ADDRESS_SIZE + 1)
        with pytest.raises(EncodeError, match="leader address"):
            LeaderResponse(node_id=0, address=oversize).encode_body_legacy()

    def test_leader_response_legacy_accepts_address_at_cap(self) -> None:
        at_cap = "a" * MAX_ADDRESS_SIZE
        body = LeaderResponse(node_id=0, address=at_cap).encode_body_legacy()
        decoded = LeaderResponse.decode_body_legacy(body)
        assert decoded.address == at_cap

    def test_leader_response_legacy_rejects_non_zero_node_id(self) -> None:
        """Legacy encoding has no field for ``node_id``; non-zero
        callers must use ``encode_body()`` (modern format) instead.
        Pin the rejection so a future refactor cannot silently drop a
        meaningful ``node_id`` on the wire."""
        with pytest.raises(EncodeError, match="cannot carry node_id"):
            LeaderResponse(node_id=7, address="x:1").encode_body_legacy()

    def test_servers_response_rejects_oversize_node_count(self) -> None:
        # ``range(1, MAX_NODE_COUNT + 2)`` skips node_id=0 (rejected
        # at construction by NodeInfo.__post_init__'s raft-config
        # invariant). Still produces MAX_NODE_COUNT + 1 entries to
        # trip the count cap.
        nodes = [
            NodeInfo(node_id=i, address="n:9", role=NodeRole.SPARE)
            for i in range(1, MAX_NODE_COUNT + 2)
        ]
        with pytest.raises(EncodeError, match="node count"):
            ServersResponse(nodes=nodes).encode_body()

    def test_servers_response_rejects_oversize_address(self) -> None:
        oversize = "a" * (MAX_ADDRESS_SIZE + 1)
        nodes = [NodeInfo(node_id=1, address=oversize, role=NodeRole.SPARE)]
        with pytest.raises(EncodeError, match="server address"):
            ServersResponse(nodes=nodes).encode_body()

    def test_servers_response_accepts_address_at_cap(self) -> None:
        at_cap = "a" * MAX_ADDRESS_SIZE
        nodes = [NodeInfo(node_id=1, address=at_cap, role=NodeRole.SPARE)]
        body = ServersResponse(nodes=nodes).encode_body()
        decoded = ServersResponse.decode_body(body)
        assert decoded.nodes[0].address == at_cap

    def test_rows_response_rejects_oversize_column_count(self) -> None:
        # No rows — just exercise the column_count cap.
        names = [f"c{i}" for i in range(MAX_COLUMN_COUNT + 1)]
        with pytest.raises(EncodeError, match="column count"):
            RowsResponse(column_names=names, rows=[]).encode_body()

    def test_rows_response_rejects_oversize_column_name(self) -> None:
        oversize = "a" * (MAX_COLUMN_NAME_SIZE + 1)
        with pytest.raises(EncodeError, match="(?i)column name"):
            RowsResponse(column_names=[oversize], rows=[]).encode_body()

    def test_rows_response_accepts_column_name_at_cap(self) -> None:
        at_cap = "a" * MAX_COLUMN_NAME_SIZE
        body = RowsResponse(column_names=[at_cap], rows=[]).encode_body()
        decoded = RowsResponse.decode_body(body)
        assert decoded.column_names == [at_cap]

    def test_stmt_response_rejects_oversize_num_params(self) -> None:
        with pytest.raises(EncodeError, match="num_params"):
            StmtResponse(db_id=1, stmt_id=1, num_params=MAX_PARAM_COUNT + 1).encode_body()

    def test_stmt_response_accepts_num_params_at_cap(self) -> None:
        body = StmtResponse(db_id=1, stmt_id=1, num_params=MAX_PARAM_COUNT).encode_body()
        decoded = StmtResponse.decode_body(body, schema=0)
        assert decoded.num_params == MAX_PARAM_COUNT

    def test_stmt_response_rejects_oversize_tail_offset(self) -> None:
        with pytest.raises(EncodeError, match="tail_offset"):
            StmtResponse(
                db_id=1, stmt_id=1, num_params=0, tail_offset=MAX_TAIL_OFFSET + 1
            ).encode_body()

    def test_stmt_response_accepts_tail_offset_at_cap(self) -> None:
        body = StmtResponse(
            db_id=1, stmt_id=1, num_params=0, tail_offset=MAX_TAIL_OFFSET
        ).encode_body()
        decoded = StmtResponse.decode_body(body, schema=1)
        assert decoded.tail_offset == MAX_TAIL_OFFSET

    def test_files_response_rejects_oversize_filename(self) -> None:
        oversize = "a" * (MAX_FILENAME_SIZE + 1)
        with pytest.raises(EncodeError, match="filename"):
            FilesResponse(files={oversize: b""}).encode_body()

    def test_files_response_accepts_filename_at_cap(self) -> None:
        at_cap = "a" * MAX_FILENAME_SIZE
        body = FilesResponse(files={at_cap: b""}).encode_body()
        decoded = FilesResponse.decode_body(body)
        assert at_cap in decoded.files


class TestNodeInfoFrozenSlotted:
    """Mirror of ``dqliteclient.node_store.NodeInfo`` invariants."""

    def test_is_frozen(self) -> None:
        import dataclasses

        node = NodeInfo(node_id=1, address="host:1", role=NodeRole.SPARE)
        with pytest.raises(dataclasses.FrozenInstanceError):
            node.node_id = 2  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        a = NodeInfo(node_id=1, address="host:1", role=NodeRole.SPARE)
        b = NodeInfo(node_id=1, address="host:1", role=NodeRole.SPARE)
        assert hash(a) == hash(b)
        assert {a, b} == {a}


class TestDecodeTextCapsAreByteBased:
    """Sized text decoders must short-circuit on UTF-8 byte count, not
    Python character count.

    A 3-byte-per-codepoint UTF-8 string (e.g. CJK) at ``_MAX_X // 2``
    characters occupies 1.5× ``_MAX_X`` bytes on the wire. A
    post-decode ``len(str) > cap`` guard accepts it; a byte-level
    ``decode_text(max_size=cap)`` rejects it before UTF-8 decoding.
    """

    def test_failure_response_rejects_oversize_utf8_message(self) -> None:
        # "漢" is U+6F22, 3 UTF-8 bytes.
        char_count = MAX_FAILURE_MESSAGE_SIZE // 3 + 1
        msg = "漢" * char_count
        body = encode_uint64(1) + encode_text(msg)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            FailureResponse.decode_body(body)

    def test_leader_response_rejects_oversize_utf8_address(self) -> None:
        char_count = MAX_ADDRESS_SIZE // 3 + 1
        address = "漢" * char_count
        body = encode_uint64(1) + encode_text(address)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            LeaderResponse.decode_body(body)

    def test_leader_response_legacy_rejects_oversize_utf8_address(self) -> None:
        char_count = MAX_ADDRESS_SIZE // 3 + 1
        address = "漢" * char_count
        body = encode_text(address)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            LeaderResponse.decode_body_legacy(body)

    def test_servers_response_rejects_oversize_utf8_address(self) -> None:
        char_count = MAX_ADDRESS_SIZE // 3 + 1
        address = "漢" * char_count
        body = encode_uint64(1) + encode_uint64(1) + encode_text(address) + encode_uint64(2)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            ServersResponse.decode_body(body)

    def test_rows_response_rejects_oversize_utf8_column_name(self) -> None:
        char_count = MAX_COLUMN_NAME_SIZE // 3 + 1
        name = "漢" * char_count
        body = encode_uint64(1) + encode_text(name) + encode_uint64(0)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            RowsResponse.decode_body(body)

    def test_files_response_rejects_oversize_utf8_filename(self) -> None:
        char_count = MAX_FILENAME_SIZE // 3 + 1
        name = "漢" * char_count
        content = b"\x00" * 8
        body = encode_uint64(1) + encode_text(name) + encode_uint64(len(content)) + content
        with pytest.raises(DecodeError, match="exceeds maximum"):
            FilesResponse.decode_body(body)


class TestStmtResponsePostInitValidation:
    """Pin StmtResponse.__post_init__'s schema and tail_offset
    sanity checks. Both are reachable via direct construction; no
    other test exercises the post-init raises directly."""

    def test_post_init_rejects_schema_outside_known_set(self) -> None:
        from dqlitewire.exceptions import EncodeError
        from dqlitewire.messages.responses import StmtResponse

        with pytest.raises(EncodeError, match="must be 0 or 1"):
            StmtResponse(db_id=0, stmt_id=0, num_params=0, tail_offset=None, schema=2)

    def test_post_init_rejects_v0_schema_with_tail_offset(self) -> None:
        from dqlitewire.exceptions import EncodeError
        from dqlitewire.messages.responses import StmtResponse

        with pytest.raises(EncodeError, match="schema=0"):
            StmtResponse(db_id=0, stmt_id=0, num_params=0, tail_offset=42, schema=0)


class TestFilesResponseEncodeCountCap:
    """Pin the encode-side count cap on FilesResponse — symmetric
    with the existing decode-side cap test in TestEncodeSideCaps."""

    def test_encode_rejects_count_above_max(self) -> None:
        from dqlitewire.limits import MAX_FILE_COUNT
        from dqlitewire.messages.responses import FilesResponse

        files = {f"f{i}": b"\x00" * 8 for i in range(MAX_FILE_COUNT + 1)}
        with pytest.raises(EncodeError, match="exceeds maximum"):
            FilesResponse(files=files).encode_body()


class TestResponsePostInitValidation:
    """Construction-time uint validation on response dataclasses, symmetric
    with the request-side __post_init__ validators. Without these, an
    out-of-range value surfaces as a confusing EncodeError at encode time
    instead of failing fast at construction."""

    @pytest.mark.parametrize("code", [-1, 2**64, 2**64 + 1])
    def test_failure_response_rejects_out_of_range_code(self, code: int) -> None:
        with pytest.raises(EncodeError, match="out of range for uint64"):
            FailureResponse(code=code, message="")

    def test_failure_response_rejects_bool_code(self) -> None:
        with pytest.raises(EncodeError, match="must be int"):
            FailureResponse(code=True, message="")

    @pytest.mark.parametrize("node_id", [-1, 2**64])
    def test_leader_response_rejects_out_of_range_node_id(self, node_id: int) -> None:
        with pytest.raises(EncodeError, match="out of range for uint64"):
            LeaderResponse(node_id=node_id, address="x")

    @pytest.mark.parametrize("heartbeat_timeout", [-1, 2**64])
    def test_welcome_response_rejects_out_of_range_heartbeat(self, heartbeat_timeout: int) -> None:
        with pytest.raises(EncodeError, match="out of range for uint64"):
            WelcomeResponse(heartbeat_timeout=heartbeat_timeout)

    @pytest.mark.parametrize("db_id", [-1, 2**32])
    def test_db_response_rejects_out_of_range_db_id(self, db_id: int) -> None:
        with pytest.raises(EncodeError, match="out of range for uint32"):
            DbResponse(db_id=db_id)

    @pytest.mark.parametrize("last_insert_id", [-1, 2**64])
    def test_result_response_rejects_out_of_range_last_insert_id(self, last_insert_id: int) -> None:
        with pytest.raises(EncodeError, match="out of range for uint64"):
            ResultResponse(last_insert_id=last_insert_id, rows_affected=0)

    @pytest.mark.parametrize("rows_affected", [-1, 2**64])
    def test_result_response_rejects_out_of_range_rows_affected(self, rows_affected: int) -> None:
        with pytest.raises(EncodeError, match="out of range for uint64"):
            ResultResponse(last_insert_id=0, rows_affected=rows_affected)

    @pytest.mark.parametrize("failure_domain", [-1, 2**64])
    def test_metadata_response_rejects_out_of_range_failure_domain(
        self, failure_domain: int
    ) -> None:
        with pytest.raises(EncodeError, match="out of range for uint64"):
            MetadataResponse(failure_domain=failure_domain, weight=0)

    @pytest.mark.parametrize("weight", [-1, 2**64])
    def test_metadata_response_rejects_out_of_range_weight(self, weight: int) -> None:
        with pytest.raises(EncodeError, match="out of range for uint64"):
            MetadataResponse(failure_domain=0, weight=weight)

    def test_in_range_values_construct_cleanly(self) -> None:
        """Negative pin: in-spec values do not raise."""
        # FailureResponse code=0 is accepted: upstream's gateway emits
        # ``failure(req, 0, "empty statement")`` for empty / comment-only
        # SQL — see test_construct_with_code_zero_accepted.
        FailureResponse(code=0, message="empty statement")
        FailureResponse(code=1, message="")
        FailureResponse(code=2**64 - 1, message="x")
        LeaderResponse(node_id=0, address="")
        LeaderResponse(node_id=2**64 - 1, address="x")
        WelcomeResponse(heartbeat_timeout=15000)
        DbResponse(db_id=0)
        DbResponse(db_id=2**32 - 1)
        ResultResponse(last_insert_id=0, rows_affected=0)
        MetadataResponse(failure_domain=0, weight=0)


class TestEncodeBodyLinearTime:
    """Perf-floor pins: encoders accumulate into ``bytearray`` so the
    body grows in amortised O(1) per append. The prior ``bytes += bytes``
    shape paid Θ(N²) memcopy on the running result, which on a max-cap
    payload turned a sub-millisecond encode into a multi-second loop
    stall. The ceilings are deliberately generous (CI runners with
    variable load) but tight enough that the regressed shape breaches
    them at the chosen sizes — the only point is to fail if a refactor
    reintroduces the quadratic shape.
    """

    def test_rows_response_encode_body_linear_on_row_count(self) -> None:
        import time

        # 20k rows × small payload: under the regressed ``bytes += bytes``
        # shape this measures ~5+ s on commodity hardware; under the
        # linear shape it is well under 100 ms. 1.5 s is the slack
        # budget — anything near the regression cost signals the
        # quadratic shape has returned.
        rows: list[list[WireValue]] = [[i, f"row-{i:08d}"] for i in range(20000)]
        msg = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=rows,
            has_more=False,
        )
        start = time.monotonic()
        msg.encode_body()
        elapsed = time.monotonic() - start
        assert elapsed < 1.5, (
            f"RowsResponse.encode_body for 20000 rows took {elapsed:.3f}s; "
            "expected linear-time behaviour (regressed shape takes 5+ s)"
        )

    def test_files_response_encode_body_linear_on_file_count(self) -> None:
        import time

        # ``MAX_FILE_COUNT`` is 100 by design, so the file-count axis
        # is small; the regression sensitivity here comes from per-file
        # ``content`` size. Use a per-file payload large enough that
        # repeated ``bytes += bytes`` would copy the running result
        # tens of times.
        files = {f"file-{i:05d}": b"\x00" * 65536 for i in range(99)}
        msg = FilesResponse(files=files)
        start = time.monotonic()
        msg.encode_body()
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, (
            f"FilesResponse.encode_body for {len(files)} files took {elapsed:.3f}s; "
            "expected linear-time behaviour"
        )

    def test_servers_response_encode_body_linear_on_node_count(self) -> None:
        import time

        # Up to ``MAX_NODE_COUNT`` (10k) nodes; 8000 makes the
        # regression detectable while staying under the cap.
        nodes = [
            NodeInfo(node_id=i + 1, address=f"node-{i}.example:8080", role=NodeRole.VOTER)
            for i in range(8000)
        ]
        msg = ServersResponse(nodes=nodes)
        start = time.monotonic()
        msg.encode_body()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, (
            f"ServersResponse.encode_body for {len(nodes)} nodes took {elapsed:.3f}s; "
            "expected linear-time behaviour"
        )


class TestScalarResponseClassesFrozenSlotted:
    """Pin: scalar response classes are frozen + slotted, matching
    the NodeInfo precedent. Wire-decoded values are handed off for
    routing decisions; mutation would invalidate caller-held
    references and hashability lets instances live in sets / dict
    keys."""

    def test_failure_response_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        e = FailureResponse(code=5, message="boom")
        with pytest.raises(FrozenInstanceError):
            e.code = 99  # type: ignore[misc]

    def test_welcome_response_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        e = WelcomeResponse(heartbeat_timeout=15000)
        with pytest.raises(FrozenInstanceError):
            e.heartbeat_timeout = 0  # type: ignore[misc]

    def test_db_response_is_hashable(self) -> None:
        a = DbResponse(db_id=1)
        b = DbResponse(db_id=1)
        c = DbResponse(db_id=2)
        assert hash(a) == hash(b)
        assert {a, b, c} == {a, c}

    def test_metadata_response_is_frozen(self) -> None:
        """Pin frozen for the smaller scalar classes too."""
        from dataclasses import FrozenInstanceError

        m = MetadataResponse(failure_domain=1, weight=2)
        with pytest.raises(FrozenInstanceError):
            m.weight = 99  # type: ignore[misc]


# ---- merged from test_responses_strict_length.py ----
# Fixed-length response decoders must reject trailing bytes.
#
# Strict-parse peers in this module (``LeaderRequest.decode_body``,
# ``StmtResponse.decode_body``) assert exact body lengths. The three
# messages audited here previously accepted ``len >= expected``,
# silently ignoring trailing bytes — asymmetric with peers and
# permissive in a way that masks frame-corruption.


class TestEmptyResponseStrictLength:
    def test_short_body_rejected(self) -> None:
        with pytest.raises(DecodeError, match="EmptyResponse body must be exactly 8 bytes"):
            EmptyResponse.decode_body(b"\x00" * 7)

    def test_exact_length_accepted(self) -> None:
        msg = EmptyResponse.decode_body(b"\x00" * 8)
        assert isinstance(msg, EmptyResponse)

    def test_trailing_bytes_rejected(self) -> None:
        """16-byte body: decoders used to silently discard the
        trailing 8 bytes. Must now raise.
        """
        with pytest.raises(DecodeError, match="EmptyResponse body must be exactly 8 bytes"):
            EmptyResponse.decode_body(b"\x00" * 16)

    def test_trailing_zero_bytes_still_rejected(self) -> None:
        """Trailing zeros look innocuous but must still fail — the
        decoder should not need to introspect the padding."""
        with pytest.raises(DecodeError, match="EmptyResponse body must be exactly 8 bytes"):
            EmptyResponse.decode_body(b"\x00" * 9)

    def test_reserved_nonzero_accepted(self) -> None:
        """Match Go's ``response.getUint64()`` discard
        (``internal/protocol/response.go:186``). The reserved field
        is documented as unused; a future server reusing it must not
        break Python clients while Go clients continue working.
        """
        msg = EmptyResponse.decode_body(b"\x01" + b"\x00" * 7)
        assert isinstance(msg, EmptyResponse)


class TestDbResponseStrictLength:
    def test_short_body_rejected(self) -> None:
        with pytest.raises(DecodeError, match="DbResponse body must be exactly 8 bytes"):
            DbResponse.decode_body(b"\x00" * 7)

    def test_exact_length_accepted(self) -> None:
        msg = DbResponse.decode_body(b"\x05\x00\x00\x00" + b"\x00" * 4)
        assert isinstance(msg, DbResponse)
        assert msg.db_id == 5

    def test_trailing_bytes_rejected(self) -> None:
        with pytest.raises(DecodeError, match="DbResponse body must be exactly 8 bytes"):
            DbResponse.decode_body(b"\x00" * 16)


class TestResultResponseStrictLength:
    def test_short_body_rejected_with_type_name(self) -> None:
        """Error message names ``ResultResponse``, not just
        ``uint64`` — so operators reading logs can trace the
        framed message."""
        with pytest.raises(DecodeError, match="ResultResponse body must be exactly 16 bytes"):
            ResultResponse.decode_body(b"\x00" * 15)

    def test_exact_length_accepted(self) -> None:
        body = (42).to_bytes(8, "little") + (7).to_bytes(8, "little")
        msg = ResultResponse.decode_body(body)
        assert msg.last_insert_id == 42
        assert msg.rows_affected == 7

    def test_trailing_bytes_rejected(self) -> None:
        with pytest.raises(DecodeError, match="ResultResponse body must be exactly 16 bytes"):
            ResultResponse.decode_body(b"\x00" * 17)


class TestStmtResponseStrictLength:
    """StmtResponse carries db_id + stmt_id + num_params (+ optional
    tail_offset for schema>=1). Body is exactly 16 bytes (schema 0) or
    24 bytes (schema 1). Trailing bytes had been silently accepted;
    sibling responses reject them.
    """

    @staticmethod
    def _body_schema0(db_id: int = 1, stmt_id: int = 42, num_params: int = 3) -> bytes:
        return (
            db_id.to_bytes(4, "little")
            + stmt_id.to_bytes(4, "little")
            + num_params.to_bytes(8, "little")
        )

    @staticmethod
    def _body_schema1(
        db_id: int = 1, stmt_id: int = 42, num_params: int = 3, tail_offset: int = 0
    ) -> bytes:
        return TestStmtResponseStrictLength._body_schema0(
            db_id, stmt_id, num_params
        ) + tail_offset.to_bytes(8, "little")

    def test_short_body_rejected_schema0(self) -> None:
        with pytest.raises(DecodeError, match=r"StmtResponse schema=0 body must be exactly 16"):
            StmtResponse.decode_body(b"\x00" * 15, schema=0)

    def test_exact_length_accepted_schema0(self) -> None:
        msg = StmtResponse.decode_body(self._body_schema0(), schema=0)
        assert msg.db_id == 1
        assert msg.stmt_id == 42
        assert msg.num_params == 3
        assert msg.tail_offset is None

    def test_trailing_bytes_rejected_schema0(self) -> None:
        """Previous decoder silently accepted extra bytes; must now
        raise so a conforming StmtResponse round-trips exactly."""
        body = self._body_schema0() + b"\x01"
        with pytest.raises(DecodeError, match=r"StmtResponse schema=0 body must be exactly 16"):
            StmtResponse.decode_body(body, schema=0)

    def test_short_body_rejected_schema1(self) -> None:
        with pytest.raises(DecodeError, match=r"StmtResponse schema=1 body must be exactly 24"):
            StmtResponse.decode_body(b"\x00" * 23, schema=1)

    def test_exact_length_accepted_schema1(self) -> None:
        msg = StmtResponse.decode_body(self._body_schema1(tail_offset=7), schema=1)
        assert msg.db_id == 1
        assert msg.stmt_id == 42
        assert msg.num_params == 3
        assert msg.tail_offset == 7

    def test_trailing_bytes_rejected_schema1(self) -> None:
        body = self._body_schema1() + b"\x02"
        with pytest.raises(DecodeError, match=r"StmtResponse schema=1 body must be exactly 24"):
            StmtResponse.decode_body(body, schema=1)


class TestWelcomeResponseStrictLength:
    """Body is uint64(heartbeat_timeout) — exactly 8 bytes."""

    def test_short_body_rejected(self) -> None:
        from dqlitewire.messages.responses import WelcomeResponse

        with pytest.raises(DecodeError, match=r"WelcomeResponse body must be exactly 8"):
            WelcomeResponse.decode_body(b"\x00" * 7)

    def test_exact_length_accepted(self) -> None:
        from dqlitewire.messages.responses import WelcomeResponse

        msg = WelcomeResponse.decode_body((42).to_bytes(8, "little"))
        assert msg.heartbeat_timeout == 42

    def test_trailing_bytes_rejected(self) -> None:
        from dqlitewire.messages.responses import WelcomeResponse

        with pytest.raises(DecodeError, match=r"WelcomeResponse body must be exactly 8"):
            WelcomeResponse.decode_body(b"\x00" * 9)


class TestServersResponseStrictLength:
    """Variable-length body: ``uint64 count`` then ``count`` × (id, text
    address, role). Trailing bytes after the last node had been silently
    dropped.
    """

    @staticmethod
    def _single_node_body(addr: str = "1.2.3.4:9001") -> bytes:
        from dqlitewire.types import encode_text, encode_uint64

        return (
            encode_uint64(1)  # count
            + encode_uint64(1)  # node_id
            + encode_text(addr)  # address (padded)
            + encode_uint64(0)  # role = voter
        )

    def test_empty_list_exact_round_trip(self) -> None:
        from dqlitewire.messages.responses import ServersResponse
        from dqlitewire.types import encode_uint64

        msg = ServersResponse.decode_body(encode_uint64(0))
        assert msg.nodes == []

    def test_single_node_exact_round_trip(self) -> None:
        from dqlitewire.messages.responses import ServersResponse

        msg = ServersResponse.decode_body(self._single_node_body())
        assert len(msg.nodes) == 1
        assert msg.nodes[0].address == "1.2.3.4:9001"

    def test_multi_node_exact_round_trip(self) -> None:
        """Offset accumulation through multiple iterations — exercises
        the per-node loop boundary condition that the single-node test
        cannot reach. node_id starts at 1 because raft reserves id=0
        for "no node" and ``ServersResponse`` enforces the
        ``(node_id, address)`` atomicity invariant — a 0-id entry
        with a non-empty address is rejected at the wire boundary."""
        from dqlitewire.messages.responses import ServersResponse
        from dqlitewire.types import encode_text, encode_uint64

        body = encode_uint64(3)
        for i in range(1, 4):
            body += encode_uint64(i)
            body += encode_text(f"node{i}.example:900{i}")
            body += encode_uint64(0)
        msg = ServersResponse.decode_body(body)
        assert [n.address for n in msg.nodes] == [
            "node1.example:9001",
            "node2.example:9002",
            "node3.example:9003",
        ]

    def test_trailing_bytes_rejected(self) -> None:
        from dqlitewire.messages.responses import ServersResponse

        body = self._single_node_body() + b"\x01"
        with pytest.raises(DecodeError, match=r"ServersResponse has 1 trailing byte"):
            ServersResponse.decode_body(body)

    def test_trailing_word_rejected(self) -> None:
        from dqlitewire.messages.responses import ServersResponse

        body = self._single_node_body() + b"\x00" * 8
        with pytest.raises(DecodeError, match=r"ServersResponse has 8 trailing byte"):
            ServersResponse.decode_body(body)

    def test_empty_list_trailing_bytes_rejected(self) -> None:
        """Count=0 with trailing bytes must still raise — otherwise a
        server could hide bytes behind an empty list."""
        from dqlitewire.messages.responses import ServersResponse
        from dqlitewire.types import encode_uint64

        body = encode_uint64(0) + b"\x00" * 8
        with pytest.raises(DecodeError, match=r"ServersResponse has 8 trailing byte"):
            ServersResponse.decode_body(body)


class TestFailureResponseStrictLength:
    """Body: uint64 code + padded text message. Unlike the fixed-length
    decoders above, the failure body is NOT strictly fixed-length:
    the server may append the genuine failure record after an
    un-rewound partial rows header (column count + names), so trailing
    bytes are tolerated. When the trailing region does not parse as a
    recoverable failure record, the decoder falls back to the first
    record — matching the reference Go client, which reads one record
    and ignores the rest — rather than raising on benign trailing data."""

    @staticmethod
    def _body(code: int = 5, message: str = "database is locked") -> bytes:
        from dqlitewire.types import encode_text, encode_uint64

        return encode_uint64(code) + encode_text(message)

    def test_exact_round_trip(self) -> None:
        from dqlitewire.messages.responses import FailureResponse

        msg = FailureResponse.decode_body(self._body())
        assert msg.code == 5
        assert msg.message == "database is locked"

    def test_trailing_garbage_byte_falls_back_to_first_record(self) -> None:
        # A single stray trailing byte does not parse as the column-name
        # sequence + trailing failure record the recovery path expects,
        # so the decoder surfaces the first record rather than raising.
        from dqlitewire.messages.responses import FailureResponse

        msg = FailureResponse.decode_body(self._body() + b"\x01")
        assert msg.code == 5
        assert msg.message == "database is locked"

    def test_trailing_word_falls_back_to_first_record(self) -> None:
        from dqlitewire.messages.responses import FailureResponse

        msg = FailureResponse.decode_body(self._body() + b"\x00" * 8)
        assert msg.code == 5
        assert msg.message == "database is locked"


class TestLeaderResponseStrictLength:
    """Modern body: uint64 node_id + padded text address. Legacy body:
    padded text address only. Trailing bytes after the address had
    been silently dropped."""

    @staticmethod
    def _modern_body(node_id: int = 7, addr: str = "1.2.3.4:9001") -> bytes:
        from dqlitewire.types import encode_text, encode_uint64

        return encode_uint64(node_id) + encode_text(addr)

    def test_modern_exact_round_trip(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse

        msg = LeaderResponse.decode_body(self._modern_body())
        assert msg.node_id == 7
        assert msg.address == "1.2.3.4:9001"

    def test_modern_trailing_byte_rejected(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse

        body = self._modern_body() + b"\x01"
        with pytest.raises(DecodeError, match=r"LeaderResponse has 1 trailing byte"):
            LeaderResponse.decode_body(body)

    def test_modern_trailing_word_rejected(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse

        body = self._modern_body() + b"\x00" * 8
        with pytest.raises(DecodeError, match=r"LeaderResponse has 8 trailing byte"):
            LeaderResponse.decode_body(body)

    def test_legacy_exact_round_trip(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse
        from dqlitewire.types import encode_text

        msg = LeaderResponse.decode_body_legacy(encode_text("10.0.0.1:9001"))
        assert msg.node_id == 0
        assert msg.address == "10.0.0.1:9001"

    def test_legacy_trailing_byte_rejected(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse
        from dqlitewire.types import encode_text

        body = encode_text("10.0.0.1:9001") + b"\x01"
        with pytest.raises(DecodeError, match=r"LeaderResponse \(legacy\) has 1 trailing byte"):
            LeaderResponse.decode_body_legacy(body)

    def test_legacy_trailing_word_rejected(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse
        from dqlitewire.types import encode_text

        body = encode_text("10.0.0.1:9001") + b"\x00" * 8
        with pytest.raises(DecodeError, match=r"LeaderResponse \(legacy\) has 8 trailing byte"):
            LeaderResponse.decode_body_legacy(body)

    def test_modern_short_body_rejected_at_response_layer(self) -> None:
        # ``decode_uint64`` already rejects bodies shorter than 8 bytes,
        # but its diagnostic does not mention which response was being
        # decoded. Sibling response decoders (FailureResponse,
        # WelcomeResponse, MetadataResponse) all emit an explicit
        # response-level length guard so the diagnostic is self-
        # describing. Pin the wording so a future refactor of
        # ``decode_uint64`` cannot silently demote LeaderResponse to
        # the helper's wording.
        from dqlitewire.messages.responses import LeaderResponse

        with pytest.raises(DecodeError, match=r"LeaderResponse body too short"):
            LeaderResponse.decode_body(b"\x01" * 7)

    def test_legacy_short_body_rejected_at_response_layer(self) -> None:
        # Sibling pin for the legacy decoder. The legacy body is a
        # single ``decode_text`` field — NUL-terminated UTF-8 padded
        # to the 8-byte boundary; there is NO length prefix on the
        # wire. The 8-byte minimum is the smallest padded TEXT
        # (1-byte NUL terminator + 7-byte zero-padding). Without
        # this response-layer guard, ``decode_text``'s generic
        # diagnostic does not identify the response being decoded.
        from dqlitewire.messages.responses import LeaderResponse

        with pytest.raises(
            DecodeError,
            match=r"LeaderResponse \(legacy\) body too short.*NUL-terminated",
        ):
            LeaderResponse.decode_body_legacy(b"\x01" * 7)


class TestMetadataResponseStrictLength:
    """Body is two uint64s (failure_domain + weight) — exactly 16 bytes."""

    def test_short_body_rejected(self) -> None:
        from dqlitewire.messages.responses import MetadataResponse

        with pytest.raises(DecodeError, match=r"MetadataResponse body must be exactly 16"):
            MetadataResponse.decode_body(b"\x00" * 15)

    def test_exact_length_accepted(self) -> None:
        from dqlitewire.messages.responses import MetadataResponse

        body = (3).to_bytes(8, "little") + (7).to_bytes(8, "little")
        msg = MetadataResponse.decode_body(body)
        assert msg.failure_domain == 3
        assert msg.weight == 7

    def test_trailing_bytes_rejected(self) -> None:
        from dqlitewire.messages.responses import MetadataResponse

        with pytest.raises(DecodeError, match=r"MetadataResponse body must be exactly 16"):
            MetadataResponse.decode_body(b"\x00" * 17)


# ---- merged from test_response_decoders_validate_schema.py ----
# Pin: every response-side ``decode_body`` validates the ``schema``
# kwarg at the per-class layer, mirroring the request-side discipline.
#
# The production dispatcher (``MessageDecoder.decode_bytes``) caps
# ``schema`` at ``_RESPONSE_MAX_SCHEMA[type]`` before calling per-class
# decoders, but direct callers (tests, proxies, fuzzers,
# golden-byte harnesses) bypass the dispatcher. Until this fix the
# per-class layer silently accepted bogus schemas — request decoders
# have always rejected. Pin the symmetric defense-in-depth.


# Minimally-valid body per response class so the bogus-schema reject
# fires before any body-shape decode would raise a different error.
def _body_for(cls: type) -> bytes:
    if cls is LeaderResponse:
        # node_id=0, address="" + padding = 8 + 8 bytes.
        return struct.pack("<Q", 0) + b"\x00" * 8
    # Most decoders accept a string of NULs as a degenerate body.
    return b"\x00" * 64


@pytest.mark.parametrize("msg_cls", list(RESPONSE_TYPES.values()))
def test_response_decode_body_rejects_unknown_schema(msg_cls: type) -> None:
    body = _body_for(msg_cls)
    with pytest.raises(DecodeError, match="(?i)schema"):
        msg_cls.decode_body(body, schema=99)  # type: ignore[attr-defined]


def test_stmt_response_v1_schema_still_accepted() -> None:
    """Regression: ``StmtResponse`` already accepted schema=1 (V1)
    and continues to do so — the broad reject is "unknown schema",
    not "non-zero schema"."""
    body = struct.pack("<II", 0, 1) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    msg = StmtResponse.decode_body(body, schema=1)
    assert msg.tail_offset == 0


# ---- merged from test_response_dataclass_bounded_repr.py ----
# Pin: response dataclasses with unbounded list / dict fields emit a
# bounded summary ``__repr__`` rather than enumerating every element
# (the dataclass-generated repr would produce multi-megabyte strings).


def test_rows_response_repr_is_bounded_under_large_rows() -> None:
    rows: list[list[WireValue]] = [[i, f"name-{i}"] for i in range(10_000)]
    row_types = [[ValueType.INTEGER, ValueType.TEXT] for _ in range(10_000)]
    msg = RowsResponse(
        column_names=["id", "name"],
        column_types=[ValueType.INTEGER, ValueType.TEXT],
        row_types=row_types,
        rows=rows,
        has_more=False,
    )
    rendered = repr(msg)

    assert len(rendered) < 500, (
        f"RowsResponse repr is {len(rendered)} chars for 10k rows; expected a bounded summary"
    )
    assert "<10000 items>" in rendered
    assert "has_more=False" in rendered


def test_files_response_repr_is_bounded_under_large_per_file_content() -> None:
    """A 50-file response with multi-MB content per file must
    produce a short repr that does not dump the bytes."""
    files = {f"file-{i:04d}": b"\x00" * (1 << 20) for i in range(50)}
    msg = FilesResponse(files=files)
    rendered = repr(msg)

    assert len(rendered) < 500, (
        f"FilesResponse repr is {len(rendered)} chars for 50 × 1MB files; "
        "expected a bounded summary"
    )
    # Truncated form should show a small head sample and an "and N more"
    # marker.
    assert "more" in rendered
    assert "bytes" in rendered


def test_servers_response_repr_is_bounded_under_large_node_count() -> None:
    """A 1000-node response must produce a short repr that does not
    enumerate every entry."""
    nodes = [
        NodeInfo(node_id=i + 1, address=f"10.0.0.{i // 256}.{i % 256}:9001", role=NodeRole.VOTER)
        for i in range(1000)
    ]
    msg = ServersResponse(nodes=nodes)
    rendered = repr(msg)

    assert len(rendered) < 500, (
        f"ServersResponse repr is {len(rendered)} chars for 1k nodes; expected a bounded summary"
    )
    assert "more" in rendered


def test_rows_response_small_repr_still_includes_field_names() -> None:
    """Small payloads still render the field names + summary count
    so the summary is informative."""
    msg = RowsResponse(
        column_names=["x"],
        column_types=[ValueType.INTEGER],
        rows=[[1], [2]],
        has_more=True,
    )
    rendered = repr(msg)
    assert "column_names=['x']" in rendered
    assert "<2 items>" in rendered
    assert "has_more=True" in rendered


def test_files_response_empty_repr() -> None:
    """Empty FilesResponse renders cleanly."""
    msg = FilesResponse()
    rendered = repr(msg)
    assert "FilesResponse" in rendered


def test_servers_response_empty_repr() -> None:
    """Empty ServersResponse renders cleanly."""
    msg = ServersResponse()
    rendered = repr(msg)
    assert "ServersResponse" in rendered


# ---- merged from test_result_response_caps.py ----
# ``ResultResponse.rows_affected`` is decode-capped at ``INT_MAX``;
# ``last_insert_id_signed`` returns the int64-cast for stdlib parity.
#
# The upstream C server returns ``sqlite3_changes(...)`` which is C
# ``int`` (``INT_MAX``) before being cast to uint64 on the wire
# (``gateway.c:484``). A real cluster never emits above INT_MAX.
#
# The signed accessor mirrors stdlib ``sqlite3.Connection.lastrowid``:
# SQLite's ``sqlite3_int64`` rowid can be negative; the unsigned wire
# value is ``2**64 - abs(rowid)``.
#
# Encode-side cap is symmetric with decode (mirrors the
# ``StmtResponse.MAX_TAIL_OFFSET`` symmetric-cap precedent at the
# sibling response): a Python encoder constructing a ``ResultResponse``
# with ``rows_affected > MAX_ROWS_AFFECTED`` is rejected up front, so
# a same-process round-trip via Python encoder + decoder cannot break
# on the encoder's own bytes.

_INT_MAX = (1 << 31) - 1


def test_rows_affected_at_int_max_accepted() -> None:
    body = encode_uint64(0) + encode_uint64(_INT_MAX)
    resp = ResultResponse.decode_body(body)
    assert resp.rows_affected == _INT_MAX


def test_rows_affected_above_int_max_rejected() -> None:
    body = encode_uint64(0) + encode_uint64(_INT_MAX + 1)
    with pytest.raises(DecodeError, match="rows_affected"):
        ResultResponse.decode_body(body)


def test_rows_affected_two_to_the_63_rejected() -> None:
    """Hostile-server scenario: explicit huge value."""
    body = encode_uint64(0) + encode_uint64(1 << 63)
    with pytest.raises(DecodeError, match="rows_affected"):
        ResultResponse.decode_body(body)


def test_last_insert_id_signed_round_trip_negative() -> None:
    """A SQLite rowid of -1 round-trips through uint64 as 2**64 - 1.
    The signed accessor reverses the cast."""
    on_wire = (1 << 64) - 1  # -1 as uint64
    resp = ResultResponse(last_insert_id=on_wire, rows_affected=0)
    assert resp.last_insert_id == on_wire
    assert resp.last_insert_id_signed == -1


def test_last_insert_id_signed_round_trip_positive() -> None:
    resp = ResultResponse(last_insert_id=42, rows_affected=0)
    assert resp.last_insert_id == 42
    assert resp.last_insert_id_signed == 42


def test_last_insert_id_signed_at_int64_max() -> None:
    """The signed boundary: 2**63 - 1 stays positive both
    unsigned and signed."""
    boundary = (1 << 63) - 1
    resp = ResultResponse(last_insert_id=boundary, rows_affected=0)
    assert resp.last_insert_id_signed == boundary


def test_last_insert_id_signed_at_int64_min() -> None:
    """``2**63`` on the wire is the most-negative int64 (=-2**63)."""
    on_wire = 1 << 63
    resp = ResultResponse(last_insert_id=on_wire, rows_affected=0)
    assert resp.last_insert_id_signed == -(1 << 63)


def test_rows_affected_at_int_max_construction_accepted() -> None:
    """Construction at the boundary is accepted (mirrors the decode
    side, which accepts at the boundary too)."""
    resp = ResultResponse(last_insert_id=0, rows_affected=_INT_MAX)
    assert resp.rows_affected == _INT_MAX


def test_rows_affected_above_int_max_construction_rejected() -> None:
    """Construction-time cap rejection — mirrors the decode-time cap
    so that a same-process Python encoder + decoder round-trip cannot
    break on its own bytes. Without this guard, ``ResultResponse(0,
    rows_affected=2**32).encode()`` succeeds and the same Python
    decoder then raises ``DecodeError`` on the produced bytes — a
    confusing self-inflicted asymmetry for mock-server / proxy authors.
    """
    with pytest.raises(EncodeError, match="rows_affected"):
        ResultResponse(last_insert_id=0, rows_affected=_INT_MAX + 1)


def test_rows_affected_two_to_the_32_construction_rejected() -> None:
    """The exact reproducer in the issue file: a `2**32` value is the
    canonical mock-server / proxy author trap."""
    with pytest.raises(EncodeError, match="rows_affected"):
        ResultResponse(last_insert_id=0, rows_affected=1 << 32)


def test_rows_affected_round_trip_at_boundary_succeeds() -> None:
    """Same-process round-trip at the boundary: construction +
    encode + decode all succeed."""
    resp = ResultResponse(last_insert_id=42, rows_affected=_INT_MAX)
    encoded = resp.encode()
    from dqlitewire.constants import HEADER_SIZE

    decoded = ResultResponse.decode_body(encoded[HEADER_SIZE:])
    assert decoded.rows_affected == _INT_MAX
    assert decoded.last_insert_id == 42


def test_last_insert_id_not_capped_at_int_max_at_construction() -> None:
    """Negative-pin: ``last_insert_id`` must NOT be capped at INT_MAX.
    The wire field is uint64 (per the protocol spec); SQLite's
    ``sqlite3_int64`` rowid can be the most-negative int64, which
    appears on the wire as ``2**63``. Capping ``last_insert_id``
    would break the existing ``last_insert_id_signed`` accessor
    contract. A future refactor copying the rows_affected cap to
    both fields would silently pass without this pin.
    """
    on_wire = (1 << 63) | 0xFEDCBA  # safely above INT_MAX
    resp = ResultResponse(last_insert_id=on_wire, rows_affected=0)
    assert resp.last_insert_id == on_wire


# ---- merged from test_stmt_response_post_init_setattr_idiom.py ----
# StmtResponse.__post_init__ coerces a V1-implicit tail_offset to zero.


def test_stmt_response_v1_default_tail_offset_normalised_to_zero() -> None:
    """schema=1 with tail_offset=None normalises to tail_offset=0."""
    msg = StmtResponse(db_id=1, stmt_id=2, num_params=0, schema=1)
    assert msg.tail_offset == 0


# ---- merged from test_stmt_response_rejects_unsupported_schema.py ----
# StmtResponse.decode_body rejects schema outside {0, 1} (defense-in-depth
# companion to the codec dispatch-table cap, for direct decode_body callers).


def _body_schema1(tail_offset: int = 0) -> bytes:
    """Build a syntactically valid V1 body (24 bytes)."""
    db_id = b"\x01\x00\x00\x00"
    stmt_id = b"\x02\x00\x00\x00"
    num_params = b"\x00" * 8
    tail = tail_offset.to_bytes(8, "little")
    return db_id + stmt_id + num_params + tail


def test_decode_body_rejects_schema_two() -> None:
    """schema=2 is undefined upstream (only V0/V1 exist), so reject it."""
    body = _body_schema1()
    with pytest.raises(DecodeError, match="unsupported schema"):
        StmtResponse.decode_body(body, schema=2)


def test_decode_body_rejects_negative_schema() -> None:
    body = _body_schema1()
    with pytest.raises(DecodeError, match="unsupported schema"):
        StmtResponse.decode_body(body, schema=-1)


def test_decode_body_accepts_schema_zero() -> None:
    body = b"\x01\x00\x00\x00" + b"\x02\x00\x00\x00" + b"\x00" * 8
    msg = StmtResponse.decode_body(body, schema=0)
    assert msg.db_id == 1
    assert msg.stmt_id == 2


def test_decode_body_accepts_schema_one() -> None:
    body = _body_schema1(tail_offset=42)
    msg = StmtResponse.decode_body(body, schema=1)
    assert msg.db_id == 1
    assert msg.stmt_id == 2
    assert msg.tail_offset == 42


# ---- merged from test_stmt_response_tail_offset_construction_cap.py ----
# StmtResponse.__post_init__ enforces MAX_TAIL_OFFSET at construction,
# in addition to the encode/decode caps (kept as defense-in-depth).


def test_stmt_response_tail_offset_construction_over_cap_rejected() -> None:
    with pytest.raises(EncodeError, match="exceeds maximum"):
        StmtResponse(
            db_id=0,
            stmt_id=0,
            num_params=0,
            tail_offset=MAX_TAIL_OFFSET + 1,
            schema=1,
        )


def test_stmt_response_tail_offset_at_cap_accepted() -> None:
    """The cap is exclusive: exactly at the cap is accepted."""
    msg = StmtResponse(
        db_id=0,
        stmt_id=0,
        num_params=0,
        tail_offset=MAX_TAIL_OFFSET,
        schema=1,
    )
    assert msg.tail_offset == MAX_TAIL_OFFSET


def test_stmt_response_tail_offset_none_unaffected() -> None:
    """tail_offset=None (the V0 default) is unaffected by the cap."""
    msg = StmtResponse(db_id=0, stmt_id=0, num_params=0)
    assert msg.tail_offset is None


def test_stmt_response_negative_tail_offset_still_rejected_by_uint64_validator() -> None:
    """_validate_uint64 must run before the cap check, else a negative value
    would compare under the positive cap and silently succeed."""
    with pytest.raises(EncodeError, match="out of range|must be int"):
        StmtResponse(
            db_id=0,
            stmt_id=0,
            num_params=0,
            tail_offset=-1,
            schema=1,
        )


def test_stmt_response_encode_body_cap_still_defense_in_depth() -> None:
    """encode-time cap still fires when tail_offset is mutated past construction."""
    msg = StmtResponse(db_id=0, stmt_id=0, num_params=0, tail_offset=0, schema=1)
    object.__setattr__(msg, "tail_offset", MAX_TAIL_OFFSET + 1)
    with pytest.raises(EncodeError, match="exceeds maximum"):
        msg.encode_body()


# ---- merged from test_stmt_response_validation_and_round_trip.py ----
# StmtResponse.__post_init__ validates uint ranges and normalises
# schema=1/tail_offset=None to tail_offset=0 so encode/decode round-trips equal.


class TestPostInitValidation:
    def test_negative_db_id_rejected(self) -> None:
        with pytest.raises(EncodeError, match="db_id"):
            StmtResponse(db_id=-1, stmt_id=0, num_params=0)

    def test_negative_stmt_id_rejected(self) -> None:
        with pytest.raises(EncodeError, match="stmt_id"):
            StmtResponse(db_id=0, stmt_id=-1, num_params=0)

    def test_db_id_overflow_rejected(self) -> None:
        with pytest.raises(EncodeError, match="db_id"):
            StmtResponse(db_id=2**32, stmt_id=0, num_params=0)

    def test_stmt_id_overflow_rejected(self) -> None:
        with pytest.raises(EncodeError, match="stmt_id"):
            StmtResponse(db_id=0, stmt_id=2**32, num_params=0)

    def test_num_params_overflow_rejected(self) -> None:
        with pytest.raises(EncodeError, match="num_params"):
            StmtResponse(db_id=0, stmt_id=0, num_params=2**64)

    def test_negative_tail_offset_rejected(self) -> None:
        with pytest.raises(EncodeError, match="tail_offset"):
            StmtResponse(db_id=0, stmt_id=0, num_params=0, tail_offset=-1)

    def test_bool_db_id_rejected(self) -> None:
        # bool is an int subclass, so db_id=True must be rejected (not encoded as 1).
        with pytest.raises(EncodeError, match="db_id"):
            StmtResponse(db_id=True, stmt_id=0, num_params=0)

    def test_num_params_above_max_rejected_at_construction(self) -> None:
        """num_params > MAX_PARAM_COUNT is rejected at construction, not
        deferred to encode_body."""
        with pytest.raises(EncodeError, match="num_params"):
            StmtResponse(db_id=0, stmt_id=0, num_params=2**40)

    def test_num_params_at_max_allowed(self) -> None:
        """The cap is inclusive: num_params == MAX_PARAM_COUNT constructs cleanly."""
        r = StmtResponse(db_id=0, stmt_id=0, num_params=MAX_PARAM_COUNT)
        assert r.num_params == MAX_PARAM_COUNT

    def test_num_params_encode_cap_still_enforced(self) -> None:
        """Defense-in-depth: encode_body keeps its own num_params cap even
        though __post_init__ now blocks it at construction."""
        r = StmtResponse(db_id=0, stmt_id=0, num_params=0)
        r.num_params = MAX_PARAM_COUNT + 1
        with pytest.raises(EncodeError, match="num_params"):
            r.encode_body()


class TestRoundTripIdentity:
    def test_v1_implicit_zero_normalises_to_zero(self) -> None:
        """schema=1/tail_offset=None normalises to 0 so it equals the decode result."""
        r = StmtResponse(db_id=1, stmt_id=2, num_params=3, schema=1)
        assert r.tail_offset == 0

    def test_v1_round_trip_identity_implicit_zero(self) -> None:
        r1 = StmtResponse(db_id=1, stmt_id=2, num_params=3, schema=1)
        body = r1.encode_body()
        r2 = StmtResponse.decode_body(body, schema=1)
        assert r1 == r2

    def test_v1_round_trip_identity_explicit_zero(self) -> None:
        r1 = StmtResponse(db_id=1, stmt_id=2, num_params=3, tail_offset=0, schema=1)
        body = r1.encode_body()
        r2 = StmtResponse.decode_body(body, schema=1)
        assert r1 == r2

    def test_v0_round_trip_identity_with_explicit_schema(self) -> None:
        """V0 dataclass equality needs an explicit schema=0 to match the decoder."""
        r1 = StmtResponse(db_id=1, stmt_id=2, num_params=3, schema=0)
        body = r1.encode_body()
        r2 = StmtResponse.decode_body(body, schema=0)
        assert r1 == r2


# ---- merged from test_failure_response_recovers_real_error_after_partial_rows_header.py ----
# FailureResponse.decode_body recovers the real (code, message) when the
# server frames a failure after an un-rewound partial rows header:
#
#     [ col_count ][ col_name_1 .. col_name_N ][ real code ][ real message ]
#
# Non-matching bodies fall back to the first record (matching the Go client);
# truncated bodies still raise.


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


# ---- merged from test_failure_response_round_trip_matrix.py ----
# End-to-end round-trip matrix for FailureResponse: extended/boundary/
# dqlite-namespace codes against empty/short/common messages.


@pytest.mark.parametrize(
    ("code", "message"),
    [
        # code=0 is genuinely emitted upstream (gateway.c:372 / :890).
        (0, "empty statement"),
        (1, ""),
        (1, "constraint violated"),
        (5, "checkpoint in progress"),
        (SQLITE_IOERR_NOT_LEADER, ""),
        (SQLITE_IOERR_NOT_LEADER, "not leader"),
        (SQLITE_IOERR_LEADERSHIP_LOST, ""),
        (SQLITE_IOERR_LEADERSHIP_LOST, "lost leadership mid-transaction"),
        (DQLITE_NOTFOUND, "no database opened"),
        (DQLITE_PARSE, "unknown request type"),
        # uint64 boundary.
        (2**63 - 1, ""),
        (2**63, ""),
        (2**64 - 1, "max code"),
    ],
)
def test_failure_response_round_trip(code: int, message: str) -> None:
    original = FailureResponse(code=code, message=message)
    encoded = original.encode()
    decoder = MessageDecoder(is_request=False)
    decoder.feed(encoded)
    decoded = decoder.decode()
    assert isinstance(decoded, FailureResponse)
    assert decoded.code == code
    assert decoded.message == message
    assert not decoder.is_poisoned


# ---- merged from test_column_name_label_symmetric.py ----
# RowsResponse column-name diagnostic labels must be byte-identical (lowercase
# "column name") across the encode and decode paths, so one monitoring match lifts
# both halves of the round-trip.


def test_encode_side_column_name_uses_lowercase_label() -> None:
    oversize = "x" * (MAX_COLUMN_NAME_SIZE + 1)
    with pytest.raises(EncodeError) as exc:
        RowsResponse(column_names=[oversize], rows=[]).encode_body()
    # Lowercase only — the title-case form must not slip back in.
    assert "column name" in str(exc.value)
    assert "Column name" not in str(exc.value)


def _build_rows_frame_with_oversize_col_name(name_size: int) -> bytes:
    """Hand-code an oversize column-name body (bypassing the encode-side cap) to drive
    the decode-side label. Body: uint64 column_count, then per column a padded
    NUL-terminated UTF-8 string, then the row terminator."""
    # decode_text scans for the NUL; with no terminator in max_size+1 bytes the
    # cap-exceeded diagnostic fires with the field label.
    payload = b"a" * name_size + b"\x00"
    pad = (-len(payload)) % 8
    text_bytes = payload + b"\x00" * pad
    # Trailing zero word passes the column-count vs remaining-body bounds check.
    return encode_uint64(1) + text_bytes + b"\x00" * 8


def test_decode_side_column_name_uses_lowercase_label() -> None:
    body = _build_rows_frame_with_oversize_col_name(MAX_COLUMN_NAME_SIZE + 1)
    with pytest.raises(DecodeError) as exc:
        RowsResponse.decode_body(body)
    assert "column name" in str(exc.value)
    assert "Column name" not in str(exc.value)


def test_encode_and_decode_labels_are_byte_identical() -> None:
    """The same lowercase token appears in both the encode and decode diagnostics."""
    oversize = "x" * (MAX_COLUMN_NAME_SIZE + 1)
    with pytest.raises(EncodeError) as enc_exc:
        RowsResponse(column_names=[oversize], rows=[]).encode_body()

    body = _build_rows_frame_with_oversize_col_name(MAX_COLUMN_NAME_SIZE + 1)
    with pytest.raises(DecodeError) as dec_exc:
        RowsResponse.decode_body(body)

    enc_text = str(enc_exc.value)
    dec_text = str(dec_exc.value)
    assert "column name" in enc_text
    assert "column name" in dec_text


# ---- merged from test_rows_response_inference_no_double_encode.py ----
# Pin: ``RowsResponse.encode_body``'s inference fallback path runs
# ``encode_value`` once per cell, not twice.
#
# Previously ``_get_row_types``'s inference branch called
# ``encode_value(v)[1]`` per cell — running the full encode pipeline
# (length-cap checks, BLOB materialisation, UTF-8 encode) just to extract
# the type tag from the returned tuple — and then ``encode_body`` ran
# ``encode_value(v, vtype)`` again to consume the bytes. For BLOB-heavy
# frames the cost was 2 * N * M ``encode_value`` calls instead of N * M;
# each large BLOB cell was materialised twice. The fix introduces
# ``_infer_value_type`` (type-only ladder, no encode work) and uses it
# on the inference path.
#
# The pin counts ``encode_value`` invocations on the inference path and
# asserts it equals one per cell, not two.


def test_inference_path_runs_encode_value_once_per_cell() -> None:
    """For a 3-row x 2-col inference frame, ``encode_value`` runs 6
    times total (one per cell), not 12 (the pre-fix double-encode
    shape). The type-only helper handles the inference pass; the
    actual emission still calls ``encode_value`` once per cell via
    ``encode_row_values`` in ``tuples.py``."""
    rows: list[list[WireValue]] = [[b"a", "x"], [b"b", "y"], [b"c", "z"]]
    msg = RowsResponse(column_names=["blob", "text"], rows=rows)

    from dqlitewire.types import encode_value as real_encode_value

    call_count = 0

    def counting_encode_value(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return real_encode_value(*args, **kwargs)

    # Patch the emission-side call site (in tuples.py); the inference
    # path no longer calls encode_value at all, so the count reflects
    # only the emission pass.
    with patch("dqlitewire.tuples.encode_value", side_effect=counting_encode_value):
        msg.encode_body()

    # 3 rows * 2 cols = 6 cells. Pre-fix: the inference pass also
    # called encode_value, doubling the count to 12. Post-fix: only
    # the emission pass remains.
    assert call_count == 6, f"expected 6 encode_value calls, got {call_count}"


def test_inference_path_round_trip_unchanged() -> None:
    """The fix changes the call count, not the wire bytes. Verify
    inference still picks the right types and the round-trip succeeds."""
    rows: list[list[WireValue]] = [[b"a"], [1], ["text"], [1.5], [True], [None]]
    msg = RowsResponse(column_names=["c"], rows=rows)
    encoded = msg.encode_body()
    decoded = RowsResponse.decode_body(encoded)
    # Inference picks BLOB for bytes, INTEGER for int (not BOOLEAN —
    # the bool is checked first per the helper's ladder), etc. The
    # decoded values reflect what the inference picked.
    assert decoded.rows[0] == [b"a"]
    assert decoded.rows[1] == [1]
    assert decoded.rows[2] == ["text"]
    assert decoded.rows[3] == [1.5]
    # bool infers to BOOLEAN, which round-trips as int 1 (SQLite stores
    # both as integer column value regardless of the tag) — documented
    # in encode_value's docstring.
    assert decoded.rows[4] == [1]
    assert decoded.rows[5] == [None]


def test_infer_value_type_helper_directly() -> None:
    """Direct unit coverage for the helper."""
    from dqlitewire.types import _infer_value_type

    assert _infer_value_type(None) == ValueType.NULL
    assert _infer_value_type(True) == ValueType.BOOLEAN
    assert _infer_value_type(False) == ValueType.BOOLEAN
    assert _infer_value_type(42) == ValueType.INTEGER
    assert _infer_value_type(1.5) == ValueType.FLOAT
    assert _infer_value_type("text") == ValueType.TEXT
    assert _infer_value_type(b"bytes") == ValueType.BLOB
    assert _infer_value_type(bytearray(b"ba")) == ValueType.BLOB
    assert _infer_value_type(memoryview(b"mv")) == ValueType.BLOB


# ---- merged from test_rows_response_inference_null_override.py ----
# Pin: ``RowsResponse._get_row_types`` applies the None→NULL override
# uniformly across all three type-selection paths (``row_types`` set,
# ``column_types`` set, or inference from values).
#
# The None→NULL override is critical because Go's per-row type header
# emits the NULL nibble when the value is None regardless of the declared
# column type. The override used to be skipped on the inference branch,
# relying on ``_infer_value_type(None) == ValueType.NULL`` to produce
# the right answer accidentally. A future helper that learned a
# typed-null shape would silently regress the inference path.


def test_none_in_row_with_row_types_set_overrides_to_null() -> None:
    row: list[WireValue] = [42, None]
    msg = RowsResponse(
        column_names=["a", "b"],
        row_types=[[ValueType.INTEGER, ValueType.TEXT]],
        rows=[row],
    )
    types = msg._get_row_types(0, row)
    assert types == [ValueType.INTEGER, ValueType.NULL]


def test_none_in_row_with_column_types_set_overrides_to_null() -> None:
    row: list[WireValue] = [42, None]
    msg = RowsResponse(
        column_names=["a", "b"],
        column_types=[ValueType.INTEGER, ValueType.TEXT],
        rows=[row],
    )
    types = msg._get_row_types(0, row)
    assert types == [ValueType.INTEGER, ValueType.NULL]


def test_none_in_row_with_inference_falls_through_to_null() -> None:
    """Inference branch: no ``column_types`` / ``row_types`` declared.
    The None→NULL override applies uniformly with the other branches
    (was previously implicit via ``_infer_value_type(None) == NULL``;
    now applied at the same site for defensive symmetry)."""
    row: list[WireValue] = [42, None]
    msg = RowsResponse(column_names=["a", "b"], rows=[row])
    types = msg._get_row_types(0, row)
    assert types == [ValueType.INTEGER, ValueType.NULL]


def test_no_none_values_does_not_change_types() -> None:
    """Negative pin: the override loop must NOT touch non-None cells."""
    row: list[WireValue] = [42, "x"]
    msg = RowsResponse(
        column_names=["a", "b"],
        column_types=[ValueType.INTEGER, ValueType.TEXT],
        rows=[row],
    )
    types = msg._get_row_types(0, row)
    assert types == [ValueType.INTEGER, ValueType.TEXT]


# ---- merged from test_rows_response_trailing_bytes_after_marker.py ----
# Pin: ``RowsResponse.decode_body`` rejects trailing bytes after the
# DONE / PART marker on the non-zero-column path, in parity with the
# zero-column fast path and with sibling decoders
# (``LeaderResponse``, ``FailureResponse``, ``ServersResponse``).
#
# A prior alignment cycle established this parity: previously the
# zero-column path raised on trailing bytes but the non-zero-column
# path returned immediately on the marker, silently consuming any
# trailing bytes via the body slice. A byte-replay against the same
# stray-trailing pattern produced different results between the two
# paths — a strict-decode posture inconsistency now closed.


def _build_one_row_one_column_body_with_trailer(trailer: bytes) -> bytes:
    """Construct a one-column / one-row RowsResponse body whose marker
    is followed by ``trailer`` bytes."""
    column_count = 1
    body = encode_uint64(column_count)
    # Column name (text + 8-byte alignment built into encode_text).
    body += encode_text("c0")
    # Row header: a single 8-byte word holding the type-nibble for one
    # INTEGER column. encode_value(1) returns (data, type_code).
    value_bytes, type_code = encode_value(1)
    assert type_code == ValueType.INTEGER.value
    # Row type header: byte 0 holds the type for column 0; word-padded.
    row_header = bytes([type_code]) + b"\x00" * 7
    body += row_header
    body += value_bytes
    # End-of-rows marker.
    body += encode_uint64(ROW_DONE_MARKER)
    body += trailer
    return body


def test_rows_response_rejects_trailing_byte_after_done_marker() -> None:
    body = _build_one_row_one_column_body_with_trailer(b"\x00")
    with pytest.raises(DecodeError, match=r"trailing bytes after DONE"):
        RowsResponse.decode_body(body)


def test_rows_response_rejects_trailing_word_after_done_marker() -> None:
    body = _build_one_row_one_column_body_with_trailer(b"\x00" * 8)
    with pytest.raises(DecodeError, match=r"trailing bytes after DONE"):
        RowsResponse.decode_body(body)


def test_rows_response_clean_done_still_decodes() -> None:
    """Regression guard: the strict check does not break the
    no-trailing-bytes happy path."""
    body = _build_one_row_one_column_body_with_trailer(b"")
    response = RowsResponse.decode_body(body)
    assert response.column_names == ["c0"]
    assert response.rows == [[1]]
    assert response.has_more is False


# ---- merged from test_max_column_count_cap.py ----
# ``MAX_COLUMN_COUNT`` is set to SQLite's documented column limit
# (``SQLITE_MAX_COLUMN = 2000``) so legitimate wide-table SELECT
# results decode while still rejecting absurd peer emissions.
#
# The C server emits ``sqlite3_column_count(stmt)`` as a uint64
# without cap (``query.c:111-120``); ``stmt.c:10``'s
# ``STMT__MAX_COLUMNS = (1 << 8) - 1 = 255`` macro is defined but
# never referenced. SQLite's compile-time default is 2000 (raisable
# to 32767 via ``SQLITE_MAX_COLUMN`` build flag); a wide-table
# SELECT against an analytics / feature-store schema legitimately
# crosses 255 columns.
#
# The per-name cap (``MAX_COLUMN_NAME_SIZE = 4096``) and the frame-
# envelope cap (default 64 MiB) already bound memory growth from the
# N × name allocation; this cap is defence-in-depth against
# pathological peer emissions, not the load-bearing memory bound.


def test_max_column_count_pinned_to_sqlite_default() -> None:
    """SQLite's documented default ``SQLITE_MAX_COLUMN`` is 2000."""
    assert MAX_COLUMN_COUNT == 2000


def test_rows_response_rejects_count_above_cap() -> None:
    body = encode_uint64(MAX_COLUMN_COUNT + 1)
    with pytest.raises(DecodeError, match="(?i)column count"):
        RowsResponse.decode_body(body)


def test_rows_response_accepts_count_at_cap() -> None:
    """A 2000-column rows response is well-formed and must not be
    rejected by the cap; it fails the body-size check instead
    because we only sent the count, not the column names."""
    body = encode_uint64(MAX_COLUMN_COUNT)
    with pytest.raises(DecodeError, match="exceeds maximum possible"):
        RowsResponse.decode_body(body)


def test_rows_response_accepts_count_above_old_255_cap() -> None:
    """Pin the regression-vs-old-cap shape: a 1500-column emission
    (legitimate wide table, above the prior 255 cap but below the
    new 2000 cap) must NOT trip the column-count cap. It still
    fails the body-size check below because we only sent the count,
    not the per-column name payload."""
    body = encode_uint64(1500)
    with pytest.raises(DecodeError, match="exceeds maximum possible"):
        RowsResponse.decode_body(body)


def test_rows_response_rejects_absurd_count() -> None:
    """A pathological emission (``column_count = 2^31``) must still
    be rejected so a hostile peer cannot inflate Python-side
    allocations."""
    body = encode_uint64(1 << 31)
    with pytest.raises(DecodeError, match="(?i)column count"):
        RowsResponse.decode_body(body)


def test_servers_response_uses_separate_cap() -> None:
    """``ServersResponse`` uses ``MAX_NODE_COUNT = 10_000``; the
    column cap does not apply. Pinning here is a sanity check that
    the cap constant was not accidentally inlined into an unrelated
    field."""
    assert MAX_COLUMN_COUNT < 10_000


def test_stmt_response_num_params_unaffected() -> None:
    """``StmtResponse.num_params`` uses ``MAX_PARAM_COUNT``
    (32_766) — verify the column cap tighten did not collide."""
    # 1000 params is fine — within MAX_PARAM_COUNT but well above
    # the column cap. The body needs db_id+stmt_id+num_params.
    body = encode_uint64(0) + encode_uint64(1) + encode_uint64(1000)
    # Not a real well-formed response but the num_params cap is what
    # we're pinning; it should NOT raise on 1000.
    try:
        StmtResponse.decode_body(body)
    except DecodeError as e:
        # Any decode error must NOT cite the column-count cap.
        assert "column count" not in str(e)


# ---- merged from test_welcome_response_zero_heartbeat_warning.py ----
# Pin: ``WelcomeResponse.decode_body`` emits a ``logger.warning``
# when ``heartbeat_timeout == 0``.
#
# The docstring explicitly calls a zero heartbeat "semantically
# ambiguous" / "misconfigured peer or non-conforming server" — that
# diagnostic content used to live only in source comments, so operators
# running a dqlite cluster with a misconfigured peer got no log signal.
#
# The wire layer keeps its permissive-accept contract (the decoder
# still returns the response; no DecodeError raised) but emits a
# single warning that surfaces the docstring's diagnostic content into
# the log stream. Aligns with the in-tree ``ServersResponse.decode_body``
# ``unknown_role_policy="warn"`` precedent.


def test_decode_body_zero_heartbeat_emits_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A zero heartbeat is accepted but produces a single
    logger.warning at decode time."""
    body = encode_uint64(0)
    with caplog.at_level(logging.WARNING, logger="dqlitewire.messages.responses"):
        resp = WelcomeResponse.decode_body(body)
    assert resp.heartbeat_timeout == 0
    # Single warning emitted.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].message
    # Diagnostic content from the docstring surfaces in the log line.
    assert "heartbeat_timeout=0" in msg or "heartbeat" in msg.lower()
    assert "15000" in msg or "non-conforming" in msg.lower() or "misconfig" in msg.lower()


def test_decode_body_default_heartbeat_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A legitimate heartbeat (e.g. 15000ms upstream default) does
    NOT trigger the warning."""
    body = encode_uint64(15000)
    with caplog.at_level(logging.WARNING, logger="dqlitewire.messages.responses"):
        resp = WelcomeResponse.decode_body(body)
    assert resp.heartbeat_timeout == 15000
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_decode_body_zero_heartbeat_is_still_accepted() -> None:
    """The warning is observability-only — the decoder still returns
    a valid WelcomeResponse with heartbeat_timeout=0 (preserves the
    documented permissive-accept contract)."""
    body = encode_uint64(0)
    resp = WelcomeResponse.decode_body(body)
    assert resp.heartbeat_timeout == 0
    assert resp.heartbeat_timeout_seconds == 0.0
