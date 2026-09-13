"""Tests for request message encoding/decoding."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from dqlitewire.constants import HEADER_SIZE, NodeRole, RequestType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.limits import MAX_DUMP_FILENAME_SIZE
from dqlitewire.messages.base import Header, Message
from dqlitewire.messages.requests import (
    AddRequest,
    AssignRequest,
    ClientRequest,
    ClusterRequest,
    DescribeRequest,
    DumpRequest,
    ExecRequest,
    ExecSqlRequest,
    FinalizeRequest,
    InterruptRequest,
    LeaderRequest,
    OpenRequest,
    PrepareRequest,
    QueryRequest,
    QuerySqlRequest,
    RemoveRequest,
    TransferRequest,
    WeightRequest,
    _ConnectRequest,
    _HeartbeatRequest,
)
from dqlitewire.types import encode_text, encode_uint32, encode_uint64


class TestHeaderReservedField:
    """The header's reserved/extra uint16: written as 0, ignored on decode."""

    def test_encoded_reserved_is_zero(self) -> None:
        cases = [LeaderRequest(), _HeartbeatRequest(timestamp=0), OpenRequest(name="db")]
        for msg in cases:
            encoded = msg.encode()
            header = Header.decode(encoded[:HEADER_SIZE])
            assert header.reserved == 0, f"{type(msg).__name__} emitted non-zero reserved"

    def test_decoded_reserved_is_zero_after_encode_decode(self) -> None:
        msg = LeaderRequest()
        header = Header.decode(msg.encode()[:HEADER_SIZE])
        assert header.reserved == 0

    def test_nonzero_reserved_is_carried_not_rejected(self) -> None:
        """go-dqlite ignores the field on decode; a future server may use it."""
        header = Header.decode(Header(size_words=1, msg_type=0, schema=0, reserved=0xBEEF).encode())
        assert header.reserved == 0xBEEF


class TestLeaderRequest:
    def test_encode_has_body(self) -> None:
        """LeaderRequest body must contain a reserved uint64 per Go spec."""
        msg = LeaderRequest()
        encoded = msg.encode()
        assert len(encoded) == HEADER_SIZE + 8  # Header + reserved uint64

    def test_header(self) -> None:
        msg = LeaderRequest()
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == RequestType.LEADER
        assert header.size_words == 1  # 1 word = 8 bytes reserved field

    def test_roundtrip(self) -> None:
        msg = LeaderRequest()
        encoded = msg.encode()
        decoded = LeaderRequest.decode_body(encoded[HEADER_SIZE:])
        assert isinstance(decoded, LeaderRequest)

    def test_decode_rejects_short_body(self) -> None:
        import pytest

        from dqlitewire.exceptions import DecodeError

        with pytest.raises(DecodeError, match="must be 8 bytes"):
            LeaderRequest.decode_body(b"")
        with pytest.raises(DecodeError, match="must be 8 bytes"):
            LeaderRequest.decode_body(b"\x00" * 7)

    def test_decode_rejects_extended_body(self) -> None:
        import pytest

        from dqlitewire.exceptions import DecodeError

        with pytest.raises(DecodeError, match="must be 8 bytes"):
            LeaderRequest.decode_body(b"\x00" * 16)

    def test_decode_rejects_nonzero_reserved(self) -> None:
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        with pytest.raises(DecodeError, match="reserved field must be 0"):
            LeaderRequest.decode_body(encode_uint64(1))


class TestClientRequest:
    def test_encode(self) -> None:
        msg = ClientRequest(client_id=12345)
        encoded = msg.encode()
        assert len(encoded) == HEADER_SIZE + 8  # Header + uint64

    def test_header(self) -> None:
        msg = ClientRequest(client_id=1)
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == RequestType.CLIENT
        assert header.size_words == 1

    def test_roundtrip(self) -> None:
        msg = ClientRequest(client_id=98765)
        encoded = msg.encode()
        decoded = ClientRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.client_id == 98765


class Test_HeartbeatRequest:
    def test_roundtrip(self) -> None:
        msg = _HeartbeatRequest(timestamp=1234567890)
        encoded = msg.encode()
        decoded = _HeartbeatRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.timestamp == 1234567890

    def test_not_in_public_request_registry(self) -> None:
        """``_HeartbeatRequest`` must stay private: no real upstream server
        accepts a type-2 frame, so the public ``REQUEST_TYPES`` registry
        and ``messages/__init__.py`` ``__all__`` do not export it.
        """
        import dqlitewire.messages as messages
        from dqlitewire.codec import REQUEST_TYPES

        assert RequestType.HEARTBEAT not in REQUEST_TYPES
        assert "HeartbeatRequest" not in getattr(messages, "__all__", [])
        assert not hasattr(messages, "HeartbeatRequest")


class TestOpenRequest:
    def test_encode(self) -> None:
        msg = OpenRequest(name="test.db", flags=0, vfs="")
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == RequestType.OPEN

    def test_roundtrip(self) -> None:
        msg = OpenRequest(name="mydb.sqlite", flags=6, vfs="unix")
        encoded = msg.encode()
        decoded = OpenRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.name == "mydb.sqlite"
        assert decoded.flags == 6
        assert decoded.vfs == "unix"

    def test_roundtrip_defaults(self) -> None:
        msg = OpenRequest(name="test")
        encoded = msg.encode()
        decoded = OpenRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.name == "test"
        assert decoded.flags == 0
        assert decoded.vfs == ""


class TestPrepareRequest:
    def test_roundtrip(self) -> None:
        msg = PrepareRequest(db_id=1, sql="SELECT * FROM users WHERE id = ?")
        encoded = msg.encode()
        decoded = PrepareRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 1
        assert decoded.sql == "SELECT * FROM users WHERE id = ?"

    def test_v1_schema_in_header(self) -> None:
        """V1 PrepareRequest sets schema=1 in the header for multi-statement support.

        Note: schema=1 is not used by the canonical Go client (go-dqlite).
        This is a feature supported by the C dqlite server but not exercised
        by the Go reference implementation.
        """
        msg = PrepareRequest(db_id=1, sql="SELECT 1; SELECT 2", schema=1)
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.schema == 1

    def test_default_schema_is_0_matching_go(self) -> None:
        """Default schema=0 matches Go's EncodePrepare which always uses schema=0."""
        msg = PrepareRequest(db_id=1, sql="SELECT 1")
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.schema == 0

    def test_rejects_invalid_schema(self) -> None:
        """PrepareRequest should reject schema values other than 0 or 1."""
        import pytest

        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="schema must be 0 or 1"):
            PrepareRequest(db_id=1, sql="SELECT 1", schema=2)

        with pytest.raises(EncodeError, match="schema must be 0 or 1"):
            PrepareRequest(db_id=1, sql="SELECT 1", schema=-1)

    def test_encode_body_caps_sql_at_decode_max_size(self) -> None:
        """Pin: encode/decode round-trip is symmetric on the SQL
        text field. ``encode_text`` accepts any byte length the outer
        frame admits; ``decode_text`` defaults to ``MAX_TEXT_VALUE_SIZE``.
        Without an explicit ``max_size`` on encode the outbound body
        could be larger than the inbound decoder accepts. Pin the
        rejection."""
        import pytest

        from dqlitewire.exceptions import EncodeError
        from dqlitewire.limits import MAX_TEXT_VALUE_SIZE

        oversize = "x" * (MAX_TEXT_VALUE_SIZE + 1)
        with pytest.raises(EncodeError):
            PrepareRequest(db_id=1, sql=oversize).encode_body()


class TestExecRequest:
    def test_schema_v0_for_small_params(self) -> None:
        """Go uses schema=0 (V0 uint8 count) when params <= 255."""
        msg = ExecRequest(db_id=1, stmt_id=2, params=[1, 2, 3])
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.schema == 0

    def test_encode_no_params(self) -> None:
        msg = ExecRequest(db_id=1, stmt_id=2, params=[])
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == RequestType.EXEC

    def test_body_structure(self) -> None:
        from dqlitewire.types import decode_uint32

        msg = ExecRequest(db_id=1, stmt_id=2, params=[])
        body = msg.encode_body()
        assert len(body) == 8
        assert decode_uint32(body[:4]) == 1  # db_id
        assert decode_uint32(body[4:8]) == 2  # stmt_id

    def test_roundtrip_with_params(self) -> None:
        """Parameters must survive encode/decode round-trip."""
        msg = ExecRequest(db_id=1, stmt_id=2, params=[42, "hello", 3.14])
        encoded = msg.encode()
        decoded = ExecRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 1
        assert decoded.stmt_id == 2
        assert len(decoded.params) == 3
        assert decoded.params[0] == 42
        assert decoded.params[1] == "hello"
        assert decoded.params[2] == pytest.approx(3.14)

    def test_schema_v0_for_small_params_in_header(self) -> None:
        """Go uses schema=0 for <= 255 params, schema=1 for > 255."""
        msg = ExecRequest(db_id=1, stmt_id=2, params=[42])
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.schema == 0

    def test_params_use_v0_uint8_count_for_small_lists(self) -> None:
        """Parameters use V0 format (uint8 count) when <= 255 params."""
        msg = ExecRequest(db_id=1, stmt_id=2, params=[42])
        body = msg.encode_body()
        # After db_id (4) + stmt_id (4) = 8, params start
        params_data = body[8:]
        # V0: first byte is uint8 count
        assert params_data[0] == 1

    def test_roundtrip_v1_more_than_255_params(self) -> None:
        """ExecRequest with >255 params must roundtrip correctly via V1 schema."""
        from dqlitewire.codec import decode_message, encode_message

        params = list(range(256))
        msg = ExecRequest(db_id=1, stmt_id=2, params=params)
        encoded = encode_message(msg)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, ExecRequest)
        assert decoded.db_id == 1
        assert decoded.stmt_id == 2
        assert list(decoded.params) == params


class TestQueryRequest:
    def test_encode_no_params(self) -> None:
        msg = QueryRequest(db_id=1, stmt_id=2, params=[])
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == RequestType.QUERY

    def test_roundtrip_with_params(self) -> None:
        """Parameters must survive encode/decode round-trip."""
        msg = QueryRequest(db_id=1, stmt_id=2, params=[100, "world"])
        encoded = msg.encode()
        decoded = QueryRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 1
        assert decoded.stmt_id == 2
        assert decoded.params == [100, "world"]

    def test_roundtrip_v1_more_than_255_params(self) -> None:
        """QueryRequest with >255 params must roundtrip correctly via V1 schema."""
        from dqlitewire.codec import decode_message, encode_message

        params = list(range(256))
        msg = QueryRequest(db_id=1, stmt_id=2, params=params)
        encoded = encode_message(msg)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, QueryRequest)
        assert decoded.db_id == 1
        assert decoded.stmt_id == 2
        assert list(decoded.params) == params


class TestFinalizeRequest:
    def test_roundtrip(self) -> None:
        msg = FinalizeRequest(db_id=5, stmt_id=10)
        encoded = msg.encode()
        decoded = FinalizeRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 5
        assert decoded.stmt_id == 10


class TestExecSqlRequest:
    def test_roundtrip_no_params(self) -> None:
        msg = ExecSqlRequest(db_id=1, sql="CREATE TABLE test (id INTEGER)")
        encoded = msg.encode()
        decoded = ExecSqlRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 1
        assert decoded.sql == "CREATE TABLE test (id INTEGER)"

    def test_roundtrip_with_params(self) -> None:
        """Parameters must survive encode/decode round-trip."""
        msg = ExecSqlRequest(db_id=1, sql="INSERT INTO t VALUES(?)", params=[42])
        encoded = msg.encode()
        decoded = ExecSqlRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 1
        assert decoded.sql == "INSERT INTO t VALUES(?)"
        assert decoded.params == [42]

    def test_roundtrip_v1_more_than_255_params(self) -> None:
        """ExecSqlRequest with >255 params must roundtrip correctly via V1 schema."""
        from dqlitewire.codec import decode_message, encode_message

        params = list(range(256))
        msg = ExecSqlRequest(db_id=1, sql="SELECT 1", params=params)
        encoded = encode_message(msg)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, ExecSqlRequest)
        assert list(decoded.params) == params


class TestQuerySqlRequest:
    def test_roundtrip(self) -> None:
        msg = QuerySqlRequest(db_id=1, sql="SELECT 1")
        encoded = msg.encode()
        decoded = QuerySqlRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 1
        assert decoded.sql == "SELECT 1"

    def test_roundtrip_with_params(self) -> None:
        """Parameters must survive encode/decode round-trip."""
        msg = QuerySqlRequest(db_id=1, sql="SELECT * FROM t WHERE id=?", params=[99])
        encoded = msg.encode()
        decoded = QuerySqlRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 1
        assert decoded.sql == "SELECT * FROM t WHERE id=?"
        assert decoded.params == [99]

    def test_roundtrip_v1_more_than_255_params(self) -> None:
        """QuerySqlRequest with >255 params must roundtrip correctly via V1 schema."""
        from dqlitewire.codec import decode_message, encode_message

        params = list(range(256))
        msg = QuerySqlRequest(db_id=1, sql="SELECT 1", params=params)
        encoded = encode_message(msg)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, QuerySqlRequest)
        assert list(decoded.params) == params


class TestInterruptRequest:
    def test_roundtrip(self) -> None:
        msg = InterruptRequest(db_id=42)
        encoded = msg.encode()
        decoded = InterruptRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 42


class Test_ConnectRequest:
    """``_ConnectRequest`` is a private class exposed only to mock /
    golden-byte harnesses — upstream ``gateway.c`` rejects type-11
    frames with ``DQLITE_PARSE``. The public ``REQUEST_TYPES`` registry
    does not include it; see ``test_not_in_public_request_registry``.
    """

    def test_roundtrip(self) -> None:
        msg = _ConnectRequest(node_id=3, address="node3:9001")
        encoded = msg.encode()
        decoded = _ConnectRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.node_id == 3
        assert decoded.address == "node3:9001"

    def test_type_code_is_11(self) -> None:
        msg = _ConnectRequest(node_id=1, address="localhost:9001")
        assert msg.MSG_TYPE == 11

    def test_not_in_public_request_registry(self) -> None:
        """No real upstream server accepts a type-11 gateway frame, so
        the public ``REQUEST_TYPES`` registry and
        ``messages/__init__.py`` ``__all__`` do not export it.
        """
        import dqlitewire.messages as messages
        from dqlitewire.codec import REQUEST_TYPES

        assert RequestType.CONNECT not in REQUEST_TYPES
        assert "ConnectRequest" not in getattr(messages, "__all__", [])
        assert not hasattr(messages, "ConnectRequest")


class TestAddRequest:
    def test_roundtrip(self) -> None:
        msg = AddRequest(node_id=2, address="node2:9001")
        encoded = msg.encode()
        decoded = AddRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.node_id == 2
        assert decoded.address == "node2:9001"


class TestAssignRequest:
    def test_roundtrip(self) -> None:
        msg = AssignRequest(node_id=1, role=2)
        encoded = msg.encode()
        decoded = AssignRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.node_id == 1
        assert decoded.role == 2

    def test_decode_promote_single_field_maps_to_voter(self) -> None:
        """Legacy PROMOTE request shares type code 13 but has only
        node_id. The Go/C server distinguishes PROMOTE from ASSIGN by
        body size:

        - PROMOTE: 1 word (8 bytes) = just node_id
        - ASSIGN: 2 words (16 bytes) = node_id + role

        decode_body maps the legacy PROMOTE shape to
        ``NodeRole.VOTER`` (PROMOTE elevates a non-voter to voter,
        per upstream semantics) so the dataclass round-trips through
        ``encode_body`` cleanly. Storing ``role=None`` from the
        legacy decode would mean encode_body raises EncodeError for
        any decoded legacy frame.
        """
        from dqlitewire.constants import NodeRole
        from dqlitewire.types import encode_uint64

        promote_body = encode_uint64(5)
        decoded = AssignRequest.decode_body(promote_body)
        assert decoded.node_id == 5
        assert decoded.role == NodeRole.VOTER

    def test_encode_without_role_raises_encode_error(self) -> None:
        """Constructing AssignRequest with bare role=None now raises
        EncodeError at construction time so an accidental omission
        (typo: forgetting the role= kwarg) doesn't silently downgrade
        to the legacy 1-word PROMOTE shape and surface only at
        encode."""
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="(?i)role"):
            AssignRequest(node_id=1)

    def test_encode_body_legacy_explicit_opt_in(self) -> None:
        """encode_body_legacy() is the explicit opt-in for the
        1-word PROMOTE wire shape; callers asking for the legacy
        encode flag _legacy_intent=True at construction."""
        from dqlitewire.types import encode_uint64

        msg = AssignRequest(node_id=42, role=None, _legacy_intent=True)
        body = msg.encode_body_legacy()
        assert body == encode_uint64(42)

    def test_legacy_decode_encode_byte_identical(self) -> None:
        """Pin: a captured 8-byte legacy PROMOTE body round-trips
        byte-identically through ``decode_body → encode_body_legacy``.
        ``encode_body`` itself rejects ``role=None`` by design; the
        legacy round-trip path uses ``encode_body_legacy`` to
        reproduce the captured shape exactly.
        """
        from dqlitewire.types import encode_uint64

        captured_legacy = encode_uint64(42)
        decoded = AssignRequest.decode_body(captured_legacy)
        assert decoded.encode_body_legacy() == captured_legacy

    def test_decode_legacy_then_encode_body_yields_modern_16_byte_assign(self) -> None:
        """Pin: the documented one-way upgrade.

        Decode a 1-word PROMOTE body → role defaults to VOTER (the
        upstream-semantic of PROMOTE). Re-encoding via
        ``encode_body()`` (the modern path) produces the 2-word
        (16-byte) ASSIGN shape — NOT the legacy 1-word body.

        The docstring on ``AssignRequest`` is the spec; without
        this pin a regression that defaulted ``role=None`` on
        legacy decode (or kept ``encode_body`` emitting the legacy
        shape) would not surface in unit CI.
        """
        from dqlitewire.constants import NodeRole
        from dqlitewire.types import encode_uint64

        legacy_body = encode_uint64(7)  # 8 bytes, just node_id
        decoded = AssignRequest.decode_body(legacy_body)
        # Docstring promises VOTER (the upstream-semantic of PROMOTE).
        assert decoded.role == NodeRole.VOTER

        re_encoded = decoded.encode_body()
        # Modern shape: 2 words (16 bytes).
        assert len(re_encoded) == 16, (
            "encode_body() of a legacy-decoded AssignRequest must "
            "produce the modern 2-word (16-byte) ASSIGN shape. The "
            "docstring documents this as a deliberate one-way upgrade."
        )

    def test_encode_with_role_succeeds(self) -> None:
        """Encoding AssignRequest with role goes through cleanly."""
        msg = AssignRequest(node_id=1, role=0)
        encoded = msg.encode()
        # 16-byte body (modern ASSIGN).
        assert len(encoded[HEADER_SIZE:]) == 16


class TestAssignRequestDecodeRoleValidation:
    """The 16-byte ASSIGN branch narrows the raw uint64 role through
    ``NodeRole(raw)``. A value outside VOTER/STANDBY/SPARE must surface
    as ``DecodeError("AssignRequest: unknown role N")`` rather than
    silently constructing a dataclass with a raw int. Mirrors the
    sibling ``ServersResponse`` coverage at
    ``tests/test_messages_responses.py`` and the symmetric
    construction-side validator on ``AssignRequest.__post_init__``.
    """

    @pytest.mark.parametrize("bad_role", [4, 999, 0xFFFFFFFFFFFFFFFF])
    def test_decode_rejects_unknown_role(self, bad_role: int) -> None:
        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        body = encode_uint64(1) + encode_uint64(bad_role)  # 16-byte ASSIGN
        with pytest.raises(DecodeError, match=f"unknown role {bad_role}"):
            AssignRequest.decode_body(body)

    def test_decode_accepts_known_roles(self) -> None:
        from dqlitewire.constants import NodeRole
        from dqlitewire.types import encode_uint64

        for role in NodeRole:
            decoded = AssignRequest.decode_body(encode_uint64(1) + encode_uint64(role))
            assert decoded.role == role

    def test_post_init_rejects_unknown_role_int(self) -> None:
        """Construction-side narrowing: passing a raw int role that is
        not a known ``NodeRole`` value must surface as
        ``EncodeError("AssignRequest: unknown role N")``. Sibling pin
        to the decode-side check above; the construction path is
        what tests / mock servers / direct callers exercise."""
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="unknown role 999"):
            AssignRequest(node_id=1, role=999)


class TestDescribeRequestStrictLength:
    """``DescribeRequest.decode_body`` rejects bodies of any length
    other than exactly 8 bytes. Pinned alongside the other strict-
    length decoders in ``test_decoder_strict_length.py`` would also
    work; placing it here keeps the per-message-class test cohesion."""

    def test_decode_body_rejects_short_body(self) -> None:
        from dqlitewire.messages.requests import DescribeRequest

        with pytest.raises(DecodeError, match="must be 8 bytes"):
            DescribeRequest.decode_body(b"")

    def test_decode_body_rejects_wrong_length(self) -> None:
        from dqlitewire.messages.requests import DescribeRequest

        with pytest.raises(DecodeError, match="must be 8 bytes"):
            DescribeRequest.decode_body(b"\x00" * 7)


class TestRemoveRequest:
    def test_roundtrip(self) -> None:
        msg = RemoveRequest(node_id=3)
        encoded = msg.encode()
        decoded = RemoveRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.node_id == 3


class TestDumpRequest:
    def test_roundtrip(self) -> None:
        msg = DumpRequest(name="mydb")
        encoded = msg.encode()
        decoded = DumpRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.name == "mydb"


class TestClusterRequest:
    def test_roundtrip(self) -> None:
        msg = ClusterRequest(format=1)
        encoded = msg.encode()
        decoded = ClusterRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.format == 1

    def test_default_format_is_v1(self) -> None:
        """Go client defaults to DQLITE_REQUEST_CLUSTER_FORMAT_V1 (1)."""
        msg = ClusterRequest()
        assert msg.format == 1

    def test_format_v0_rejected(self) -> None:
        """120: V0 cluster format not implemented by ServersResponse decoder."""
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="format=0.*not implemented"):
            ClusterRequest(format=0)

    def test_decode_format_v0_accepted(self) -> None:
        """A relaying proxy / mock server / captured-traffic replay
        tool may need to round-trip a V0 ClusterRequest. Decode-side
        accepts V0 so the byte shape is preserved; construction-side
        and encode-side still reject V0 because this client's
        outbound shape is V1-only (the matched ServersResponse
        decoder reads the role fields)."""
        from dqlitewire.types import encode_uint64

        body = encode_uint64(0)
        decoded = ClusterRequest.decode_body(body)
        assert decoded.format == 0

    def test_re_encode_decoded_v0_round_trips_byte_identical(self) -> None:
        """Decode-then-encode of a V0 ClusterRequest now round-trips
        byte-identically: the ``_decoded`` sentinel exempts the
        encode-time reject so proxy / replay / capture-replay tooling
        can route V0 traffic through the dataclass. Fresh construction
        with ``format=0`` is still rejected at construction (see
        ``test_unknown_format_rejected_in_constructor`` and
        ``test_fresh_construct_v0_still_rejected``)."""
        from dqlitewire.types import encode_uint64

        body = encode_uint64(0)
        decoded = ClusterRequest.decode_body(body)
        assert decoded.encode_body() == body

    def test_fresh_construct_v0_still_rejected(self) -> None:
        """Construction-time rejection of a bare ``format=0`` is
        unchanged: the V0 escape hatch requires the ``_decoded``
        sentinel from ``decode_body``."""
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="V0"):
            ClusterRequest(format=0)

    @pytest.mark.parametrize("fmt", [2, 3, 255, 0xFFFFFFFFFFFFFFFF])
    def test_unknown_format_rejected_in_constructor(self, fmt: int) -> None:
        # Upstream defines only V0=0 and V1=1. Anything else is
        # undefined and rejected client-side so callers see a local
        # EncodeError instead of a confusing server-side failure.
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="format must be 0"):
            ClusterRequest(format=fmt)

    @pytest.mark.parametrize("fmt", [2, 3, 255, 0xFFFFFFFFFFFFFFFF])
    def test_decode_unknown_format_raises_decode_error(self, fmt: int) -> None:
        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        body = encode_uint64(fmt)
        with pytest.raises(DecodeError, match="format must be 0"):
            ClusterRequest.decode_body(body)

    def test_format_v1_positive_case(self) -> None:
        # Pin the lower edge of the legal set so a future refactor
        # that narrows past {1} would fail this test too.
        msg = ClusterRequest(format=1)
        assert msg.format == 1
        encoded = msg.encode()
        decoded = ClusterRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.format == 1


class TestTransferRequest:
    def test_roundtrip(self) -> None:
        msg = TransferRequest(target_node_id=2)
        encoded = msg.encode()
        decoded = TransferRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.target_node_id == 2


class TestDescribeRequest:
    def test_roundtrip(self) -> None:
        msg = DescribeRequest(format=0)
        encoded = msg.encode()
        decoded = DescribeRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.format == 0


class TestWeightRequest:
    def test_roundtrip(self) -> None:
        msg = WeightRequest(weight=100)
        encoded = msg.encode()
        decoded = WeightRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.weight == 100


class TestParamsTupleWordAlignment:
    """Verify that params tuples start at word-aligned offsets in all message types.

    The params tuple padding calculation uses the relative header length
    (count_size + num_types) rather than the absolute buffer offset. This
    produces correct padding only when the params tuple starts at a
    word-aligned offset (multiple of 8) within the message body.

    This test ensures the assumption holds for every message type that
    embeds a params tuple, so that any future protocol change that violates
    it will be caught immediately.
    """

    def _body_offset_before_params(self, msg_class: type, **kwargs: object) -> int:
        """Calculate the byte offset where the params tuple begins in the body."""
        from dqlitewire.types import encode_text, encode_uint32, encode_uint64

        if msg_class is ExecRequest or msg_class is QueryRequest:
            # Body: uint32 db_id + uint32 stmt_id = 8 bytes
            return len(encode_uint32(0)) + len(encode_uint32(0))
        if msg_class is ExecSqlRequest or msg_class is QuerySqlRequest:
            # Body: uint64 db_id + text sql
            sql = kwargs.get("sql", "SELECT 1")
            assert isinstance(sql, str)
            return len(encode_uint64(0)) + len(encode_text(sql))
        raise ValueError(f"Unknown message class: {msg_class}")

    def test_exec_request_params_at_word_boundary(self) -> None:
        offset = self._body_offset_before_params(ExecRequest)
        assert offset % 8 == 0, f"ExecRequest params start at offset {offset}, not word-aligned"

    def test_query_request_params_at_word_boundary(self) -> None:
        offset = self._body_offset_before_params(QueryRequest)
        assert offset % 8 == 0, f"QueryRequest params start at offset {offset}, not word-aligned"

    def test_exec_sql_request_params_at_word_boundary(self) -> None:
        # Text encoding always pads to word boundary, so any SQL string works
        for sql in ["SELECT 1", "X", "", "SELECT * FROM very_long_table_name WHERE id = ?"]:
            offset = self._body_offset_before_params(ExecSqlRequest, sql=sql)
            assert offset % 8 == 0, (
                f"ExecSqlRequest params start at offset {offset} for sql={sql!r}, not word-aligned"
            )

    def test_query_sql_request_params_at_word_boundary(self) -> None:
        for sql in ["SELECT 1", "X", "", "SELECT * FROM very_long_table_name WHERE id = ?"]:
            offset = self._body_offset_before_params(QuerySqlRequest, sql=sql)
            assert offset % 8 == 0, (
                f"QuerySqlRequest params start at offset {offset} for sql={sql!r}, not word-aligned"
            )


class TestRequestFieldValidation:
    """Request fields should be validated at construction time, not just at encode time.

    Construction-time validation and encode-time validation share a single
    helper in ``types.py`` so both raise ``EncodeError`` with the same
    message shape. Callers previously catching ``TypeError``/``ValueError``
    around construction need to widen to ``EncodeError``.
    """

    def test_negative_uint32_field_rejected(self) -> None:
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="db_id"):
            ExecRequest(db_id=-1, stmt_id=0, params=[])

    def test_overflow_uint32_field_rejected(self) -> None:
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="stmt_id"):
            FinalizeRequest(db_id=0, stmt_id=2**32)

    def test_negative_uint64_field_rejected(self) -> None:
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="client_id"):
            ClientRequest(client_id=-1)

    def test_overflow_uint64_field_rejected(self) -> None:
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="timestamp"):
            _HeartbeatRequest(timestamp=2**64)

    def test_valid_values_accepted(self) -> None:
        """Valid values should not raise."""
        ExecRequest(db_id=0, stmt_id=0, params=[])
        ExecRequest(db_id=2**32 - 1, stmt_id=2**32 - 1, params=[])
        ClientRequest(client_id=0)
        ClientRequest(client_id=2**64 - 1)
        _HeartbeatRequest(timestamp=0)
        OpenRequest(name="test.db", flags=0, vfs="")

    def test_exec_sql_accepts_large_db_id(self) -> None:
        """134: ExecSqlRequest uses uint64 db_id, accepting values > uint32 max."""
        msg = ExecSqlRequest(db_id=2**32, sql="SELECT 1")
        assert msg.db_id == 2**32

    def test_exec_rejects_large_db_id(self) -> None:
        """134: ExecRequest uses uint32 db_id, rejecting values > uint32 max."""
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="db_id"):
            ExecRequest(db_id=2**32, stmt_id=0)

    def test_bool_rejected_for_uint32_db_id(self) -> None:
        """Bool must not be silently accepted as a uint32 field."""
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="db_id must be int"):
            ExecRequest(db_id=True, stmt_id=0)

    def test_bool_rejected_for_uint32_stmt_id(self) -> None:
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="stmt_id must be int"):
            ExecRequest(db_id=0, stmt_id=False)

    def test_bool_rejected_for_uint64_client_id(self) -> None:
        """Bool must not be silently accepted as a uint64 field."""
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="client_id must be int"):
            ClientRequest(client_id=True)

    def test_construction_and_encode_raise_same_exception_class(self) -> None:
        """Construction-time range checks and direct ``encode_uintN`` both
        raise ``EncodeError`` now that the helpers share a single
        implementation in ``types.py``.
        """
        from dqlitewire.exceptions import EncodeError
        from dqlitewire.types import encode_uint32, encode_uint64

        with pytest.raises(EncodeError):
            encode_uint64(-1)
        with pytest.raises(EncodeError):
            encode_uint32(-1)
        with pytest.raises(EncodeError):
            ClientRequest(client_id=-1)
        with pytest.raises(EncodeError):
            ExecRequest(db_id=-1, stmt_id=0, params=[])


class TestParamsBodySchemaRoundtrip:
    """Upstream C clients emit schema=1 for Exec/Query*Request
    unconditionally, regardless of param count. Decoding a small-param
    schema=1 body and re-encoding must be byte-identical, so a proxy
    or mock server that round-trips the bytes faithfully does not
    downgrade the schema bit seen over the wire.
    """

    @pytest.mark.parametrize(
        "cls_name",
        ["ExecRequest", "QueryRequest"],
    )
    def test_prepared_schema_1_small_params_roundtrip(self, cls_name: str) -> None:
        """ExecRequest / QueryRequest: 4-byte db_id + 4-byte stmt_id +
        PARAMS32 body with 3 params. Construct with schema=1 explicitly,
        then verify decode → re-encode == original bytes.
        """
        from dqlitewire.codec import encode_message
        from dqlitewire.messages import ExecRequest, QueryRequest

        classes = {"ExecRequest": ExecRequest, "QueryRequest": QueryRequest}
        cls = classes[cls_name]

        from dqlitewire.codec import decode_message

        # Manually force schema=1 by setting _decoded_schema on the
        # source message.
        original = cls(db_id=1, stmt_id=2, params=[42, "hello", None], _decoded_schema=1)
        original_bytes = encode_message(original)

        # Header schema byte is the 6th byte (after 4-byte size_words
        # + 1-byte msg_type). Confirm the wire reflects schema=1.
        assert original_bytes[5] == 1, "expected schema=1 in the header"

        decoded = decode_message(original_bytes, is_request=True)
        assert isinstance(decoded, cls)
        # ``cls`` is a TypeVar holding one of two request types; both
        assert list(decoded.params) == list(original.params)

        re_encoded = encode_message(decoded)
        assert re_encoded == original_bytes

    @pytest.mark.parametrize(
        "cls_name",
        ["ExecSqlRequest", "QuerySqlRequest"],
    )
    def test_sql_schema_1_small_params_roundtrip(self, cls_name: str) -> None:
        from dqlitewire.codec import decode_message, encode_message
        from dqlitewire.messages import ExecSqlRequest, QuerySqlRequest

        classes = {"ExecSqlRequest": ExecSqlRequest, "QuerySqlRequest": QuerySqlRequest}
        cls = classes[cls_name]

        original = cls(db_id=1, sql="SELECT 1", params=[1, 2, 3], _decoded_schema=1)
        original_bytes = encode_message(original)
        assert original_bytes[5] == 1

        decoded = decode_message(original_bytes, is_request=True)
        assert isinstance(decoded, cls)
        assert list(decoded.params) == list(original.params)
        assert decoded.sql == "SELECT 1"

        re_encoded = encode_message(decoded)
        assert re_encoded == original_bytes

    def test_default_construction_uses_heuristic(self) -> None:
        """When a caller constructs a fresh ExecRequest without a
        decoded schema hint, the count heuristic still applies: ≤255
        params → schema=0.
        """
        from dqlitewire.messages import ExecRequest

        msg = ExecRequest(db_id=1, stmt_id=2, params=[1, 2, 3])
        assert msg._get_schema() == 0

    def test_large_params_force_schema_1_without_hint(self) -> None:
        from dqlitewire.messages import ExecRequest

        msg = ExecRequest(db_id=1, stmt_id=2, params=list(range(300)))
        assert msg._get_schema() == 1


class TestDecodedSchemaConstructionValidation:
    """``_decoded_schema`` is a private round-trip hint used by decoders
    so re-encode is byte-identical even when the count heuristic would
    otherwise downgrade the schema. Misuse was previously deferred until
    encode time; catch it at construction so the error message names the
    field and fires from the caller's frame.
    """

    @pytest.mark.parametrize(
        "cls_name",
        ["ExecRequest", "QueryRequest", "ExecSqlRequest", "QuerySqlRequest"],
    )
    def test_rejects_schema_values_other_than_0_1_none(self, cls_name: str) -> None:
        import dqlitewire.messages as m

        cls = getattr(m, cls_name)
        kwargs: dict[str, object] = {"db_id": 1, "_decoded_schema": 2}
        if cls_name in ("ExecRequest", "QueryRequest"):
            kwargs["stmt_id"] = 1
        else:
            kwargs["sql"] = "SELECT 1"
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="_decoded_schema"):
            cls(**kwargs)

    @pytest.mark.parametrize(
        "cls_name",
        ["ExecRequest", "QueryRequest", "ExecSqlRequest", "QuerySqlRequest"],
    )
    def test_rejects_schema_0_with_more_than_255_params(self, cls_name: str) -> None:
        import dqlitewire.messages as m

        cls = getattr(m, cls_name)
        kwargs: dict[str, object] = {
            "db_id": 1,
            "_decoded_schema": 0,
            "params": [None] * 256,
        }
        if cls_name in ("ExecRequest", "QueryRequest"):
            kwargs["stmt_id"] = 1
        else:
            kwargs["sql"] = "SELECT 1"
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="255 parameters"):
            cls(**kwargs)

    @pytest.mark.parametrize(
        "cls_name",
        ["ExecRequest", "QueryRequest", "ExecSqlRequest", "QuerySqlRequest"],
    )
    def test_accepts_schema_1_with_many_params(self, cls_name: str) -> None:
        import dqlitewire.messages as m

        cls = getattr(m, cls_name)
        kwargs: dict[str, object] = {
            "db_id": 1,
            "_decoded_schema": 1,
            "params": [None] * 256,
        }
        if cls_name in ("ExecRequest", "QueryRequest"):
            kwargs["stmt_id"] = 1
        else:
            kwargs["sql"] = "SELECT 1"
        # Must not raise.
        cls(**kwargs)

    @pytest.mark.parametrize(
        "cls_name",
        ["ExecRequest", "QueryRequest", "ExecSqlRequest", "QuerySqlRequest"],
    )
    def test_rejects_schema_1_with_more_than_max_param_count(self, cls_name: str) -> None:
        """Mirror the V0 schema's construction-time cap: V1 schema with
        more than ``MAX_PARAM_COUNT`` (32766) params must reject at
        construction with an actionable EncodeError naming the field,
        not surface deep inside ``encode_params_tuple`` at first
        encode. Pre-fix, a 40k-param V1-schema request constructed
        successfully and only failed inside the tuple encoder."""
        import dqlitewire.messages as m
        from dqlitewire.limits import MAX_PARAM_COUNT

        cls = getattr(m, cls_name)
        kwargs: dict[str, object] = {
            "db_id": 1,
            "_decoded_schema": 1,
            "params": [None] * (MAX_PARAM_COUNT + 1),
        }
        if cls_name in ("ExecRequest", "QueryRequest"):
            kwargs["stmt_id"] = 1
        else:
            kwargs["sql"] = "SELECT 1"
        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match=r"_decoded_schema=1"):
            cls(**kwargs)


class TestDecodeBodySchemaGuard:
    """Direct ``decode_body(body, schema=N)`` invocations (bypassing the
    codec) must surface a message-class-named DecodeError for schema>1
    rather than propagating the generic ``decode_params_tuple`` error.
    """

    def test_prepare_request_rejects_schema_2(self) -> None:
        body = encode_uint64(1) + encode_text("SELECT 1")
        with pytest.raises(DecodeError, match="PrepareRequest unsupported schema"):
            PrepareRequest.decode_body(body, schema=2)

    def test_exec_request_rejects_schema_2(self) -> None:
        body = encode_uint32(1) + encode_uint32(2) + b"\x00" * 8
        with pytest.raises(DecodeError, match="ExecRequest unsupported schema"):
            ExecRequest.decode_body(body, schema=2)

    def test_query_request_rejects_schema_2(self) -> None:
        body = encode_uint32(1) + encode_uint32(2) + b"\x00" * 8
        with pytest.raises(DecodeError, match="QueryRequest unsupported schema"):
            QueryRequest.decode_body(body, schema=2)

    def test_exec_sql_request_rejects_schema_2(self) -> None:
        body = encode_uint64(1) + encode_text("SELECT 1") + b"\x00" * 8
        with pytest.raises(DecodeError, match="ExecSqlRequest unsupported schema"):
            ExecSqlRequest.decode_body(body, schema=2)

    def test_query_sql_request_rejects_schema_2(self) -> None:
        body = encode_uint64(1) + encode_text("SELECT 1") + b"\x00" * 8
        with pytest.raises(DecodeError, match="QuerySqlRequest unsupported schema"):
            QuerySqlRequest.decode_body(body, schema=2)

    def test_open_request_rejects_schema_2(self) -> None:
        """OpenRequest's upstream handler uses the ``START_V0`` (i.e.
        ``INIT_V0``) macro in ``gateway.c::handle_open`` which rejects
        any ``req->schema != 0`` with ``DQLITE_PARSE`` before calling
        ``request_open__decode``. The Python decoder must mirror that
        narrowing so direct callers (tests, mock-server harnesses,
        captured-traffic replay) cannot silently bypass the
        wire-format gate the dispatcher otherwise enforces."""
        body = encode_text("main") + b"\x01\x00\x00\x00\x00\x00\x00\x00" + encode_text("")
        with pytest.raises(DecodeError, match="OpenRequest unsupported schema"):
            OpenRequest.decode_body(body, schema=2)


# ---- merged from test_admin_request_byte_identity.py ----
# Admin-request encoders are byte-identity round-trippable; field-equality
# round-trips miss encoder/decoder asymmetries (e.g. spurious padding).

# Lambdas wrap constructors so parametrise IDs stay stable across pytest
# versions (dataclass __repr__ changes would otherwise leak into the ID).
_CASES: list[tuple[str, Callable[[], Message]]] = [
    ("FinalizeRequest-min", lambda: FinalizeRequest(db_id=0, stmt_id=0)),
    ("FinalizeRequest-max", lambda: FinalizeRequest(db_id=0xFFFFFFFF, stmt_id=0xFFFFFFFF)),
    ("InterruptRequest-min", lambda: InterruptRequest(db_id=0)),
    ("InterruptRequest-max", lambda: InterruptRequest(db_id=0xFFFFFFFFFFFFFFFF)),
    ("AddRequest-empty-addr", lambda: AddRequest(node_id=1, address="")),
    ("AddRequest-1char", lambda: AddRequest(node_id=1, address="a")),
    ("AddRequest-7char-exact-word", lambda: AddRequest(node_id=1, address="a" * 7)),
    ("AddRequest-8char-forces-pad", lambda: AddRequest(node_id=1, address="a" * 8)),
    ("AddRequest-15char", lambda: AddRequest(node_id=1, address="a" * 15)),
    ("AddRequest-large", lambda: AddRequest(node_id=1, address="a" * 200)),
    ("AssignRequest-voter", lambda: AssignRequest(node_id=1, role=NodeRole.VOTER)),
    ("AssignRequest-standby", lambda: AssignRequest(node_id=1, role=NodeRole.STANDBY)),
    ("AssignRequest-spare", lambda: AssignRequest(node_id=1, role=NodeRole.SPARE)),
    ("AssignRequest-id0-voter", lambda: AssignRequest(node_id=0, role=NodeRole.VOTER)),
    (
        "AssignRequest-id-max",
        lambda: AssignRequest(node_id=0xFFFFFFFFFFFFFFFF, role=NodeRole.VOTER),
    ),
    ("RemoveRequest-min", lambda: RemoveRequest(node_id=0)),
    ("RemoveRequest-max", lambda: RemoveRequest(node_id=0xFFFFFFFFFFFFFFFF)),
    ("DumpRequest-empty", lambda: DumpRequest(name="")),
    ("DumpRequest-1char", lambda: DumpRequest(name="x")),
    ("DumpRequest-7char", lambda: DumpRequest(name="x" * 7)),
    ("DumpRequest-8char", lambda: DumpRequest(name="x" * 8)),
    ("DumpRequest-utf8", lambda: DumpRequest(name="café-db")),
    ("ClusterRequest-v1", lambda: ClusterRequest(format=1)),
    ("TransferRequest-min", lambda: TransferRequest(target_node_id=0)),
    ("TransferRequest-max", lambda: TransferRequest(target_node_id=0xFFFFFFFFFFFFFFFF)),
    ("DescribeRequest-v0", lambda: DescribeRequest(format=0)),
    ("WeightRequest-min", lambda: WeightRequest(weight=0)),
    ("WeightRequest-max", lambda: WeightRequest(weight=0xFFFFFFFFFFFFFFFF)),
]


@pytest.mark.parametrize(("label", "constructor"), _CASES, ids=[c[0] for c in _CASES])
def test_admin_request_encode_decode_reencode_byte_identical(
    label: str, constructor: Callable[[], Message]
) -> None:
    """Encode → decode → re-encode must produce identical bytes."""
    msg = constructor()
    encoded_1 = msg.encode()
    cls = type(msg)
    decoded = cls.decode_body(encoded_1[HEADER_SIZE:])
    encoded_2 = decoded.encode()
    assert encoded_1 == encoded_2, (
        f"{cls.__name__} ({label}) encode → decode → re-encode "
        f"is not byte-identical: {encoded_1.hex()} != {encoded_2.hex()}"
    )


def test_assign_request_legacy_shape_byte_identical_via_encode_body_legacy() -> None:
    """Byte-identity for the legacy 8-byte PROMOTE AssignRequest shape."""
    legacy_body = (42).to_bytes(8, "little")
    decoded = AssignRequest.decode_body(legacy_body)
    reencoded = decoded.encode_body_legacy()
    assert legacy_body == reencoded


# ---- merged from test_admin_request_schema_validation.py ----
# Admin-request decoders reject a non-zero schema kwarg on the direct-caller
# path (the wire dispatcher already gates schema before decode_body). Mirrors
# upstream C's INIT_V0 macro (gateway.c) rejecting req->schema != 0.

# Each body is a valid V0 frame so the only thing causing a decode failure
# is the schema kwarg; the schema gate must fire before length/content checks.
_ADMIN_CASES = [
    (LeaderRequest, encode_uint64(0)),
    (ClientRequest, encode_uint64(0)),
    (_HeartbeatRequest, encode_uint64(0)),
    (OpenRequest, encode_text("db") + encode_uint64(0) + encode_text("vfs")),
    (FinalizeRequest, encode_uint32(0) + encode_uint32(0)),
    (InterruptRequest, encode_uint64(0)),
    (_ConnectRequest, encode_uint64(1) + encode_text("a:1")),
    (AddRequest, encode_uint64(1) + encode_text("a:1")),
    (AssignRequest, encode_uint64(1) + encode_uint64(0)),
    (RemoveRequest, encode_uint64(1)),
    (DumpRequest, encode_text("db")),
    (ClusterRequest, encode_uint64(1)),
    (TransferRequest, encode_uint64(1)),
    (DescribeRequest, encode_uint64(0)),
    (WeightRequest, encode_uint64(0)),
]


@pytest.mark.parametrize(("cls", "body"), _ADMIN_CASES, ids=[c.__name__ for c, _ in _ADMIN_CASES])
def test_admin_decoder_rejects_nonzero_schema(cls: Any, body: bytes) -> None:
    with pytest.raises(DecodeError, match="unsupported schema version"):
        cls.decode_body(body, schema=1)


@pytest.mark.parametrize(("cls", "body"), _ADMIN_CASES, ids=[c.__name__ for c, _ in _ADMIN_CASES])
def test_admin_decoder_rejects_garbage_schema(cls: Any, body: bytes) -> None:
    with pytest.raises(DecodeError, match="unsupported schema version"):
        cls.decode_body(body, schema=99)


@pytest.mark.parametrize(("cls", "body"), _ADMIN_CASES, ids=[c.__name__ for c, _ in _ADMIN_CASES])
def test_admin_decoder_accepts_default_schema(cls: Any, body: bytes) -> None:
    cls.decode_body(body)
    cls.decode_body(body, schema=0)


# ---- merged from test_assign_request_docstring_and_construction.py ----
# AssignRequest docstring records the deliberate divergence from C's
# silent-fold-to-VOTER for unknown roles (we reject so future role codes
# aren't masked in mixed-version rollouts), plus a frozen=True tripwire on
# the post-init role coercion.


def test_assign_request_raw_int_role_coerces_to_nodeRole_and_equates() -> None:
    """Tripwire for a frozen=True flip: post-init role coercion must succeed."""
    msg = AssignRequest(node_id=1, role=0)
    assert msg.role is NodeRole.VOTER
    assert msg == AssignRequest(node_id=1, role=NodeRole.VOTER)


# ---- merged from test_assign_request_role_none_construction_check.py ----
# AssignRequest rejects bare role=None at construction (failing early, not
# at encode time) unless the _legacy_intent=True sentinel is set.


def test_assign_request_bare_construction_rejects_role_none() -> None:
    """Bare AssignRequest(node_id=42) must fail at construction, not encode."""
    with pytest.raises(EncodeError, match="role"):
        AssignRequest(node_id=42)


def test_assign_request_role_none_with_legacy_intent_constructs() -> None:
    """_legacy_intent=True opts into the legacy PROMOTE body with role=None."""
    msg = AssignRequest(node_id=42, role=None, _legacy_intent=True)
    assert msg.role is None
    assert msg.encode_body_legacy() == encode_uint64(42)


def test_assign_request_legacy_intent_does_not_compare_or_repr() -> None:
    """The sentinel field must not appear in equality or repr."""
    bare = AssignRequest(node_id=42, role=None, _legacy_intent=True)
    other = AssignRequest(node_id=42, role=None, _legacy_intent=True)
    assert bare == other
    assert "_legacy_intent" not in repr(bare)


# ---- merged from test_assignrequest_legacy_docstring.py ----
# AssignRequest.encode_body_legacy docstring must not claim parity with
# LeaderResponse.encode_body_legacy: the sibling rejects information loss,
# but this one silently drops role (legacy PROMOTE has no role field).


def test_assignrequest_legacy_silently_drops_role_behaviour() -> None:
    """Legacy encoder is information-lossy: role=SPARE and role=None encode
    to the same 8-byte body."""
    from dqlitewire.constants import NodeRole

    req_with_role = AssignRequest(node_id=42, role=NodeRole.SPARE)
    req_no_role = AssignRequest(node_id=42, role=None, _legacy_intent=True)
    assert req_with_role.encode_body_legacy() == req_no_role.encode_body_legacy()
    assert len(req_with_role.encode_body_legacy()) == 8


# ---- merged from test_cluster_request_decoded_v0_round_trip.py ----
# A V0 (``format=0``) ``ClusterRequest`` obtained via ``decode_body`` re-encodes
# byte-identically (the ``_decoded=True`` sentinel), but a fresh ``ClusterRequest(format=0)``
# is still rejected at construction.


def test_cluster_request_v0_decoded_round_trips_byte_identical() -> None:
    body = bytes(8)
    req = ClusterRequest.decode_body(body)
    assert req.format == 0
    assert req.encode_body() == body


def test_cluster_request_v1_round_trip_unchanged() -> None:
    body = b"\x01" + b"\x00" * 7
    req = ClusterRequest.decode_body(body)
    assert req.format == 1
    assert req.encode_body() == body


def test_cluster_request_fresh_v0_still_rejected_at_construction() -> None:
    with pytest.raises(EncodeError, match="V0"):
        ClusterRequest(format=0)


# ---- merged from test_clusterrequest_decode_no_new_bypass.py ----
# ``ClusterRequest.decode_body`` constructs via the dataclass __init__
# (``_decoded=True`` kwarg); a V0 request round-trips through that path.


def test_cluster_request_decode_v0_constructor_kwarg_path() -> None:
    """A V0 request constructs directly via ``_decoded=True`` — the decoder's path."""
    req = ClusterRequest(format=0, _decoded=True)
    assert req.format == 0
    decoded = ClusterRequest.decode_body(b"\x00" * 8)
    assert req == decoded


# ---- merged from test_clusterrequest_decoded_field.py ----
# ``ClusterRequest._decoded`` is a declared dataclass field (repr=False,
# compare=False), not a runtime attribute, and survives ``dataclasses.replace``.


def test_clusterrequest_decoded_does_not_appear_in_vars() -> None:
    req = ClusterRequest.decode_body(b"\x00" * 8)
    assert req.format == 0
    instance_vars = vars(req)
    assert "_decoded" not in instance_vars or instance_vars.get("_decoded") is True
    # Equality ignores _decoded (compare=False).
    v1_request = ClusterRequest(format=1)
    v0_decoded = ClusterRequest.decode_body(b"\x00" * 8)
    assert v0_decoded == ClusterRequest.decode_body(b"\x00" * 8)
    assert v0_decoded != v1_request


def test_clusterrequest_dataclasses_replace_v0_preserves_decoded_sentinel() -> None:
    """``dataclasses.replace`` of a V0-decoded request preserves ``_decoded``,
    so the V0 gate short-circuits on the copy (ExecRequest._decoded_schema parity)."""
    req = ClusterRequest.decode_body(b"\x00" * 8)
    replaced = dataclasses.replace(req)
    assert replaced.format == 0
    assert replaced == req
    # Explicitly clearing the sentinel re-triggers the construction-time V0 gate.
    with pytest.raises(EncodeError, match="V0"):
        dataclasses.replace(req, _decoded=False)


def test_clusterrequest_v0_via_public_constructor_still_rejected() -> None:
    """Only the decoder bypass accepts V0; the public constructor still rejects it."""
    with pytest.raises(EncodeError, match="V0"):
        ClusterRequest(format=0)


def test_clusterrequest_repr_does_not_leak_decoded() -> None:
    """``_decoded`` is repr=False, so a decoded V0 request prints like a V1."""
    req = ClusterRequest.decode_body(b"\x00" * 8)
    assert "_decoded" not in repr(req)


# ---- merged from test_describe_request_decoded_non_zero_format.py ----
# ``DescribeRequest.decode_body`` admits non-zero format only under
# ``strict=False`` (for proxy/replay/fuzz tools); the default stays strict.


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


# ---- merged from test_dump_request_filename_c_server_cap.py ----
# ``DumpRequest.name`` is capped at the C gateway's WAL filename ceiling
# (1019 = 1024-byte buffer - len("-wal") - NUL). A longer name encodes valid
# wire bytes but the C side silently truncates the WAL filename, returning a
# ``FilesResponse`` whose ``-wal`` entry mismatches the main entry (silent dump
# corruption); reject it at the Python wire boundary instead.


def test_dump_request_filename_at_c_server_ceiling_accepted() -> None:
    """Exactly the C ceiling (1019 bytes) must encode cleanly."""
    name = "a" * MAX_DUMP_FILENAME_SIZE
    body = DumpRequest(name).encode_body()
    decoded = DumpRequest.decode_body(body)
    assert decoded.name == name


def test_dump_request_filename_one_past_c_server_ceiling_rejected() -> None:
    """One byte past the ceiling (the WAL truncation boundary) must raise."""
    name = "a" * (MAX_DUMP_FILENAME_SIZE + 1)
    with pytest.raises(EncodeError):
        DumpRequest(name).encode_body()


def test_dump_request_decoder_rejects_oversize_peer_request() -> None:
    """Decode side must also reject an oversize peer-supplied name, keeping the
    wire-symmetric contract against a misbehaving or pre-fix peer."""
    from dqlitewire.types import encode_text

    # Synthesise a body via the lax 4 KiB encoder cap to bypass DumpRequest's
    # own cap; decode_body must still refuse it.
    oversize_name = "a" * (MAX_DUMP_FILENAME_SIZE + 1)
    bogus_body = encode_text(oversize_name, max_size=4096, label="database name")
    with pytest.raises(DecodeError):
        DumpRequest.decode_body(bogus_body)


def test_dump_request_pre_fix_4kib_no_longer_accepted() -> None:
    """Regression against the pre-fix 4 KiB cap: a 2 KiB name must now refuse."""
    name = "a" * 2048
    with pytest.raises(EncodeError):
        DumpRequest(name).encode_body()


# ---- merged from test_sql_requests_sql_type_validation.py ----
# Pin: ``PrepareRequest`` / ``ExecSqlRequest`` / ``QuerySqlRequest``
# reject non-``str`` ``sql`` at construction (``EncodeError``) rather
# than at ``encode_body()``. Plus: the encoded label diagnostics name
# the field ("SQL") rather than the generic "Text".


@pytest.mark.parametrize(
    "cls",
    [PrepareRequest, ExecSqlRequest, QuerySqlRequest],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        b"SELECT 1",  # bytes
        123,  # int
        None,  # None
        memoryview(b"SELECT 1"),  # memoryview
    ],
)
def test_sql_field_must_be_str_at_construction(cls: type, bad_value: object) -> None:
    with pytest.raises(EncodeError, match="sql must be str"):
        cls(db_id=0, sql=bad_value)


@pytest.mark.parametrize(
    "cls",
    [PrepareRequest, ExecSqlRequest, QuerySqlRequest],
)
def test_sql_encode_oversize_error_names_field(cls: type) -> None:
    """Encode-side cap diagnostics carry the ``SQL`` label so an
    operator triaging a wire capture knows which field overflowed
    without walking the traceback."""
    from dqlitewire.limits import MAX_TEXT_VALUE_SIZE

    huge_sql = "X" * (MAX_TEXT_VALUE_SIZE + 1)
    req = cls(db_id=0, sql=huge_sql)
    with pytest.raises(EncodeError, match="SQL"):
        req.encode_body()


@pytest.mark.parametrize(
    "cls",
    [PrepareRequest, ExecSqlRequest, QuerySqlRequest],
)
def test_sql_decode_oversize_error_names_field(cls: type) -> None:
    """Decode-side cap diagnostics carry the ``SQL`` label, symmetric
    with the encode side, so an operator reading a wire capture's
    ``DecodeError`` sees which field overflowed without walking the
    traceback. Regression-resistant against a refactor that drops
    ``label="SQL"`` (or default-restores ``max_size``) on the decode
    side — that change would be silent today (only encode-side has
    a pin)."""
    from dqlitewire.limits import MAX_TEXT_VALUE_SIZE

    # Body shape: uint64 db_id + text payload (NUL-terminated, padded
    # to 8-byte boundary). Build a payload longer than the cap that
    # still contains a NUL at the end so the decoder hits the
    # size-cap branch (not the unterminated branch).
    db_id_bytes = (0).to_bytes(8, "little")
    huge_text = b"X" * (MAX_TEXT_VALUE_SIZE + 8)
    payload = huge_text + b"\x00"
    pad = (-len(payload)) % 8
    body = db_id_bytes + payload + b"\x00" * pad

    with pytest.raises(DecodeError, match="SQL"):
        cls.decode_body(body)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "cls",
    [PrepareRequest, ExecSqlRequest, QuerySqlRequest],
)
def test_sql_decode_unterminated_error_names_field(cls: type) -> None:
    """Decode-side null-termination diagnostic also carries the SQL
    label so a truncated payload's error message is field-specific
    rather than the generic ``"Text not null-terminated"``."""
    db_id_bytes = (0).to_bytes(8, "little")
    sql_bytes = b"SELECT 1 FROM t"
    pad = (-len(db_id_bytes + sql_bytes)) % 8
    body = db_id_bytes + sql_bytes + b"Y" * pad  # no NUL anywhere

    with pytest.raises(DecodeError, match="SQL"):
        cls.decode_body(body)  # type: ignore[attr-defined]


# ---- merged from test_empty_params_roundtrip_byte_identity.py ----
# A foreign-encoded request with EMPTY ``params`` must round-trip
# byte-identically for both valid wire shapes: Go-style (no tuple bytes, body 8
# bytes) and C-style (explicit 8-byte zero header, body 16 bytes). The
# ``_decoded_empty_header`` field records which shape the input carried so the
# encoder re-emits the same one.


@pytest.mark.parametrize("schema", [0, 1])
@pytest.mark.parametrize("RequestCls", [ExecRequest, QueryRequest])
def test_empty_params_go_style_roundtrip_byte_identical(
    schema: int, RequestCls: type[ExecRequest] | type[QueryRequest]
) -> None:
    """A Go-encoded EXEC/QUERY with empty params (8 bytes) must not gain a fake
    8-byte empty-params header on re-encode."""
    data = (1).to_bytes(4, "little") + (2).to_bytes(4, "little")
    req = RequestCls.decode_body(data, schema=schema)
    assert list(req.params) == []
    assert req.encode_body() == data, (
        f"Go-style empty-params {RequestCls.__name__} (schema={schema}) "
        f"round-trip is not byte-identical: got {len(req.encode_body())} bytes, "
        f"expected {len(data)}"
    )


@pytest.mark.parametrize("schema", [0, 1])
@pytest.mark.parametrize("RequestCls", [ExecRequest, QueryRequest])
def test_empty_params_c_style_roundtrip_byte_identical(
    schema: int, RequestCls: type[ExecRequest] | type[QueryRequest]
) -> None:
    """A C-encoded EXEC/QUERY with empty params (16 bytes) must keep the
    explicit zero header on re-encode."""
    data = (1).to_bytes(4, "little") + (2).to_bytes(4, "little") + b"\x00" * 8
    req = RequestCls.decode_body(data, schema=schema)
    assert list(req.params) == []
    assert req.encode_body() == data


@pytest.mark.parametrize("schema", [0, 1])
@pytest.mark.parametrize("RequestCls", [ExecSqlRequest, QuerySqlRequest])
def test_sql_empty_params_go_style_roundtrip_byte_identical(
    schema: int,
    RequestCls: type[ExecSqlRequest] | type[QuerySqlRequest],
) -> None:
    # SQL "x" + NUL = 2 bytes, padded to an 8-byte word; no params bytes.
    sql_bytes = b"x\x00" + b"\x00" * 6
    data = (1).to_bytes(8, "little") + sql_bytes
    req = RequestCls.decode_body(data, schema=schema)
    assert list(req.params) == []
    assert req.sql == "x"
    assert req.encode_body() == data


@pytest.mark.parametrize("schema", [0, 1])
@pytest.mark.parametrize("RequestCls", [ExecSqlRequest, QuerySqlRequest])
def test_sql_empty_params_c_style_roundtrip_byte_identical(
    schema: int,
    RequestCls: type[ExecSqlRequest] | type[QuerySqlRequest],
) -> None:
    sql_bytes = b"x\x00" + b"\x00" * 6
    data = (1).to_bytes(8, "little") + sql_bytes + b"\x00" * 8
    req = RequestCls.decode_body(data, schema=schema)
    assert list(req.params) == []
    assert req.sql == "x"
    assert req.encode_body() == data


def test_caller_originated_empty_params_emits_go_style() -> None:
    """A caller-constructed request (no decode) defaults to the Go-style
    omission via ``_decoded_empty_header=None``."""
    req = ExecRequest(db_id=1, stmt_id=2, params=[])
    body = req.encode_body()
    assert len(body) == 8
    assert body == (1).to_bytes(4, "little") + (2).to_bytes(4, "little")


# ---- merged from test_decoded_schema_empty_params_byte_identity.py ----
# _decoded_schema hint gives byte-identical re-emission of foreign-encoded empty-params bodies.


@pytest.mark.parametrize("schema", [0, 1])
def test_exec_request_empty_params_byte_identical(schema: int) -> None:
    db_id, stmt_id = 1, 2
    body = db_id.to_bytes(4, "little") + stmt_id.to_bytes(4, "little") + b"\x00" * 8
    req = ExecRequest.decode_body(body, schema=schema)
    assert req.params == []
    assert req._decoded_schema == schema
    assert req.encode_body() == body, "round-trip not byte-identical"


@pytest.mark.parametrize("schema", [0, 1])
def test_query_request_empty_params_byte_identical(schema: int) -> None:
    db_id, stmt_id = 1, 2
    body = db_id.to_bytes(4, "little") + stmt_id.to_bytes(4, "little") + b"\x00" * 8
    req = QueryRequest.decode_body(body, schema=schema)
    assert req.params == []
    assert req._decoded_schema == schema
    assert req.encode_body() == body


@pytest.mark.parametrize("schema", [0, 1])
def test_exec_sql_request_empty_params_byte_identical(schema: int) -> None:
    """ExecSqlRequest body: db_id (8) + sql (text + padding) + params."""
    db_id = 1
    from dqlitewire.types import encode_text

    body = db_id.to_bytes(8, "little") + encode_text("SELECT 1") + b"\x00" * 8
    req = ExecSqlRequest.decode_body(body, schema=schema)
    assert req.params == []
    assert req._decoded_schema == schema
    assert req.encode_body() == body


@pytest.mark.parametrize("schema", [0, 1])
def test_query_sql_request_empty_params_byte_identical(schema: int) -> None:
    from dqlitewire.types import encode_text

    db_id = 1
    body = db_id.to_bytes(8, "little") + encode_text("SELECT 1") + b"\x00" * 8
    req = QuerySqlRequest.decode_body(body, schema=schema)
    assert req.params == []
    assert req._decoded_schema == schema
    assert req.encode_body() == body


def test_self_originated_empty_params_still_zero_bytes() -> None:
    """Self-originated requests (no _decoded_schema hint) keep the Go-style 0-byte empty-params."""
    req = ExecRequest(db_id=1, stmt_id=2, params=[])
    assert req._decoded_schema is None
    encoded = req.encode_body()
    assert len(encoded) == 8


# ---- merged from test_decoded_empty_header_field_validator.py ----
# _decoded_empty_header is validated at construction to match its wire-byte semantics.


@pytest.mark.parametrize(
    "RequestCls,kwargs",
    [
        (ExecRequest, {"db_id": 1, "stmt_id": 2}),
        (QueryRequest, {"db_id": 1, "stmt_id": 2}),
        (ExecSqlRequest, {"db_id": 1, "sql": "x"}),
        (QuerySqlRequest, {"db_id": 1, "sql": "x"}),
    ],
)
def test_decoded_empty_header_true_with_non_empty_params_rejected(
    RequestCls: type, kwargs: dict[str, object]
) -> None:
    """_decoded_empty_header=True only makes sense with empty params; bad combos rejected."""
    with pytest.raises(EncodeError, match="_decoded_empty_header"):
        RequestCls(**kwargs, params=[42], _decoded_empty_header=True)


@pytest.mark.parametrize(
    "RequestCls,kwargs",
    [
        (ExecRequest, {"db_id": 1, "stmt_id": 2}),
        (QueryRequest, {"db_id": 1, "stmt_id": 2}),
        (ExecSqlRequest, {"db_id": 1, "sql": "x"}),
        (QuerySqlRequest, {"db_id": 1, "sql": "x"}),
    ],
)
def test_decoded_empty_header_true_with_empty_params_accepted(
    RequestCls: type, kwargs: dict[str, object]
) -> None:
    """Legitimate use: params=[] AND _decoded_empty_header=True (round-trip a C-style frame)."""
    req = RequestCls(**kwargs, params=[], _decoded_empty_header=True)
    assert req._decoded_empty_header is True


@pytest.mark.parametrize(
    "RequestCls,kwargs",
    [
        (ExecRequest, {"db_id": 1, "stmt_id": 2}),
        (QueryRequest, {"db_id": 1, "stmt_id": 2}),
    ],
)
def test_decoded_empty_header_false_with_non_empty_params_accepted(
    RequestCls: type, kwargs: dict[str, object]
) -> None:
    """False ("decoded a Go-style frame") is admissible with any param count."""
    req = RequestCls(**kwargs, params=[42, 43], _decoded_empty_header=False)
    assert req._decoded_empty_header is False


def test_caller_originated_default_is_none() -> None:
    """None is the caller-originated default; False would lose the decoded-vs-never bit."""
    req = ExecRequest(db_id=1, stmt_id=2)
    assert req._decoded_empty_header is None


@pytest.mark.parametrize("bad_value", [1, 0, "yes", "", object(), [], (1,)])
def test_decoded_empty_header_non_bool_rejected(bad_value: object) -> None:
    """Reject non-bool/non-None inputs: else bool() at encode time promotes truthy values."""
    with pytest.raises(EncodeError, match="_decoded_empty_header must be None or bool"):
        ExecRequest(db_id=1, stmt_id=2, params=[], _decoded_empty_header=bad_value)  # type: ignore[arg-type]


def test_decoded_empty_header_none_accepted_explicitly() -> None:
    req = ExecRequest(db_id=1, stmt_id=2, params=[], _decoded_empty_header=None)
    assert req._decoded_empty_header is None
