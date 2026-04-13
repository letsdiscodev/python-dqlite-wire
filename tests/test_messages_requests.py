"""Tests for request message encoding/decoding."""

import pytest

from dqlitewire.constants import HEADER_SIZE, RequestType
from dqlitewire.messages.base import Header
from dqlitewire.messages.requests import (
    AddRequest,
    AssignRequest,
    ClientRequest,
    ClusterRequest,
    ConnectRequest,
    DescribeRequest,
    DumpRequest,
    ExecRequest,
    ExecSqlRequest,
    FinalizeRequest,
    HeartbeatRequest,
    InterruptRequest,
    LeaderRequest,
    OpenRequest,
    PrepareRequest,
    QueryRequest,
    QuerySqlRequest,
    RemoveRequest,
    TransferRequest,
    WeightRequest,
)


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


class TestHeartbeatRequest:
    def test_roundtrip(self) -> None:
        msg = HeartbeatRequest(timestamp=1234567890)
        encoded = msg.encode()
        decoded = HeartbeatRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.timestamp == 1234567890


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

        with pytest.raises(ValueError, match="schema must be 0 or 1"):
            PrepareRequest(db_id=1, sql="SELECT 1", schema=2)

        with pytest.raises(ValueError, match="schema must be 0 or 1"):
            PrepareRequest(db_id=1, sql="SELECT 1", schema=-1)


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
        assert decoded.params[0] == 42
        assert decoded.params[1] == "hello"
        assert abs(decoded.params[2] - 3.14) < 0.0001

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
        assert decoded.params[0] == 100
        assert decoded.params[1] == "world"

    def test_roundtrip_v1_more_than_255_params(self) -> None:
        """QueryRequest with >255 params must roundtrip correctly via V1 schema."""
        from dqlitewire.codec import decode_message, encode_message

        params = list(range(256))
        msg = QueryRequest(db_id=1, stmt_id=2, params=params)
        encoded = encode_message(msg)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, QueryRequest)
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
        assert decoded.params[0] == 42

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
        assert decoded.params[0] == 99

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


class TestConnectRequest:
    def test_roundtrip(self) -> None:
        msg = ConnectRequest(node_id=3, address="node3:9001")
        encoded = msg.encode()
        decoded = ConnectRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.node_id == 3
        assert decoded.address == "node3:9001"

    def test_type_code_is_11(self) -> None:
        msg = ConnectRequest(node_id=1, address="localhost:9001")
        assert msg.MSG_TYPE == 11


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

    def test_decode_promote_single_field(self) -> None:
        """Legacy PROMOTE request shares type code 13 but has only node_id.

        The Go/C server distinguishes PROMOTE from ASSIGN by body size:
        - PROMOTE: 1 word (8 bytes) = just node_id
        - ASSIGN: 2 words (16 bytes) = node_id + role
        decode_body must handle both.
        """
        from dqlitewire.types import encode_uint64

        # Simulate a PROMOTE body: just node_id, no role field
        promote_body = encode_uint64(5)
        decoded = AssignRequest.decode_body(promote_body)
        assert decoded.node_id == 5
        assert decoded.role is None

    def test_encode_without_role_emits_deprecation_warning(self) -> None:
        """Encoding AssignRequest without role should warn about legacy Promote."""
        import warnings

        msg = AssignRequest(node_id=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            msg.encode()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "legacy" in str(w[0].message).lower() or "promote" in str(w[0].message).lower()

    def test_encode_with_role_no_warning(self) -> None:
        """Encoding AssignRequest with role should not warn."""
        import warnings

        msg = AssignRequest(node_id=1, role=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            msg.encode()
            assert len(w) == 0


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
        """120: V0 cluster format not supported by ServersResponse decoder."""
        with pytest.raises(ValueError, match="format=0.*not supported"):
            ClusterRequest(format=0)


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
    """Request fields should be validated at construction time, not just at encode time."""

    def test_negative_uint32_field_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="db_id"):
            ExecRequest(db_id=-1, stmt_id=0, params=[])

    def test_overflow_uint32_field_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="stmt_id"):
            FinalizeRequest(db_id=0, stmt_id=2**32)

    def test_negative_uint64_field_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="client_id"):
            ClientRequest(client_id=-1)

    def test_overflow_uint64_field_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="timestamp"):
            HeartbeatRequest(timestamp=2**64)

    def test_valid_values_accepted(self) -> None:
        """Valid values should not raise."""
        ExecRequest(db_id=0, stmt_id=0, params=[])
        ExecRequest(db_id=2**32 - 1, stmt_id=2**32 - 1, params=[])
        ClientRequest(client_id=0)
        ClientRequest(client_id=2**64 - 1)
        HeartbeatRequest(timestamp=0)
        OpenRequest(name="test.db", flags=0, vfs="")

    def test_exec_sql_accepts_large_db_id(self) -> None:
        """134: ExecSqlRequest uses uint64 db_id, accepting values > uint32 max."""
        msg = ExecSqlRequest(db_id=2**32, sql="SELECT 1")
        assert msg.db_id == 2**32

    def test_exec_rejects_large_db_id(self) -> None:
        """134: ExecRequest uses uint32 db_id, rejecting values > uint32 max."""
        import pytest

        with pytest.raises(ValueError, match="db_id"):
            ExecRequest(db_id=2**32, stmt_id=0)

    def test_bool_rejected_for_uint32_db_id(self) -> None:
        """Bool must not be silently accepted as a uint32 field."""
        import pytest

        with pytest.raises(TypeError, match="db_id must be int"):
            ExecRequest(db_id=True, stmt_id=0)

    def test_bool_rejected_for_uint32_stmt_id(self) -> None:
        import pytest

        with pytest.raises(TypeError, match="stmt_id must be int"):
            ExecRequest(db_id=0, stmt_id=False)

    def test_bool_rejected_for_uint64_client_id(self) -> None:
        """Bool must not be silently accepted as a uint64 field."""
        import pytest

        with pytest.raises(TypeError, match="client_id must be int"):
            ClientRequest(client_id=True)
