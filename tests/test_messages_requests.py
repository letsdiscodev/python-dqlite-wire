"""Tests for request message encoding/decoding."""

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


class TestExecRequest:
    def test_encode_no_params(self) -> None:
        msg = ExecRequest(db_id=1, stmt_id=2, params=[])
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == RequestType.EXEC

    def test_body_structure(self) -> None:
        msg = ExecRequest(db_id=1, stmt_id=2, params=[])
        body = msg.encode_body()
        # db_id (4) + stmt_id (4) + no params = 8 bytes (Go writes nothing for empty params)
        assert len(body) == 8

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


class TestInterruptRequest:
    def test_roundtrip(self) -> None:
        msg = InterruptRequest(db_id=42)
        encoded = msg.encode()
        decoded = InterruptRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 42


class TestConnectRequest:
    def test_roundtrip(self) -> None:
        msg = ConnectRequest(node_id=5, address="192.168.1.1:9001")
        encoded = msg.encode()
        decoded = ConnectRequest.decode_body(encoded[HEADER_SIZE:])
        assert decoded.node_id == 5
        assert decoded.address == "192.168.1.1:9001"

    def test_body_starts_with_id(self) -> None:
        """Body must start with uint64 id, then text address per Go reference."""
        from dqlitewire.types import decode_uint64

        msg = ConnectRequest(node_id=42, address="node1:9001")
        body = msg.encode_body()
        node_id = decode_uint64(body[:8])
        assert node_id == 42


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
