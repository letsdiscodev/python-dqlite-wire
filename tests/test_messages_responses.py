"""Tests for response message encoding/decoding."""

from dqlitewire.constants import HEADER_SIZE, ResponseType, ValueType
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
        msg = FailureResponse(code=0, message="")
        encoded = msg.encode()
        decoded = FailureResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.code == 0
        assert decoded.message == ""


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


class TestWelcomeResponse:
    def test_roundtrip(self) -> None:
        msg = WelcomeResponse(heartbeat_timeout=15000)
        encoded = msg.encode()
        decoded = WelcomeResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.heartbeat_timeout == 15000


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

        msg = FilesResponse(files={"test.db": b"data"})
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
        msg = FilesResponse(files={"db.sqlite": b"database content", "wal": b"wal data"})
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files["db.sqlite"] == b"database content"
        assert decoded.files["wal"] == b"wal data"

    def test_roundtrip_single_file(self) -> None:
        msg = FilesResponse(files={"main.db": b"\x00\x01\x02\x03"})
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files["main.db"] == b"\x00\x01\x02\x03"


class TestServersResponse:
    def test_empty(self) -> None:
        msg = ServersResponse(nodes=[])
        encoded = msg.encode()
        decoded = ServersResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.nodes == []

    def test_roundtrip(self) -> None:
        nodes = [
            NodeInfo(node_id=1, address="node1:9001", role=1),
            NodeInfo(node_id=2, address="node2:9002", role=2),
            NodeInfo(node_id=3, address="node3:9003", role=2),
        ]
        msg = ServersResponse(nodes=nodes)
        encoded = msg.encode()
        decoded = ServersResponse.decode_body(encoded[HEADER_SIZE:])
        assert len(decoded.nodes) == 3
        assert decoded.nodes[0].node_id == 1
        assert decoded.nodes[0].address == "node1:9001"
        assert decoded.nodes[1].role == 2


class TestMetadataResponse:
    def test_roundtrip(self) -> None:
        msg = MetadataResponse(failure_domain=1, weight=50)
        encoded = msg.encode()
        decoded = MetadataResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.failure_domain == 1
        assert decoded.weight == 50
