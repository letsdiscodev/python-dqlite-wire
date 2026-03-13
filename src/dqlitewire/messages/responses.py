"""Server to client response messages."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from dqlitewire.constants import ResponseType, ValueType
from dqlitewire.messages.base import Message
from dqlitewire.tuples import RowMarker, decode_row_header, decode_row_values
from dqlitewire.types import (
    decode_text,
    decode_uint64,
    encode_text,
    encode_uint64,
)


@dataclass
class FailureResponse(Message):
    """Operation failed.

    Body: uint64 code, text message
    """

    MSG_TYPE: ClassVar[int] = ResponseType.FAILURE

    code: int
    message: str

    def encode_body(self) -> bytes:
        return encode_uint64(self.code) + encode_text(self.message)

    @classmethod
    def decode_body(cls, data: bytes) -> "FailureResponse":
        code = decode_uint64(data)
        message, _ = decode_text(data[8:])
        return cls(code, message)


@dataclass
class LeaderResponse(Message):
    """Leader address response.

    Body: uint64 node_id, text address
    """

    MSG_TYPE: ClassVar[int] = ResponseType.LEADER

    node_id: int
    address: str

    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id) + encode_text(self.address)

    @classmethod
    def decode_body(cls, data: bytes) -> "LeaderResponse":
        node_id = decode_uint64(data)
        address, _ = decode_text(data[8:])
        return cls(node_id, address)


@dataclass
class WelcomeResponse(Message):
    """Client registration acknowledgment.

    Body: uint64 heartbeat_timeout
    """

    MSG_TYPE: ClassVar[int] = ResponseType.WELCOME

    heartbeat_timeout: int

    def encode_body(self) -> bytes:
        return encode_uint64(self.heartbeat_timeout)

    @classmethod
    def decode_body(cls, data: bytes) -> "WelcomeResponse":
        heartbeat_timeout = decode_uint64(data)
        return cls(heartbeat_timeout)


@dataclass
class DbResponse(Message):
    """Database opened response.

    Body: uint32 db_id, uint32 reserved
    """

    MSG_TYPE: ClassVar[int] = ResponseType.DB

    db_id: int

    def encode_body(self) -> bytes:
        from dqlitewire.types import encode_uint32

        return encode_uint32(self.db_id) + encode_uint32(0)

    @classmethod
    def decode_body(cls, data: bytes) -> "DbResponse":
        from dqlitewire.types import decode_uint32

        db_id = decode_uint32(data)
        return cls(db_id)


@dataclass
class StmtResponse(Message):
    """Statement prepared response.

    Body: uint32 db_id, uint32 stmt_id, uint64 num_params
    """

    MSG_TYPE: ClassVar[int] = ResponseType.STMT

    db_id: int
    stmt_id: int
    num_params: int

    def encode_body(self) -> bytes:
        from dqlitewire.types import encode_uint32

        return (
            encode_uint32(self.db_id) + encode_uint32(self.stmt_id) + encode_uint64(self.num_params)
        )

    @classmethod
    def decode_body(cls, data: bytes) -> "StmtResponse":
        from dqlitewire.types import decode_uint32

        db_id = decode_uint32(data)
        stmt_id = decode_uint32(data[4:])
        num_params = decode_uint64(data[8:])
        return cls(db_id, stmt_id, num_params)


@dataclass
class ResultResponse(Message):
    """Statement execution result.

    Body: uint64 last_insert_id, uint64 rows_affected
    """

    MSG_TYPE: ClassVar[int] = ResponseType.RESULT

    last_insert_id: int
    rows_affected: int

    def encode_body(self) -> bytes:
        return encode_uint64(self.last_insert_id) + encode_uint64(self.rows_affected)

    @classmethod
    def decode_body(cls, data: bytes) -> "ResultResponse":
        last_insert_id = decode_uint64(data)
        rows_affected = decode_uint64(data[8:])
        return cls(last_insert_id, rows_affected)


@dataclass
class RowsResponse(Message):
    """Query result rows.

    Body: uint64 column_count, text[] column_names, then rows...
    Each row: header (types) + values, ending with marker
    """

    MSG_TYPE: ClassVar[int] = ResponseType.ROWS

    column_names: list[str] = field(default_factory=list)
    column_types: list[ValueType] = field(default_factory=list)
    row_types: list[list[ValueType]] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    has_more: bool = False

    def _get_row_types(self, row_idx: int, row: list[Any]) -> list[ValueType]:
        """Get types for a row: from row_types, column_types, or inferred."""
        if self.row_types and row_idx < len(self.row_types):
            return self.row_types[row_idx]
        if self.column_types:
            return self.column_types
        # Infer from values
        from dqlitewire.types import encode_value

        return [encode_value(v)[1] for v in row]

    def encode_body(self) -> bytes:
        from dqlitewire.tuples import (
            ROW_DONE_BYTE,
            ROW_PART_BYTE,
            encode_row_header,
            encode_row_values,
        )

        result = encode_uint64(len(self.column_names))

        # Column names
        for name in self.column_names:
            result += encode_text(name)

        # Rows - each row gets its own type header
        for i, row in enumerate(self.rows):
            types = self._get_row_types(i, row)
            result += encode_row_header(types)
            result += encode_row_values(row, types)

        # End marker: fill a word with marker bytes (matching Go)
        marker_byte = ROW_PART_BYTE if self.has_more else ROW_DONE_BYTE
        result += bytes([marker_byte]) * 8

        return result

    @classmethod
    def decode_body(cls, data: bytes) -> "RowsResponse":
        offset = 0

        # Column count
        column_count = decode_uint64(data[offset:])
        offset += 8

        # Column names
        column_names: list[str] = []
        for _ in range(column_count):
            name, consumed = decode_text(data[offset:])
            column_names.append(name)
            offset += consumed

        # Rows - each row has its own type header
        rows: list[list[Any]] = []
        all_row_types: list[list[ValueType]] = []

        while offset < len(data):
            # Read row header; markers are detected byte-by-byte inside
            result, consumed = decode_row_header(data[offset:], column_count)
            offset += consumed

            if result is RowMarker.DONE:
                return cls(column_names, row_types=all_row_types, rows=rows, has_more=False)
            if result is RowMarker.PART:
                return cls(column_names, row_types=all_row_types, rows=rows, has_more=True)

            types = result
            assert isinstance(types, list)
            all_row_types.append(types)

            # Read row values
            values, consumed = decode_row_values(data[offset:], types)
            rows.append(values)
            offset += consumed

        return cls(column_names, row_types=all_row_types, rows=rows, has_more=False)


@dataclass
class EmptyResponse(Message):
    """Empty response (for exec with no result).

    Body: uint64 (reserved, unused)
    """

    MSG_TYPE: ClassVar[int] = ResponseType.EMPTY

    def encode_body(self) -> bytes:
        return encode_uint64(0)

    @classmethod
    def decode_body(cls, data: bytes) -> "EmptyResponse":
        return cls()


@dataclass
class FilesResponse(Message):
    """Database dump files response.

    Body: uint64 count, then repeated (text filename, uint64 size, raw bytes content)
    """

    MSG_TYPE: ClassVar[int] = ResponseType.FILES

    files: dict[str, bytes] = field(default_factory=dict)

    def encode_body(self) -> bytes:
        result = encode_uint64(len(self.files))
        for name, content in self.files.items():
            result += encode_text(name)
            result += encode_uint64(len(content))
            result += content
        return result

    @classmethod
    def decode_body(cls, data: bytes) -> "FilesResponse":
        files: dict[str, bytes] = {}
        offset = 0
        count = decode_uint64(data[offset:])
        offset += 8
        for _ in range(count):
            name, consumed = decode_text(data[offset:])
            offset += consumed
            size = decode_uint64(data[offset:])
            offset += 8
            content = data[offset : offset + size]
            offset += size
            files[name] = content
        return cls(files)


@dataclass
class NodeInfo:
    """Information about a cluster node."""

    node_id: int
    address: str
    role: int


@dataclass
class ServersResponse(Message):
    """Cluster servers response.

    Body: repeated (uint64 node_id, text address, uint64 role)
    """

    MSG_TYPE: ClassVar[int] = ResponseType.SERVERS

    nodes: list[NodeInfo] = field(default_factory=list)

    def encode_body(self) -> bytes:
        result = b""
        for node in self.nodes:
            result += encode_uint64(node.node_id)
            result += encode_text(node.address)
            result += encode_uint64(node.role)
        return result

    @classmethod
    def decode_body(cls, data: bytes) -> "ServersResponse":
        nodes: list[NodeInfo] = []
        offset = 0
        while offset < len(data):
            node_id = decode_uint64(data[offset:])
            offset += 8
            address, consumed = decode_text(data[offset:])
            offset += consumed
            role = decode_uint64(data[offset:])
            offset += 8
            nodes.append(NodeInfo(node_id, address, role))
        return cls(nodes)


@dataclass
class MetadataResponse(Message):
    """Statement metadata response.

    Body: uint64 failure_domain, uint64 weight
    """

    MSG_TYPE: ClassVar[int] = ResponseType.METADATA

    failure_domain: int
    weight: int

    def encode_body(self) -> bytes:
        return encode_uint64(self.failure_domain) + encode_uint64(self.weight)

    @classmethod
    def decode_body(cls, data: bytes) -> "MetadataResponse":
        failure_domain = decode_uint64(data)
        weight = decode_uint64(data[8:])
        return cls(failure_domain, weight)
