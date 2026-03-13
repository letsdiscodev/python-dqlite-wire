"""Client to server request messages."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from dqlitewire.constants import RequestType
from dqlitewire.messages.base import Message
from dqlitewire.tuples import encode_params_tuple
from dqlitewire.types import encode_text, encode_uint32, encode_uint64


@dataclass
class LeaderRequest(Message):
    """Request current cluster leader address.

    Body: empty
    """

    MSG_TYPE: ClassVar[int] = RequestType.LEADER

    def encode_body(self) -> bytes:
        return b""

    @classmethod
    def decode_body(cls, data: bytes) -> "LeaderRequest":
        return cls()


@dataclass
class ClientRequest(Message):
    """Register as a client.

    Body: uint64 client_id
    """

    MSG_TYPE: ClassVar[int] = RequestType.CLIENT

    client_id: int

    def encode_body(self) -> bytes:
        return encode_uint64(self.client_id)

    @classmethod
    def decode_body(cls, data: bytes) -> "ClientRequest":
        from dqlitewire.types import decode_uint64

        client_id = decode_uint64(data)
        return cls(client_id)


@dataclass
class HeartbeatRequest(Message):
    """Send heartbeat to server.

    Body: uint64 timestamp
    """

    MSG_TYPE: ClassVar[int] = RequestType.HEARTBEAT

    timestamp: int

    def encode_body(self) -> bytes:
        return encode_uint64(self.timestamp)

    @classmethod
    def decode_body(cls, data: bytes) -> "HeartbeatRequest":
        from dqlitewire.types import decode_uint64

        timestamp = decode_uint64(data)
        return cls(timestamp)


@dataclass
class OpenRequest(Message):
    """Open a database.

    Body: text name, uint64 flags, text vfs
    """

    MSG_TYPE: ClassVar[int] = RequestType.OPEN

    name: str
    flags: int = 0
    vfs: str = ""

    def encode_body(self) -> bytes:
        result = encode_text(self.name)
        result += encode_uint64(self.flags)
        result += encode_text(self.vfs)
        return result

    @classmethod
    def decode_body(cls, data: bytes) -> "OpenRequest":
        from dqlitewire.types import decode_text, decode_uint64

        name, offset = decode_text(data)
        flags = decode_uint64(data[offset:])
        offset += 8
        vfs, _ = decode_text(data[offset:])
        return cls(name, flags, vfs)


@dataclass
class PrepareRequest(Message):
    """Prepare a SQL statement.

    Body: uint64 db_id, text sql
    """

    MSG_TYPE: ClassVar[int] = RequestType.PREPARE

    db_id: int
    sql: str

    def encode_body(self) -> bytes:
        return encode_uint64(self.db_id) + encode_text(self.sql)

    @classmethod
    def decode_body(cls, data: bytes) -> "PrepareRequest":
        from dqlitewire.types import decode_text, decode_uint64

        db_id = decode_uint64(data)
        sql, _ = decode_text(data[8:])
        return cls(db_id, sql)


@dataclass
class ExecRequest(Message):
    """Execute a prepared statement.

    Body: uint32 db_id, uint32 stmt_id, params tuple
    """

    MSG_TYPE: ClassVar[int] = RequestType.EXEC

    db_id: int
    stmt_id: int
    params: Sequence[Any] = field(default_factory=list)

    def encode_body(self) -> bytes:
        result = encode_uint32(self.db_id) + encode_uint32(self.stmt_id)
        result += encode_params_tuple(self.params)
        return result

    @classmethod
    def decode_body(cls, data: bytes) -> "ExecRequest":
        from dqlitewire.tuples import decode_params_tuple
        from dqlitewire.types import decode_uint32

        db_id = decode_uint32(data)
        stmt_id = decode_uint32(data[4:])
        params, _ = decode_params_tuple(data[8:])
        return cls(db_id, stmt_id, params)


@dataclass
class QueryRequest(Message):
    """Query a prepared statement.

    Body: uint32 db_id, uint32 stmt_id, params tuple
    """

    MSG_TYPE: ClassVar[int] = RequestType.QUERY

    db_id: int
    stmt_id: int
    params: Sequence[Any] = field(default_factory=list)

    def encode_body(self) -> bytes:
        result = encode_uint32(self.db_id) + encode_uint32(self.stmt_id)
        result += encode_params_tuple(self.params)
        return result

    @classmethod
    def decode_body(cls, data: bytes) -> "QueryRequest":
        from dqlitewire.tuples import decode_params_tuple
        from dqlitewire.types import decode_uint32

        db_id = decode_uint32(data)
        stmt_id = decode_uint32(data[4:])
        params, _ = decode_params_tuple(data[8:])
        return cls(db_id, stmt_id, params)


@dataclass
class FinalizeRequest(Message):
    """Finalize (close) a prepared statement.

    Body: uint32 db_id, uint32 stmt_id
    """

    MSG_TYPE: ClassVar[int] = RequestType.FINALIZE

    db_id: int
    stmt_id: int

    def encode_body(self) -> bytes:
        return encode_uint32(self.db_id) + encode_uint32(self.stmt_id)

    @classmethod
    def decode_body(cls, data: bytes) -> "FinalizeRequest":
        from dqlitewire.types import decode_uint32

        db_id = decode_uint32(data)
        stmt_id = decode_uint32(data[4:])
        return cls(db_id, stmt_id)


@dataclass
class ExecSqlRequest(Message):
    """Execute SQL directly (without prepare).

    Body: uint64 db_id, text sql, params tuple
    """

    MSG_TYPE: ClassVar[int] = RequestType.EXEC_SQL

    db_id: int
    sql: str
    params: Sequence[Any] = field(default_factory=list)

    def encode_body(self) -> bytes:
        result = encode_uint64(self.db_id)
        result += encode_text(self.sql)
        result += encode_params_tuple(self.params)
        return result

    @classmethod
    def decode_body(cls, data: bytes) -> "ExecSqlRequest":
        from dqlitewire.tuples import decode_params_tuple
        from dqlitewire.types import decode_text, decode_uint64

        db_id = decode_uint64(data)
        sql, offset = decode_text(data[8:])
        offset += 8
        params, _ = decode_params_tuple(data[offset:])
        return cls(db_id, sql, params)


@dataclass
class QuerySqlRequest(Message):
    """Query SQL directly (without prepare).

    Body: uint64 db_id, text sql, params tuple
    """

    MSG_TYPE: ClassVar[int] = RequestType.QUERY_SQL

    db_id: int
    sql: str
    params: Sequence[Any] = field(default_factory=list)

    def encode_body(self) -> bytes:
        result = encode_uint64(self.db_id)
        result += encode_text(self.sql)
        result += encode_params_tuple(self.params)
        return result

    @classmethod
    def decode_body(cls, data: bytes) -> "QuerySqlRequest":
        from dqlitewire.tuples import decode_params_tuple
        from dqlitewire.types import decode_text, decode_uint64

        db_id = decode_uint64(data)
        sql, offset = decode_text(data[8:])
        offset += 8
        params, _ = decode_params_tuple(data[offset:])
        return cls(db_id, sql, params)


@dataclass
class InterruptRequest(Message):
    """Interrupt the current operation.

    Body: uint64 db_id
    """

    MSG_TYPE: ClassVar[int] = RequestType.INTERRUPT

    db_id: int

    def encode_body(self) -> bytes:
        return encode_uint64(self.db_id)

    @classmethod
    def decode_body(cls, data: bytes) -> "InterruptRequest":
        from dqlitewire.types import decode_uint64

        db_id = decode_uint64(data)
        return cls(db_id)


@dataclass
class ConnectRequest(Message):
    """Connect to a specific node.

    Body: text address
    """

    MSG_TYPE: ClassVar[int] = RequestType.CONNECT

    address: str

    def encode_body(self) -> bytes:
        return encode_text(self.address)

    @classmethod
    def decode_body(cls, data: bytes) -> "ConnectRequest":
        from dqlitewire.types import decode_text

        address, _ = decode_text(data)
        return cls(address)


@dataclass
class AddRequest(Message):
    """Add a node to the cluster.

    Body: uint64 node_id, text address
    """

    MSG_TYPE: ClassVar[int] = RequestType.ADD

    node_id: int
    address: str

    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id) + encode_text(self.address)

    @classmethod
    def decode_body(cls, data: bytes) -> "AddRequest":
        from dqlitewire.types import decode_text, decode_uint64

        node_id = decode_uint64(data)
        address, _ = decode_text(data[8:])
        return cls(node_id, address)


@dataclass
class AssignRequest(Message):
    """Assign a role to a node.

    Body: uint64 node_id, uint64 role
    """

    MSG_TYPE: ClassVar[int] = RequestType.ASSIGN

    node_id: int
    role: int

    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id) + encode_uint64(self.role)

    @classmethod
    def decode_body(cls, data: bytes) -> "AssignRequest":
        from dqlitewire.types import decode_uint64

        node_id = decode_uint64(data)
        role = decode_uint64(data[8:])
        return cls(node_id, role)


@dataclass
class RemoveRequest(Message):
    """Remove a node from the cluster.

    Body: uint64 node_id
    """

    MSG_TYPE: ClassVar[int] = RequestType.REMOVE

    node_id: int

    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id)

    @classmethod
    def decode_body(cls, data: bytes) -> "RemoveRequest":
        from dqlitewire.types import decode_uint64

        node_id = decode_uint64(data)
        return cls(node_id)


@dataclass
class DumpRequest(Message):
    """Request a database dump.

    Body: text name
    """

    MSG_TYPE: ClassVar[int] = RequestType.DUMP

    name: str

    def encode_body(self) -> bytes:
        return encode_text(self.name)

    @classmethod
    def decode_body(cls, data: bytes) -> "DumpRequest":
        from dqlitewire.types import decode_text

        name, _ = decode_text(data)
        return cls(name)


@dataclass
class ClusterRequest(Message):
    """Request cluster information.

    Body: uint64 format
    """

    MSG_TYPE: ClassVar[int] = RequestType.CLUSTER

    format: int = 0

    def encode_body(self) -> bytes:
        return encode_uint64(self.format)

    @classmethod
    def decode_body(cls, data: bytes) -> "ClusterRequest":
        from dqlitewire.types import decode_uint64

        format_val = decode_uint64(data)
        return cls(format_val)


@dataclass
class TransferRequest(Message):
    """Request leadership transfer.

    Body: uint64 target_node_id
    """

    MSG_TYPE: ClassVar[int] = RequestType.TRANSFER

    target_node_id: int

    def encode_body(self) -> bytes:
        return encode_uint64(self.target_node_id)

    @classmethod
    def decode_body(cls, data: bytes) -> "TransferRequest":
        from dqlitewire.types import decode_uint64

        target_node_id = decode_uint64(data)
        return cls(target_node_id)


@dataclass
class DescribeRequest(Message):
    """Request database schema description.

    Body: uint64 format
    """

    MSG_TYPE: ClassVar[int] = RequestType.DESCRIBE

    format: int = 0

    def encode_body(self) -> bytes:
        return encode_uint64(self.format)

    @classmethod
    def decode_body(cls, data: bytes) -> "DescribeRequest":
        from dqlitewire.types import decode_uint64

        format_val = decode_uint64(data)
        return cls(format_val)


@dataclass
class WeightRequest(Message):
    """Set node weight for leader election.

    Body: uint64 weight
    """

    MSG_TYPE: ClassVar[int] = RequestType.WEIGHT

    weight: int

    def encode_body(self) -> bytes:
        return encode_uint64(self.weight)

    @classmethod
    def decode_body(cls, data: bytes) -> "WeightRequest":
        from dqlitewire.types import decode_uint64

        weight = decode_uint64(data)
        return cls(weight)
