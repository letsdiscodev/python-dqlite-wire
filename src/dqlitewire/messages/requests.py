"""Client to server request messages."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from dqlitewire.constants import NodeRole, RequestType
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.base import Message
from dqlitewire.tuples import decode_params_tuple, encode_params_tuple
from dqlitewire.types import (
    WireInput,
    decode_text,
    decode_uint32,
    decode_uint64,
    encode_text,
    encode_uint32,
    encode_uint64,
)


def _check_uint32(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if not (0 <= value <= 0xFFFFFFFF):
        raise ValueError(f"{name} must be uint32 (0 to 4294967295), got {value}")


def _check_uint64(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if not (0 <= value <= 0xFFFFFFFFFFFFFFFF):
        raise ValueError(f"{name} must be uint64 (0 to 18446744073709551615), got {value}")


@dataclass
class LeaderRequest(Message):
    """Request current cluster leader address.

    Body: uint64 (reserved, unused)
    """

    MSG_TYPE: ClassVar[int] = RequestType.LEADER

    def encode_body(self) -> bytes:
        return encode_uint64(0)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "LeaderRequest":
        # Strict decode symmetric with ``encode_body``: the body is
        # exactly one uint64 reserved word, defined as 0 by upstream.
        # Reject truncated/extended bodies and non-zero reserved values
        # rather than silently accepting them.
        if len(data) != 8:
            raise DecodeError(f"LeaderRequest body must be 8 bytes, got {len(data)}")
        reserved = decode_uint64(data)
        if reserved != 0:
            raise DecodeError(f"LeaderRequest reserved field must be 0, got {reserved}")
        return cls()


@dataclass
class ClientRequest(Message):
    """Register as a client.

    Body: uint64 client_id
    """

    MSG_TYPE: ClassVar[int] = RequestType.CLIENT

    client_id: int

    def __post_init__(self) -> None:
        _check_uint64("client_id", self.client_id)

    def encode_body(self) -> bytes:
        return encode_uint64(self.client_id)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "ClientRequest":
        if len(data) != 8:
            raise DecodeError(f"ClientRequest body must be 8 bytes, got {len(data)}")
        client_id = decode_uint64(data)
        return cls(client_id)


@dataclass
class HeartbeatRequest(Message):
    """Send heartbeat to server.

    .. warning::

        The ``uint64 timestamp`` body shape is **speculative**. Upstream
        (``protocol.h``) reserves the type code ``DQLITE_REQUEST_HEARTBEAT
        = 2`` but defines no schema — ``REQUEST__TYPES`` in
        ``request.h`` omits ``heartbeat``, and ``gateway.c``'s
        dispatcher falls through to ``DQLITE_PARSE`` for this type. The
        Go client's heartbeat code is also commented out.

        This dataclass preserves the historical ``uint64 timestamp``
        layout for test-mock / golden-byte compatibility, but no real
        upstream peer accepts it: a real C server replies with a
        ``FailureResponse``. If upstream ever defines a schema, this
        body shape will need to change.

    Body: uint64 timestamp (speculative — not part of any upstream spec)
    """

    MSG_TYPE: ClassVar[int] = RequestType.HEARTBEAT

    timestamp: int

    def __post_init__(self) -> None:
        _check_uint64("timestamp", self.timestamp)

    def encode_body(self) -> bytes:
        return encode_uint64(self.timestamp)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "HeartbeatRequest":
        if len(data) != 8:
            raise DecodeError(f"HeartbeatRequest body must be 8 bytes, got {len(data)}")
        timestamp = decode_uint64(data)
        return cls(timestamp)


@dataclass
class OpenRequest(Message):
    """Open a database.

    Body: text name, uint64 flags, text vfs

    Note: the upstream dqlite server (``gateway.c``/``handle_open``)
    currently IGNORES both ``flags`` and ``vfs`` fields. They are encoded
    on the wire for protocol compatibility but have no server-side effect.
    Keep the defaults (flags=0, vfs="") unless you're intentionally
    exercising a future server version or building a mock server.
    """

    MSG_TYPE: ClassVar[int] = RequestType.OPEN

    name: str
    flags: int = 0
    vfs: str = ""

    def __post_init__(self) -> None:
        _check_uint64("flags", self.flags)

    def encode_body(self) -> bytes:
        result = encode_text(self.name)
        result += encode_uint64(self.flags)
        result += encode_text(self.vfs)
        return result

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "OpenRequest":
        name, offset = decode_text(data)
        flags = decode_uint64(data[offset:])
        offset += 8
        vfs, consumed = decode_text(data[offset:])
        offset += consumed
        if offset != len(data):
            raise DecodeError(f"OpenRequest has {len(data) - offset} trailing bytes")
        return cls(name, flags, vfs)


@dataclass
class PrepareRequest(Message):
    """Prepare a SQL statement.

    Body: uint64 db_id, text sql

    Set schema=1 to request V1 response with tail_offset for multi-statement SQL.
    Note: schema=1 is not used by the canonical Go client (go-dqlite), which
    always sends schema=0. The tail_offset feature may be supported by the C
    dqlite server but is not exercised by Go.
    """

    MSG_TYPE: ClassVar[int] = RequestType.PREPARE

    db_id: int
    sql: str
    schema: int = 0

    def __post_init__(self) -> None:
        _check_uint64("db_id", self.db_id)
        if self.schema not in (0, 1):
            raise ValueError(f"schema must be 0 or 1, got {self.schema}")

    def _get_schema(self) -> int:
        return self.schema

    def encode_body(self) -> bytes:
        return encode_uint64(self.db_id) + encode_text(self.sql)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "PrepareRequest":
        db_id = decode_uint64(data)
        sql, consumed = decode_text(data[8:])
        offset = 8 + consumed
        if offset != len(data):
            raise DecodeError(f"PrepareRequest has {len(data) - offset} trailing bytes")
        return cls(db_id, sql, schema=schema)


@dataclass
class ExecRequest(Message):
    """Execute a prepared statement.

    Body: uint32 db_id, uint32 stmt_id, params tuple
    Uses V0 (uint8 count) for <= 255 params, V1 (uint32 count) otherwise.
    """

    MSG_TYPE: ClassVar[int] = RequestType.EXEC

    db_id: int
    stmt_id: int
    params: Sequence[WireInput] = field(default_factory=list)
    # Preserves the header schema byte seen on decode so a decode →
    # re-encode round-trip emits byte-identical output even when the
    # upstream C client used schema=1 with ≤255 params (which the count
    # heuristic alone would otherwise downgrade to schema=0). Excluded
    # from repr/compare so it stays an internal round-trip hint.
    _decoded_schema: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_uint32("db_id", self.db_id)
        _check_uint32("stmt_id", self.stmt_id)

    def _get_schema(self) -> int:
        if self._decoded_schema is not None:
            return self._decoded_schema
        return 1 if len(self.params) > 255 else 0

    def encode_body(self) -> bytes:
        schema = self._get_schema()
        result = encode_uint32(self.db_id) + encode_uint32(self.stmt_id)
        result += encode_params_tuple(self.params, schema=schema, buffer_offset=len(result))
        return result

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "ExecRequest":
        db_id = decode_uint32(data)
        stmt_id = decode_uint32(data[4:])
        params, consumed = decode_params_tuple(data[8:], schema=schema, buffer_offset=8)
        offset = 8 + consumed
        if offset != len(data):
            raise DecodeError(f"ExecRequest has {len(data) - offset} trailing bytes")
        return cls(db_id, stmt_id, params, _decoded_schema=schema)


@dataclass
class QueryRequest(Message):
    """Query a prepared statement.

    Body: uint32 db_id, uint32 stmt_id, params tuple
    Uses V0 (uint8 count) for <= 255 params, V1 (uint32 count) otherwise.
    """

    MSG_TYPE: ClassVar[int] = RequestType.QUERY

    db_id: int
    stmt_id: int
    params: Sequence[WireInput] = field(default_factory=list)
    _decoded_schema: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_uint32("db_id", self.db_id)
        _check_uint32("stmt_id", self.stmt_id)

    def _get_schema(self) -> int:
        if self._decoded_schema is not None:
            return self._decoded_schema
        return 1 if len(self.params) > 255 else 0

    def encode_body(self) -> bytes:
        schema = self._get_schema()
        result = encode_uint32(self.db_id) + encode_uint32(self.stmt_id)
        result += encode_params_tuple(self.params, schema=schema, buffer_offset=len(result))
        return result

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "QueryRequest":
        db_id = decode_uint32(data)
        stmt_id = decode_uint32(data[4:])
        params, consumed = decode_params_tuple(data[8:], schema=schema, buffer_offset=8)
        offset = 8 + consumed
        if offset != len(data):
            raise DecodeError(f"QueryRequest has {len(data) - offset} trailing bytes")
        return cls(db_id, stmt_id, params, _decoded_schema=schema)


@dataclass
class FinalizeRequest(Message):
    """Finalize (close) a prepared statement.

    Body: uint32 db_id, uint32 stmt_id
    """

    MSG_TYPE: ClassVar[int] = RequestType.FINALIZE

    db_id: int
    stmt_id: int

    def __post_init__(self) -> None:
        _check_uint32("db_id", self.db_id)
        _check_uint32("stmt_id", self.stmt_id)

    def encode_body(self) -> bytes:
        return encode_uint32(self.db_id) + encode_uint32(self.stmt_id)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "FinalizeRequest":
        if len(data) != 8:
            raise DecodeError(f"FinalizeRequest body must be 8 bytes, got {len(data)}")
        db_id = decode_uint32(data)
        stmt_id = decode_uint32(data[4:])
        return cls(db_id, stmt_id)


@dataclass
class ExecSqlRequest(Message):
    """Execute SQL directly (without prepare).

    Body: uint64 db_id, text sql, params tuple
    Uses V0 (uint8 count) for <= 255 params, V1 (uint32 count) otherwise.
    """

    MSG_TYPE: ClassVar[int] = RequestType.EXEC_SQL

    db_id: int
    sql: str
    params: Sequence[WireInput] = field(default_factory=list)
    _decoded_schema: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_uint64("db_id", self.db_id)

    def _get_schema(self) -> int:
        if self._decoded_schema is not None:
            return self._decoded_schema
        return 1 if len(self.params) > 255 else 0

    def encode_body(self) -> bytes:
        schema = self._get_schema()
        result = encode_uint64(self.db_id)
        result += encode_text(self.sql)
        result += encode_params_tuple(self.params, schema=schema, buffer_offset=len(result))
        return result

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "ExecSqlRequest":
        db_id = decode_uint64(data)
        sql, offset = decode_text(data[8:])
        offset += 8
        params, consumed = decode_params_tuple(data[offset:], schema=schema, buffer_offset=offset)
        offset += consumed
        if offset != len(data):
            raise DecodeError(f"ExecSqlRequest has {len(data) - offset} trailing bytes")
        return cls(db_id, sql, params, _decoded_schema=schema)


@dataclass
class QuerySqlRequest(Message):
    """Query SQL directly (without prepare).

    Body: uint64 db_id, text sql, params tuple
    Uses V0 (uint8 count) for <= 255 params, V1 (uint32 count) otherwise.
    """

    MSG_TYPE: ClassVar[int] = RequestType.QUERY_SQL

    db_id: int
    sql: str
    params: Sequence[WireInput] = field(default_factory=list)
    _decoded_schema: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_uint64("db_id", self.db_id)

    def _get_schema(self) -> int:
        if self._decoded_schema is not None:
            return self._decoded_schema
        return 1 if len(self.params) > 255 else 0

    def encode_body(self) -> bytes:
        schema = self._get_schema()
        result = encode_uint64(self.db_id)
        result += encode_text(self.sql)
        result += encode_params_tuple(self.params, schema=schema, buffer_offset=len(result))
        return result

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "QuerySqlRequest":
        db_id = decode_uint64(data)
        sql, offset = decode_text(data[8:])
        offset += 8
        params, consumed = decode_params_tuple(data[offset:], schema=schema, buffer_offset=offset)
        offset += consumed
        if offset != len(data):
            raise DecodeError(f"QuerySqlRequest has {len(data) - offset} trailing bytes")
        return cls(db_id, sql, params, _decoded_schema=schema)


@dataclass
class InterruptRequest(Message):
    """Interrupt the current operation.

    Body: uint64 db_id
    """

    MSG_TYPE: ClassVar[int] = RequestType.INTERRUPT

    db_id: int

    def __post_init__(self) -> None:
        _check_uint64("db_id", self.db_id)

    def encode_body(self) -> bytes:
        return encode_uint64(self.db_id)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "InterruptRequest":
        if len(data) != 8:
            raise DecodeError(f"InterruptRequest body must be 8 bytes, got {len(data)}")
        db_id = decode_uint64(data)
        return cls(db_id)


@dataclass
class ConnectRequest(Message):
    """Establish a Raft transport connection (node-to-node).

    Body: uint64 node_id, text address

    This message type is defined in the C dqlite protocol (DQLITE_REQUEST_CONNECT = 11)
    but is not used by the Go client library (go-dqlite), which is a client-only
    implementation. It is used for inter-node Raft transport connections within a
    dqlite cluster.
    """

    MSG_TYPE: ClassVar[int] = RequestType.CONNECT

    node_id: int
    address: str

    def __post_init__(self) -> None:
        _check_uint64("node_id", self.node_id)

    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id) + encode_text(self.address)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "ConnectRequest":
        node_id = decode_uint64(data)
        address, consumed = decode_text(data[8:])
        offset = 8 + consumed
        if offset != len(data):
            raise DecodeError(f"ConnectRequest has {len(data) - offset} trailing bytes")
        return cls(node_id, address)


@dataclass
class AddRequest(Message):
    """Add a node to the cluster.

    Body: uint64 node_id, text address
    """

    MSG_TYPE: ClassVar[int] = RequestType.ADD

    node_id: int
    address: str

    def __post_init__(self) -> None:
        _check_uint64("node_id", self.node_id)

    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id) + encode_text(self.address)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "AddRequest":
        node_id = decode_uint64(data)
        address, consumed = decode_text(data[8:])
        offset = 8 + consumed
        if offset != len(data):
            raise DecodeError(f"AddRequest has {len(data) - offset} trailing bytes")
        return cls(node_id, address)


@dataclass
class AssignRequest(Message):
    """Assign a role to a node, or promote a node (legacy).

    ASSIGN and PROMOTE share type code 13. They are distinguished by body size:
    - PROMOTE (legacy): uint64 node_id (1 word)
    - ASSIGN: uint64 node_id, uint64 role (2 words)
    """

    MSG_TYPE: ClassVar[int] = RequestType.ASSIGN

    node_id: int
    role: NodeRole | int | None = None

    def __post_init__(self) -> None:
        _check_uint64("node_id", self.node_id)
        if self.role is not None:
            # Coerce bare ints to the NodeRole enum and reject unknown
            # values. Mirrors the response-side narrowing on
            # ``ServersResponse`` (ISSUE-202) so an outbound assign or
            # a mock-server decode carries a validated role, not an
            # unknown integer that would silently surface in the
            # dataclass.
            if isinstance(self.role, NodeRole):
                _check_uint64("role", int(self.role))
            else:
                _check_uint64("role", self.role)
                try:
                    coerced = NodeRole(self.role)
                except ValueError as e:
                    raise ValueError(f"AssignRequest: unknown role {self.role}") from e
                object.__setattr__(self, "role", coerced)

    def encode_body(self) -> bytes:
        result = encode_uint64(self.node_id)
        if self.role is not None:
            result += encode_uint64(int(self.role))
        else:
            import warnings

            warnings.warn(
                "Encoding AssignRequest without role sends a legacy Promote message "
                "(1-word body). Modern dqlite servers and the Go client always send "
                "both node_id and role. Use role=0 (VOTER) for equivalent behavior.",
                DeprecationWarning,
                stacklevel=2,
            )
        return result

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "AssignRequest":
        # Upstream emits bodies of exactly 8 (PROMOTE) or 16 (ASSIGN)
        # bytes. Reject anything else rather than silently dropping
        # trailing bytes — parity with the C cursor-cap semantics.
        if len(data) == 8:
            node_id = decode_uint64(data)
            return cls(node_id, None)
        if len(data) == 16:
            node_id = decode_uint64(data)
            raw = decode_uint64(data[8:])
            try:
                role = NodeRole(raw)
            except ValueError as e:
                raise DecodeError(f"AssignRequest: unknown role {raw}") from e
            return cls(node_id, role)
        raise DecodeError(
            f"AssignRequest body must be 8 (PROMOTE) or 16 (ASSIGN) bytes, got {len(data)}"
        )


@dataclass
class RemoveRequest(Message):
    """Remove a node from the cluster.

    Body: uint64 node_id
    """

    MSG_TYPE: ClassVar[int] = RequestType.REMOVE

    node_id: int

    def __post_init__(self) -> None:
        _check_uint64("node_id", self.node_id)

    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "RemoveRequest":
        if len(data) != 8:
            raise DecodeError(f"RemoveRequest body must be 8 bytes, got {len(data)}")
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
    def decode_body(cls, data: bytes, schema: int = 0) -> "DumpRequest":
        name, consumed = decode_text(data)
        if consumed != len(data):
            raise DecodeError(f"DumpRequest has {len(data) - consumed} trailing bytes")
        return cls(name)


@dataclass
class ClusterRequest(Message):
    """Request cluster information.

    Body: uint64 format

    Note: format=0 (V0: id+address only, no role) IS a valid upstream
    dqlite wire format. This Python library chooses not to implement it
    because :class:`ServersResponse` only decodes V1 (id+address+role).
    Callers that need V0 compatibility should decode :class:`ServersResponse`
    themselves. Use ``format=1`` for the default path.
    """

    MSG_TYPE: ClassVar[int] = RequestType.CLUSTER

    format: int = 1

    def __post_init__(self) -> None:
        _check_uint64("format", self.format)
        if self.format == 0:
            raise ValueError(
                "ClusterRequest format=0 (V0) is valid in upstream dqlite but "
                "not implemented in this Python library: ServersResponse only "
                "decodes V1 (with node role fields). Use format=1."
            )
        if self.format != 1:
            # Upstream defines only V0=0 and V1=1 (include/dqlite.h); the
            # gateway rejects anything else with DQLITE_PARSE. Reject
            # client-side so callers get a local ValueError instead of a
            # confusing server failure.
            raise ValueError(
                f"ClusterRequest format must be 1 (V1); upstream defines "
                f"only V0=0 and V1=1 and this library implements only V1. "
                f"Got {self.format}."
            )

    def encode_body(self) -> bytes:
        return encode_uint64(self.format)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "ClusterRequest":
        if len(data) != 8:
            raise DecodeError(f"ClusterRequest body must be 8 bytes, got {len(data)}")
        format_val = decode_uint64(data)
        if format_val == 0:
            raise DecodeError(
                "ClusterRequest format=0 (V0) is valid in upstream dqlite but "
                "not implemented in this Python library: ServersResponse only "
                "decodes V1 (with node role fields)."
            )
        if format_val != 1:
            raise DecodeError(
                f"ClusterRequest format must be 1 (V1); upstream defines "
                f"only V0=0 and V1=1. Got {format_val}."
            )
        return cls(format_val)


@dataclass
class TransferRequest(Message):
    """Request leadership transfer.

    Body: uint64 target_node_id
    """

    MSG_TYPE: ClassVar[int] = RequestType.TRANSFER

    target_node_id: int

    def __post_init__(self) -> None:
        _check_uint64("target_node_id", self.target_node_id)

    def encode_body(self) -> bytes:
        return encode_uint64(self.target_node_id)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "TransferRequest":
        if len(data) != 8:
            raise DecodeError(f"TransferRequest body must be 8 bytes, got {len(data)}")
        target_node_id = decode_uint64(data)
        return cls(target_node_id)


@dataclass
class DescribeRequest(Message):
    """Request database schema description.

    Body: uint64 format

    Upstream defines only ``DQLITE_REQUEST_DESCRIBE_FORMAT_V0 = 0``
    (``gateway.c`` rejects anything else with ``SQLITE_PROTOCOL``).
    Reject unknown formats client-side so callers get a local
    ``ValueError`` instead of a confusing server failure.
    """

    MSG_TYPE: ClassVar[int] = RequestType.DESCRIBE

    format: int = 0

    def __post_init__(self) -> None:
        _check_uint64("format", self.format)
        if self.format != 0:
            raise ValueError(
                f"DescribeRequest format must be 0 (V0); upstream rejects "
                f"anything else with SQLITE_PROTOCOL. Got {self.format}."
            )

    def encode_body(self) -> bytes:
        return encode_uint64(self.format)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "DescribeRequest":
        if len(data) != 8:
            raise DecodeError(f"DescribeRequest body must be 8 bytes, got {len(data)}")
        format_val = decode_uint64(data)
        if format_val != 0:
            raise DecodeError(f"DescribeRequest format must be 0 (V0); got {format_val}")
        return cls(format_val)


@dataclass
class WeightRequest(Message):
    """Set node weight for leader election.

    Body: uint64 weight
    """

    MSG_TYPE: ClassVar[int] = RequestType.WEIGHT

    weight: int

    def __post_init__(self) -> None:
        _check_uint64("weight", self.weight)

    def encode_body(self) -> bytes:
        return encode_uint64(self.weight)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "WeightRequest":
        if len(data) != 8:
            raise DecodeError(f"WeightRequest body must be 8 bytes, got {len(data)}")
        weight = decode_uint64(data)
        return cls(weight)
