"""Client to server request messages."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, final, override

from dqlitewire.constants import NodeRole, RequestType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.messages.base import Message

__all__ = [
    "AddRequest",
    "AssignRequest",
    "ClientRequest",
    "ClusterRequest",
    "DescribeRequest",
    "DumpRequest",
    "ExecRequest",
    "ExecSqlRequest",
    "FinalizeRequest",
    "InterruptRequest",
    "LeaderRequest",
    "OpenRequest",
    "PrepareRequest",
    "QueryRequest",
    "QuerySqlRequest",
    "RemoveRequest",
    "TransferRequest",
    "WeightRequest",
]
from dqlitewire.messages.responses import (
    _MAX_ADDRESS_SIZE,
    _MAX_DUMP_FILENAME_SIZE,
    _MAX_FILENAME_SIZE,
)
from dqlitewire.tuples import _MAX_PARAM_COUNT, decode_params_tuple, encode_params_tuple
from dqlitewire.types import (
    _MAX_TEXT_VALUE_SIZE,
    WireInput,
    _validate_uint32,
    _validate_uint64,
    decode_text,
    decode_uint32,
    decode_uint64,
    encode_text,
    encode_uint32,
    encode_uint64,
)


def _validate_decoded_schema(decoded_schema: int | None, param_count: int) -> None:
    """Validate the optional ``_decoded_schema`` round-trip hint (None|0|1).

    schema=0 caps params at 255 (V0 uint8 count); schema=1 caps at
    _MAX_PARAM_COUNT. Enforced at construction so callers get an
    actionable error here, not deep inside encode_params_tuple.
    """
    if decoded_schema is None:
        return
    if decoded_schema not in (0, 1):
        raise EncodeError(f"_decoded_schema must be 0, 1, or None; got {decoded_schema}")
    if decoded_schema == 0 and param_count > 255:
        raise EncodeError(
            f"_decoded_schema=0 (V0 tuple format) supports at most 255 parameters; "
            f"got {param_count}"
        )
    if decoded_schema == 1 and param_count > _MAX_PARAM_COUNT:
        raise EncodeError(
            f"_decoded_schema=1 (V1 tuple format) supports at most "
            f"{_MAX_PARAM_COUNT} parameters; got {param_count}"
        )


def _validate_decoded_empty_header(decoded_empty_header: bool | None, param_count: int) -> None:
    """Validate the optional ``_decoded_empty_header`` round-trip hint (None|bool).

    True (C-style explicit 8-byte zero header) is only valid with empty
    params; otherwise the field would disagree with the wire bytes.
    isinstance(x, bool) is required: a truthy non-bool would silently
    promote to C-style header emission at encode_body.
    """
    if decoded_empty_header is not None and not isinstance(decoded_empty_header, bool):
        raise EncodeError(
            f"_decoded_empty_header must be None or bool, got {type(decoded_empty_header).__name__}"
        )
    if decoded_empty_header is True and param_count > 0:
        raise EncodeError(
            f"_decoded_empty_header=True requires empty params (the field is "
            f"a C-style 8-byte empty-header round-trip hint); got params with "
            f"{param_count} elements"
        )


@final
@dataclass
class LeaderRequest(Message):
    """Request current cluster leader address. Body: uint64 (reserved, unused)."""

    MSG_TYPE: ClassVar[int] = RequestType.LEADER

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(0)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "LeaderRequest":
        if schema != 0:
            raise DecodeError(f"LeaderRequest unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"LeaderRequest body must be 8 bytes, got {len(data)}")
        reserved = decode_uint64(data)
        if reserved != 0:
            raise DecodeError(f"LeaderRequest reserved field must be 0, got {reserved}")
        return cls()


@final
@dataclass
class ClientRequest(Message):
    """Register as a client. Body: uint64 client_id."""

    MSG_TYPE: ClassVar[int] = RequestType.CLIENT

    client_id: int

    def __post_init__(self) -> None:
        _validate_uint64("client_id", self.client_id)

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.client_id)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "ClientRequest":
        if schema != 0:
            raise DecodeError(f"ClientRequest unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"ClientRequest body must be 8 bytes, got {len(data)}")
        client_id = decode_uint64(data)
        return cls(client_id)


@dataclass
class _HeartbeatRequest(Message):
    """Heartbeat (private). Body: uint64 timestamp, Go-canonical per schema.go.

    No upstream peer accepts a heartbeat frame (C omits it from
    REQUEST__TYPES, Go's heartbeat code is commented out), so this is
    kept out of the codec registry and __all__ — for test/golden-byte
    harnesses only. schema.go is the layout anchor.
    """

    MSG_TYPE: ClassVar[int] = RequestType.HEARTBEAT

    timestamp: int

    def __post_init__(self) -> None:
        _validate_uint64("timestamp", self.timestamp)

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.timestamp)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "_HeartbeatRequest":
        if schema != 0:
            raise DecodeError(f"_HeartbeatRequest unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"_HeartbeatRequest body must be 8 bytes, got {len(data)}")
        timestamp = decode_uint64(data)
        return cls(timestamp)


@final
@dataclass
class OpenRequest(Message):
    """Open a database. Body: text name, uint64 flags, text vfs.

    The upstream server ignores flags and vfs; keep the defaults unless
    targeting a future server or a mock.
    """

    MSG_TYPE: ClassVar[int] = RequestType.OPEN

    name: str
    flags: int = 0
    vfs: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise EncodeError(f"OpenRequest.name must be str, got {type(self.name).__name__}")
        if not isinstance(self.vfs, str):
            raise EncodeError(f"OpenRequest.vfs must be str, got {type(self.vfs).__name__}")
        _validate_uint64("flags", self.flags)

    @override
    def encode_body(self) -> bytes:
        result = encode_text(self.name, max_size=_MAX_FILENAME_SIZE, label="database name")
        result += encode_uint64(self.flags)
        result += encode_text(self.vfs, max_size=_MAX_FILENAME_SIZE, label="vfs name")
        return result

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "OpenRequest":
        if schema != 0:
            raise DecodeError(f"OpenRequest unsupported schema version {schema}")
        # memoryview keeps per-slice cost O(1) on large bodies with many text fields.
        view = memoryview(data)
        name, offset = decode_text(view, max_size=_MAX_FILENAME_SIZE, label="database name")
        flags = decode_uint64(view[offset:], label="OpenRequest.flags")
        offset += 8
        vfs, consumed = decode_text(view[offset:], max_size=_MAX_FILENAME_SIZE, label="vfs name")
        offset += consumed
        if offset != len(data):
            raise DecodeError(f"OpenRequest has {len(data) - offset} trailing bytes")
        return cls(name, flags, vfs)


@final
@dataclass
class PrepareRequest(Message):
    """Prepare a SQL statement. Body: uint64 db_id, text sql.

    schema=1 requests a V1 response with tail_offset (multi-statement
    SQL); the Go client always sends schema=0.
    """

    MSG_TYPE: ClassVar[int] = RequestType.PREPARE

    db_id: int
    sql: str
    schema: int = 0

    def __post_init__(self) -> None:
        _validate_uint64("db_id", self.db_id)
        if not isinstance(self.sql, str):
            raise EncodeError(f"PrepareRequest.sql must be str, got {type(self.sql).__name__}")
        if self.schema not in (0, 1):
            raise EncodeError(f"schema must be 0 or 1, got {self.schema}")

    @override
    def _get_schema(self) -> int:
        return self.schema

    @override
    def encode_body(self) -> bytes:
        # Cap symmetric with decode so an over-cap SQL string can't encode
        # then fail its own decoder.
        return encode_uint64(self.db_id) + encode_text(
            self.sql, max_size=_MAX_TEXT_VALUE_SIZE, label="SQL"
        )

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "PrepareRequest":
        if schema not in (0, 1):
            raise DecodeError(f"PrepareRequest unsupported schema version {schema}")
        view = memoryview(data)
        db_id = decode_uint64(view)
        sql, consumed = decode_text(view[8:], max_size=_MAX_TEXT_VALUE_SIZE, label="SQL")
        offset = 8 + consumed
        if offset != len(data):
            raise DecodeError(f"PrepareRequest has {len(data) - offset} trailing bytes")
        return cls(db_id, sql, schema=schema)


@final
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
    # Round-trip hint: preserves the decoded schema byte so re-encode is
    # byte-identical even when C used schema=1 with <=255 params (which
    # the count heuristic would downgrade to schema=0).
    _decoded_schema: int | None = field(default=None, repr=False, compare=False)
    # Round-trip hint: True iff the decoded frame carried an explicit
    # empty-params header (C-style 8 zero bytes) vs. omitted it (Go-style).
    _decoded_empty_header: bool | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_uint32("db_id", self.db_id)
        _validate_uint32("stmt_id", self.stmt_id)
        _validate_decoded_schema(self._decoded_schema, len(self.params))
        _validate_decoded_empty_header(self._decoded_empty_header, len(self.params))

    @override
    def _get_schema(self) -> int:
        if self._decoded_schema is not None:
            return self._decoded_schema
        return 1 if len(self.params) > 255 else 0

    @override
    def encode_body(self) -> bytes:
        schema = self._get_schema()
        result = encode_uint32(self.db_id) + encode_uint32(self.stmt_id)
        result += encode_params_tuple(
            self.params,
            schema=schema,
            buffer_offset=len(result),
            emit_empty_header=bool(self._decoded_empty_header),
        )
        return result

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "ExecRequest":
        if schema not in (0, 1):
            raise DecodeError(f"ExecRequest unsupported schema version {schema}")
        view = memoryview(data)
        db_id = decode_uint32(view)
        stmt_id = decode_uint32(view[4:])
        params, consumed = decode_params_tuple(view[8:], schema=schema, buffer_offset=8)
        offset = 8 + consumed
        if offset != len(data):
            raise DecodeError(f"ExecRequest has {len(data) - offset} trailing bytes")
        empty_header = (not params) and consumed > 0
        return cls(
            db_id,
            stmt_id,
            params,
            _decoded_schema=schema,
            _decoded_empty_header=empty_header,
        )


@final
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
    # See ExecRequest._decoded_empty_header.
    _decoded_empty_header: bool | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_uint32("db_id", self.db_id)
        _validate_uint32("stmt_id", self.stmt_id)
        _validate_decoded_schema(self._decoded_schema, len(self.params))
        _validate_decoded_empty_header(self._decoded_empty_header, len(self.params))

    @override
    def _get_schema(self) -> int:
        if self._decoded_schema is not None:
            return self._decoded_schema
        return 1 if len(self.params) > 255 else 0

    @override
    def encode_body(self) -> bytes:
        schema = self._get_schema()
        result = encode_uint32(self.db_id) + encode_uint32(self.stmt_id)
        result += encode_params_tuple(
            self.params,
            schema=schema,
            buffer_offset=len(result),
            emit_empty_header=bool(self._decoded_empty_header),
        )
        return result

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "QueryRequest":
        if schema not in (0, 1):
            raise DecodeError(f"QueryRequest unsupported schema version {schema}")
        view = memoryview(data)
        db_id = decode_uint32(view)
        stmt_id = decode_uint32(view[4:])
        params, consumed = decode_params_tuple(view[8:], schema=schema, buffer_offset=8)
        offset = 8 + consumed
        if offset != len(data):
            raise DecodeError(f"QueryRequest has {len(data) - offset} trailing bytes")
        empty_header = (not params) and consumed > 0
        return cls(
            db_id,
            stmt_id,
            params,
            _decoded_schema=schema,
            _decoded_empty_header=empty_header,
        )


@final
@dataclass
class FinalizeRequest(Message):
    """Finalize (close) a prepared statement.

    Body: uint32 db_id, uint32 stmt_id
    """

    MSG_TYPE: ClassVar[int] = RequestType.FINALIZE

    db_id: int
    stmt_id: int

    def __post_init__(self) -> None:
        _validate_uint32("db_id", self.db_id)
        _validate_uint32("stmt_id", self.stmt_id)

    @override
    def encode_body(self) -> bytes:
        return encode_uint32(self.db_id) + encode_uint32(self.stmt_id)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "FinalizeRequest":
        if schema != 0:
            raise DecodeError(f"FinalizeRequest unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"FinalizeRequest body must be 8 bytes, got {len(data)}")
        db_id = decode_uint32(data)
        stmt_id = decode_uint32(data[4:])
        return cls(db_id, stmt_id)


@final
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
    # See ExecRequest._decoded_empty_header.
    _decoded_empty_header: bool | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_uint64("db_id", self.db_id)
        if not isinstance(self.sql, str):
            raise EncodeError(f"ExecSqlRequest.sql must be str, got {type(self.sql).__name__}")
        _validate_decoded_schema(self._decoded_schema, len(self.params))
        _validate_decoded_empty_header(self._decoded_empty_header, len(self.params))

    @override
    def _get_schema(self) -> int:
        if self._decoded_schema is not None:
            return self._decoded_schema
        return 1 if len(self.params) > 255 else 0

    @override
    def encode_body(self) -> bytes:
        schema = self._get_schema()
        result = encode_uint64(self.db_id)
        # Symmetric cap with decode (see PrepareRequest.encode_body).
        result += encode_text(self.sql, max_size=_MAX_TEXT_VALUE_SIZE, label="SQL")
        result += encode_params_tuple(
            self.params,
            schema=schema,
            buffer_offset=len(result),
            emit_empty_header=bool(self._decoded_empty_header),
        )
        return result

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "ExecSqlRequest":
        if schema not in (0, 1):
            raise DecodeError(f"ExecSqlRequest unsupported schema version {schema}")
        view = memoryview(data)
        db_id = decode_uint64(view)
        sql, offset = decode_text(view[8:], max_size=_MAX_TEXT_VALUE_SIZE, label="SQL")
        offset += 8
        params, consumed = decode_params_tuple(view[offset:], schema=schema, buffer_offset=offset)
        offset += consumed
        if offset != len(data):
            raise DecodeError(f"ExecSqlRequest has {len(data) - offset} trailing bytes")
        empty_header = (not params) and consumed > 0
        return cls(
            db_id,
            sql,
            params,
            _decoded_schema=schema,
            _decoded_empty_header=empty_header,
        )


@final
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
    # See ExecRequest._decoded_empty_header.
    _decoded_empty_header: bool | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_uint64("db_id", self.db_id)
        if not isinstance(self.sql, str):
            raise EncodeError(f"QuerySqlRequest.sql must be str, got {type(self.sql).__name__}")
        _validate_decoded_schema(self._decoded_schema, len(self.params))
        _validate_decoded_empty_header(self._decoded_empty_header, len(self.params))

    @override
    def _get_schema(self) -> int:
        if self._decoded_schema is not None:
            return self._decoded_schema
        return 1 if len(self.params) > 255 else 0

    @override
    def encode_body(self) -> bytes:
        schema = self._get_schema()
        result = encode_uint64(self.db_id)
        # Symmetric cap with decode (see PrepareRequest.encode_body).
        result += encode_text(self.sql, max_size=_MAX_TEXT_VALUE_SIZE, label="SQL")
        result += encode_params_tuple(
            self.params,
            schema=schema,
            buffer_offset=len(result),
            emit_empty_header=bool(self._decoded_empty_header),
        )
        return result

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "QuerySqlRequest":
        if schema not in (0, 1):
            raise DecodeError(f"QuerySqlRequest unsupported schema version {schema}")
        view = memoryview(data)
        db_id = decode_uint64(view)
        sql, offset = decode_text(view[8:], max_size=_MAX_TEXT_VALUE_SIZE, label="SQL")
        offset += 8
        params, consumed = decode_params_tuple(view[offset:], schema=schema, buffer_offset=offset)
        offset += consumed
        if offset != len(data):
            raise DecodeError(f"QuerySqlRequest has {len(data) - offset} trailing bytes")
        empty_header = (not params) and consumed > 0
        return cls(
            db_id,
            sql,
            params,
            _decoded_schema=schema,
            _decoded_empty_header=empty_header,
        )


@final
@dataclass
class InterruptRequest(Message):
    """Interrupt the current operation. Body: uint64 db_id (server-ignored).

    Interrupt is per-connection, not per-database; the server ignores
    db_id, so it defaults to 0.
    """

    MSG_TYPE: ClassVar[int] = RequestType.INTERRUPT

    db_id: int = 0

    def __post_init__(self) -> None:
        _validate_uint64("db_id", self.db_id)

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.db_id)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "InterruptRequest":
        if schema != 0:
            raise DecodeError(f"InterruptRequest unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"InterruptRequest body must be 8 bytes, got {len(data)}")
        db_id = decode_uint64(data)
        return cls(db_id)


@dataclass
class _ConnectRequest(Message):
    """CONNECT Raft-transport frame (private). Body: uint64 node_id, text address.

    CONNECT (type 11) is a Raft-transport frame, NOT a gateway request:
    a real gateway rejects it over the client-server link, so it is kept
    private and out of the codec registry / __all__ (for test harnesses
    only). Layout anchor is the C Raft TCP handshake (uv_tcp_listen.c /
    uv_tcp_connect.c); Go has no equivalent directive.
    """

    MSG_TYPE: ClassVar[int] = RequestType.CONNECT

    node_id: int
    address: str

    def __post_init__(self) -> None:
        _validate_uint64("node_id", self.node_id)
        if not isinstance(self.address, str):
            raise EncodeError(
                f"_ConnectRequest.address must be str, got {type(self.address).__name__}"
            )

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id) + encode_text(
            self.address, max_size=_MAX_ADDRESS_SIZE, label="connect address"
        )

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "_ConnectRequest":
        if schema != 0:
            raise DecodeError(f"_ConnectRequest unsupported schema version {schema}")
        node_id = decode_uint64(data)
        address, consumed = decode_text(
            data[8:], max_size=_MAX_ADDRESS_SIZE, label="connect address"
        )
        offset = 8 + consumed
        if offset != len(data):
            raise DecodeError(f"_ConnectRequest has {len(data) - offset} trailing bytes")
        return cls(node_id, address)


@final
@dataclass
class AddRequest(Message):
    """Add a node to the cluster. Body: uint64 node_id, text address."""

    MSG_TYPE: ClassVar[int] = RequestType.ADD

    node_id: int
    address: str

    def __post_init__(self) -> None:
        _validate_uint64("node_id", self.node_id)
        if not isinstance(self.address, str):
            raise EncodeError(f"AddRequest.address must be str, got {type(self.address).__name__}")

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id) + encode_text(
            self.address, max_size=_MAX_ADDRESS_SIZE, label="add address"
        )

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "AddRequest":
        if schema != 0:
            raise DecodeError(f"AddRequest unsupported schema version {schema}")
        node_id = decode_uint64(data)
        address, consumed = decode_text(data[8:], max_size=_MAX_ADDRESS_SIZE, label="add address")
        offset = 8 + consumed
        if offset != len(data):
            raise DecodeError(f"AddRequest has {len(data) - offset} trailing bytes")
        return cls(node_id, address)


@final
@dataclass
class AssignRequest(Message):
    """Assign a role to a node, or promote a node (legacy).

    ASSIGN and PROMOTE share type code 13, distinguished by body size:
    PROMOTE (legacy) is 1 word (uint64 node_id); ASSIGN is 2 words
    (node_id + role).

    Legacy round-trip is a deliberate one-way upgrade: a 1-word PROMOTE
    decodes to role=VOTER (its upstream semantics) and re-encodes to the
    modern 2-word ASSIGN. Use encode_body_legacy() to re-emit the 1-word
    shape.

    Role narrowing deliberately diverges from C: unknown roles raise
    DecodeError rather than being silently folded to VOTER, so a future
    role code isn't masked during a mixed-version rollout.

    Bare AssignRequest(node_id=N) (no role) is rejected at construction;
    the legacy 1-word path requires role=None plus _legacy_intent=True.
    """

    MSG_TYPE: ClassVar[int] = RequestType.ASSIGN

    node_id: int
    role: NodeRole | int | None = None
    # True opts into the legacy 1-word PROMOTE encode path with role=None;
    # the default False makes bare AssignRequest(node_id=N) fail at construction.
    _legacy_intent: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_uint64("node_id", self.node_id)
        if self.role is None and not self._legacy_intent:
            raise EncodeError(
                "AssignRequest requires role=NodeRole.VOTER/STANDBY/SPARE for the "
                "modern ASSIGN body. Bare AssignRequest(node_id=N) is rejected at "
                "construction so the typo doesn't silently surface as a cryptic "
                "encode-time error. Callers that genuinely need the legacy 1-word "
                "PROMOTE shape must pass _legacy_intent=True and call "
                "encode_body_legacy() explicitly."
            )
        if self.role is not None and not isinstance(self.role, NodeRole):
            # Coerce bare ints to NodeRole and reject unknown values; an existing
            # NodeRole is already a valid uint64 so it needs no validation.
            _validate_uint64("role", self.role)
            try:
                coerced = NodeRole(self.role)
            except ValueError as e:
                raise EncodeError(f"AssignRequest: unknown role {self.role}") from e
            # Post-init coercion idiom; raises FrozenInstanceError if this
            # is ever made frozen=True (then move coercion into __init__).
            object.__setattr__(self, "role", coerced)

    @override
    def encode_body(self) -> bytes:
        result = encode_uint64(self.node_id)
        if self.role is not None:
            result += encode_uint64(int(self.role))
            return result
        # role=None is rejected here so a forgotten role= doesn't silently
        # downgrade to the legacy 1-word shape; use encode_body_legacy() for that.
        raise EncodeError(
            "AssignRequest with role=None cannot be encoded via encode_body — "
            "modern dqlite servers and Go-dqlite always send both node_id and role. "
            "Use role=NodeRole.VOTER (or another role) for the modern ASSIGN body, "
            "or call encode_body_legacy() explicitly for the legacy PROMOTE shape."
        )

    def encode_body_legacy(self) -> bytes:
        """Encode as legacy PROMOTE body (1 word: node_id only).

        Unlike LeaderResponse.encode_body_legacy, this silently drops
        role (the legacy body has no role field); the caller opted into
        that loss.
        """
        return encode_uint64(self.node_id)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "AssignRequest":
        if schema != 0:
            raise DecodeError(f"AssignRequest unsupported schema version {schema}")
        if len(data) == 8:
            # Legacy PROMOTE: no role on the wire, map to VOTER per upstream
            # semantics (role=None wouldn't round-trip through encode_body).
            node_id = decode_uint64(data)
            return cls(node_id, NodeRole.VOTER)
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


@final
@dataclass
class RemoveRequest(Message):
    """Remove a node from the cluster. Body: uint64 node_id."""

    MSG_TYPE: ClassVar[int] = RequestType.REMOVE

    node_id: int

    def __post_init__(self) -> None:
        _validate_uint64("node_id", self.node_id)

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "RemoveRequest":
        if schema != 0:
            raise DecodeError(f"RemoveRequest unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"RemoveRequest body must be 8 bytes, got {len(data)}")
        node_id = decode_uint64(data)
        return cls(node_id)


@final
@dataclass
class DumpRequest(Message):
    """Request a database dump. Body: text name."""

    MSG_TYPE: ClassVar[int] = RequestType.DUMP

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise EncodeError(f"DumpRequest.name must be str, got {type(self.name).__name__}")

    @override
    def encode_body(self) -> bytes:
        # Cap at the C ceiling: handle_dump's 1024-byte WAL-filename buffer
        # leaves 1019 bytes after the "-wal" suffix and NUL, so a longer name
        # would yield a truncated WAL path. Fail fast here instead.
        return encode_text(self.name, max_size=_MAX_DUMP_FILENAME_SIZE, label="database name")

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "DumpRequest":
        if schema != 0:
            raise DecodeError(f"DumpRequest unsupported schema version {schema}")
        # Symmetric with encode_body: refuse a name exceeding the C ceiling.
        name, consumed = decode_text(data, max_size=_MAX_DUMP_FILENAME_SIZE, label="database name")
        if consumed != len(data):
            raise DecodeError(f"DumpRequest has {len(data) - consumed} trailing bytes")
        return cls(name)


@final
@dataclass
class ClusterRequest(Message):
    """Request cluster information. Body: uint64 format.

    format=0 (V0, no role) is valid upstream but unsupported here because
    ServersResponse only decodes V1; fresh construction with format=0 is
    rejected. A V0 frame decoded via decode_body carries _decoded=True
    and re-encodes byte-identically for proxy/replay tooling.
    """

    MSG_TYPE: ClassVar[int] = RequestType.CLUSTER

    format: int = 1
    # Decoded-V0 sentinel; True exempts format=0 from the construction-time reject.
    _decoded: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_uint64("format", self.format)
        if self.format == 0 and not self._decoded:
            raise EncodeError(
                "ClusterRequest format=0 (V0) is valid in upstream dqlite but "
                "not implemented in this Python library: ServersResponse only "
                "decodes V1 (with node role fields). Use format=1."
            )
        if self.format not in (0, 1):
            raise EncodeError(
                f"ClusterRequest format must be 0 (V0) or 1 (V1); upstream "
                f"defines only those two values. Got {self.format}."
            )

    @override
    def encode_body(self) -> bytes:
        if self.format == 0 and not self._decoded:
            raise EncodeError(
                "ClusterRequest format=0 (V0) is valid in upstream dqlite but "
                "not implemented in this Python library: ServersResponse only "
                "decodes V1. Use format=1, or construct via decode_body so the "
                "_decoded sentinel exempts the encode-time reject for "
                "round-trip / replay use cases."
            )
        return encode_uint64(self.format)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "ClusterRequest":
        if schema != 0:
            raise DecodeError(f"ClusterRequest unsupported schema version {schema}")
        # Decode accepts V0 so replay tooling can round-trip it; only fresh
        # format=0 construction is rejected.
        if len(data) != 8:
            raise DecodeError(f"ClusterRequest body must be 8 bytes, got {len(data)}")
        format_val = decode_uint64(data)
        if format_val not in (0, 1):
            raise DecodeError(
                f"ClusterRequest format must be 0 (V0) or 1 (V1); upstream "
                f"defines only those two values. Got {format_val}."
            )
        return cls(format=format_val, _decoded=True)


@final
@dataclass
class TransferRequest(Message):
    """Request leadership transfer. Body: uint64 target_node_id."""

    MSG_TYPE: ClassVar[int] = RequestType.TRANSFER

    target_node_id: int

    def __post_init__(self) -> None:
        _validate_uint64("target_node_id", self.target_node_id)

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.target_node_id)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "TransferRequest":
        if schema != 0:
            raise DecodeError(f"TransferRequest unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"TransferRequest body must be 8 bytes, got {len(data)}")
        target_node_id = decode_uint64(data)
        return cls(target_node_id)


@final
@dataclass
class DescribeRequest(Message):
    """Request database schema description. Body: uint64 format.

    Upstream defines only format=0; fresh construction rejects anything
    else client-side. decode_body(strict=False) admits any format (with
    _decoded=True) for inspecting captured/synthetic traffic.
    """

    MSG_TYPE: ClassVar[int] = RequestType.DESCRIBE

    format: int = 0
    # Decoded-non-zero sentinel; set by the strict=False decode path to exempt
    # the V0-only construction gate.
    _decoded: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_uint64("format", self.format)
        if self.format != 0 and not self._decoded:
            raise EncodeError(
                f"DescribeRequest format must be 0 (V0); upstream rejects "
                f"anything else with SQLITE_PROTOCOL. Got {self.format}."
            )

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.format)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0, *, strict: bool = True) -> "DescribeRequest":
        if schema != 0:
            raise DecodeError(f"DescribeRequest unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"DescribeRequest body must be 8 bytes, got {len(data)}")
        format_val = decode_uint64(data)
        if format_val != 0:
            if strict:
                raise DecodeError(f"DescribeRequest format must be 0 (V0); got {format_val}")
            return cls(format=format_val, _decoded=True)
        return cls(format_val)


@final
@dataclass
class WeightRequest(Message):
    """Set node weight for leader election. Body: uint64 weight."""

    MSG_TYPE: ClassVar[int] = RequestType.WEIGHT

    weight: int

    def __post_init__(self) -> None:
        _validate_uint64("weight", self.weight)

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.weight)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "WeightRequest":
        if schema != 0:
            raise DecodeError(f"WeightRequest unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"WeightRequest body must be 8 bytes, got {len(data)}")
        weight = decode_uint64(data)
        return cls(weight)
