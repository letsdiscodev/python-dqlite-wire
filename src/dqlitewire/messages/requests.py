"""Client to server request messages."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Self, final, override

from dqlitewire.constants import NodeRole, RequestType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.limits import (
    MAX_ADDRESS_SIZE,
    MAX_DUMP_FILENAME_SIZE,
    MAX_FILENAME_SIZE,
    MAX_PARAM_COUNT,
    MAX_TEXT_VALUE_SIZE,
)
from dqlitewire.messages.base import Message, require_consumed, require_length, require_schema
from dqlitewire.tuples import decode_params_tuple, encode_params_tuple
from dqlitewire.types import (
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


def _require_str(cls: type, name: str, value: object) -> None:
    if not isinstance(value, str):
        raise EncodeError(f"{cls.__name__}.{name} must be str, got {type(value).__name__}")


# -- single-word requests ------------------------------------------------------------


class _WordRequest(Message):
    """Body: one uint64, held in the dataclass field named ``_FIELD``."""

    _FIELD: ClassVar[str]

    def __post_init__(self) -> None:
        _validate_uint64(self._FIELD, getattr(self, self._FIELD))

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(getattr(self, self._FIELD))

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> Self:
        require_schema(cls, schema)
        require_length(cls, data, 8)
        return cls(decode_uint64(data))  # type: ignore[call-arg]


@final
@dataclass
class LeaderRequest(Message):
    """Body: uint64 reserved (0)."""

    MSG_TYPE: ClassVar[int] = RequestType.LEADER

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(0)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "LeaderRequest":
        require_schema(cls, schema)
        require_length(cls, data, 8)
        if (reserved := decode_uint64(data)) != 0:
            raise DecodeError(f"LeaderRequest reserved field must be 0, got {reserved}")
        return cls()


@final
@dataclass
class ClientRequest(_WordRequest):
    MSG_TYPE: ClassVar[int] = RequestType.CLIENT
    _FIELD: ClassVar[str] = "client_id"
    client_id: int


@dataclass
class _HeartbeatRequest(_WordRequest):
    """Private: no upstream peer accepts a heartbeat frame; kept for test servers."""

    MSG_TYPE: ClassVar[int] = RequestType.HEARTBEAT
    _FIELD: ClassVar[str] = "timestamp"
    timestamp: int


@final
@dataclass
class InterruptRequest(_WordRequest):
    """The server ignores ``db_id`` (interrupts are per connection)."""

    MSG_TYPE: ClassVar[int] = RequestType.INTERRUPT
    _FIELD: ClassVar[str] = "db_id"
    db_id: int = 0


@final
@dataclass
class RemoveRequest(_WordRequest):
    MSG_TYPE: ClassVar[int] = RequestType.REMOVE
    _FIELD: ClassVar[str] = "node_id"
    node_id: int


@final
@dataclass
class TransferRequest(_WordRequest):
    MSG_TYPE: ClassVar[int] = RequestType.TRANSFER
    _FIELD: ClassVar[str] = "target_node_id"
    target_node_id: int


@final
@dataclass
class WeightRequest(_WordRequest):
    MSG_TYPE: ClassVar[int] = RequestType.WEIGHT
    _FIELD: ClassVar[str] = "weight"
    weight: int


# -- node-info requests -------------------------------------------------------------


class _NodeAddressRequest(Message):
    """Body: uint64 node_id, text address."""

    _LABEL: ClassVar[str]
    node_id: int
    address: str

    def __post_init__(self) -> None:
        _validate_uint64("node_id", self.node_id)
        _require_str(type(self), "address", self.address)

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id) + encode_text(
            self.address, max_size=MAX_ADDRESS_SIZE, label=self._LABEL
        )

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> Self:
        require_schema(cls, schema)
        node_id = decode_uint64(data)
        address, consumed = decode_text(data[8:], max_size=MAX_ADDRESS_SIZE, label=cls._LABEL)
        require_consumed(cls, data, 8 + consumed)
        return cls(node_id, address)  # type: ignore[call-arg]


@final
@dataclass
class AddRequest(_NodeAddressRequest):
    """Adds the node as a spare; assign a role afterwards."""

    MSG_TYPE: ClassVar[int] = RequestType.ADD
    _LABEL: ClassVar[str] = "add address"
    node_id: int
    address: str


@dataclass
class _ConnectRequest(_NodeAddressRequest):
    """Private: the raft transport handshake frame, which a gateway rejects; kept for
    test servers."""

    MSG_TYPE: ClassVar[int] = RequestType.CONNECT
    _LABEL: ClassVar[str] = "connect address"
    node_id: int
    address: str


# -- statement requests -------------------------------------------------------------


def _validate_decoded_schema(decoded_schema: int | None, param_count: int) -> None:
    if decoded_schema is None:
        return
    if decoded_schema not in (0, 1):
        raise EncodeError(f"_decoded_schema must be 0, 1, or None; got {decoded_schema}")
    if decoded_schema == 0 and param_count > 255:
        raise EncodeError(
            f"_decoded_schema=0 (V0 tuple format) supports at most 255 parameters; "
            f"got {param_count}"
        )
    if decoded_schema == 1 and param_count > MAX_PARAM_COUNT:
        raise EncodeError(
            f"_decoded_schema=1 (V1 tuple format) supports at most "
            f"{MAX_PARAM_COUNT} parameters; got {param_count}"
        )


def _validate_decoded_empty_header(decoded_empty_header: bool | None, param_count: int) -> None:
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


class _ParamsRequest(Message):
    """A prefix (statement or SQL identification) followed by a params tuple.

    Schema 0 (uint8 count) is used up to 255 parameters, schema 1 (uint32 count)
    above. The two ``_decoded_*`` hints let a decoded frame re-encode byte-identically:
    the schema byte C used, and whether it wrote an explicit empty-params header.
    """

    params: Sequence[WireInput]
    _decoded_schema: int | None
    _decoded_empty_header: bool | None

    def _validate_params_hints(self) -> None:
        _validate_decoded_schema(self._decoded_schema, len(self.params))
        _validate_decoded_empty_header(self._decoded_empty_header, len(self.params))

    @override
    def _get_schema(self) -> int:
        if self._decoded_schema is not None:
            return self._decoded_schema
        return 1 if len(self.params) > 255 else 0

    def _encode_prefix(self) -> bytes:
        raise NotImplementedError

    @classmethod
    def _decode_prefix(cls, view: memoryview) -> tuple[tuple[object, ...], int]:
        """``(constructor args, bytes consumed)``."""
        raise NotImplementedError

    @override
    def encode_body(self) -> bytes:
        prefix = self._encode_prefix()
        return prefix + encode_params_tuple(
            self.params,
            schema=self._get_schema(),
            buffer_offset=len(prefix),
            emit_empty_header=bool(self._decoded_empty_header),
        )

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> Self:
        require_schema(cls, schema, (0, 1))
        view = memoryview(data)
        args, offset = cls._decode_prefix(view)
        params, consumed = decode_params_tuple(view[offset:], schema=schema, buffer_offset=offset)
        require_consumed(cls, data, offset + consumed)
        return cls(  # type: ignore[call-arg]
            *args,
            params,
            _decoded_schema=schema,
            _decoded_empty_header=(not params) and consumed > 0,
        )


class _PreparedStatementRequest(_ParamsRequest):
    """Prefix: uint32 db_id, uint32 stmt_id."""

    db_id: int
    stmt_id: int

    def __post_init__(self) -> None:
        _validate_uint32("db_id", self.db_id)
        _validate_uint32("stmt_id", self.stmt_id)
        self._validate_params_hints()

    @override
    def _encode_prefix(self) -> bytes:
        return encode_uint32(self.db_id) + encode_uint32(self.stmt_id)

    @classmethod
    @override
    def _decode_prefix(cls, view: memoryview) -> tuple[tuple[object, ...], int]:
        return (decode_uint32(view), decode_uint32(view[4:])), 8


class _SqlTextRequest(_ParamsRequest):
    """Prefix: uint64 db_id, text sql."""

    db_id: int
    sql: str

    def __post_init__(self) -> None:
        _validate_uint64("db_id", self.db_id)
        _require_str(type(self), "sql", self.sql)
        self._validate_params_hints()

    @override
    def _encode_prefix(self) -> bytes:
        return encode_uint64(self.db_id) + encode_text(
            self.sql, max_size=MAX_TEXT_VALUE_SIZE, label="SQL"
        )

    @classmethod
    @override
    def _decode_prefix(cls, view: memoryview) -> tuple[tuple[object, ...], int]:
        db_id = decode_uint64(view)
        sql, consumed = decode_text(view[8:], max_size=MAX_TEXT_VALUE_SIZE, label="SQL")
        return (db_id, sql), 8 + consumed


@final
@dataclass
class ExecRequest(_PreparedStatementRequest):
    MSG_TYPE: ClassVar[int] = RequestType.EXEC
    db_id: int
    stmt_id: int
    params: Sequence[WireInput] = field(default_factory=list)
    _decoded_schema: int | None = field(default=None, repr=False, compare=False)
    _decoded_empty_header: bool | None = field(default=None, repr=False, compare=False)


@final
@dataclass
class QueryRequest(_PreparedStatementRequest):
    MSG_TYPE: ClassVar[int] = RequestType.QUERY
    db_id: int
    stmt_id: int
    params: Sequence[WireInput] = field(default_factory=list)
    _decoded_schema: int | None = field(default=None, repr=False, compare=False)
    _decoded_empty_header: bool | None = field(default=None, repr=False, compare=False)


@final
@dataclass
class ExecSqlRequest(_SqlTextRequest):
    MSG_TYPE: ClassVar[int] = RequestType.EXEC_SQL
    db_id: int
    sql: str
    params: Sequence[WireInput] = field(default_factory=list)
    _decoded_schema: int | None = field(default=None, repr=False, compare=False)
    _decoded_empty_header: bool | None = field(default=None, repr=False, compare=False)


@final
@dataclass
class QuerySqlRequest(_SqlTextRequest):
    MSG_TYPE: ClassVar[int] = RequestType.QUERY_SQL
    db_id: int
    sql: str
    params: Sequence[WireInput] = field(default_factory=list)
    _decoded_schema: int | None = field(default=None, repr=False, compare=False)
    _decoded_empty_header: bool | None = field(default=None, repr=False, compare=False)


@final
@dataclass
class PrepareRequest(Message):
    """Body: uint64 db_id, text sql. ``schema=1`` asks for a V1 reply carrying the
    tail offset of multi-statement SQL (go-dqlite always sends 0)."""

    MSG_TYPE: ClassVar[int] = RequestType.PREPARE
    db_id: int
    sql: str
    schema: int = 0

    def __post_init__(self) -> None:
        _validate_uint64("db_id", self.db_id)
        _require_str(type(self), "sql", self.sql)
        if self.schema not in (0, 1):
            raise EncodeError(f"schema must be 0 or 1, got {self.schema}")

    @override
    def _get_schema(self) -> int:
        return self.schema

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.db_id) + encode_text(
            self.sql, max_size=MAX_TEXT_VALUE_SIZE, label="SQL"
        )

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "PrepareRequest":
        require_schema(cls, schema, (0, 1))
        view = memoryview(data)
        db_id = decode_uint64(view)
        sql, consumed = decode_text(view[8:], max_size=MAX_TEXT_VALUE_SIZE, label="SQL")
        require_consumed(cls, data, 8 + consumed)
        return cls(db_id, sql, schema=schema)


@final
@dataclass
class FinalizeRequest(Message):
    """Body: uint32 db_id, uint32 stmt_id."""

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
        require_schema(cls, schema)
        require_length(cls, data, 8)
        return cls(decode_uint32(data), decode_uint32(data[4:]))


# -- other requests ---------------------------------------------------------------------


@final
@dataclass
class OpenRequest(Message):
    """Body: text name, uint64 flags, text vfs. The server ignores flags and vfs."""

    MSG_TYPE: ClassVar[int] = RequestType.OPEN
    name: str
    flags: int = 0
    vfs: str = ""

    def __post_init__(self) -> None:
        _require_str(type(self), "name", self.name)
        _require_str(type(self), "vfs", self.vfs)
        _validate_uint64("flags", self.flags)

    @override
    def encode_body(self) -> bytes:
        return (
            encode_text(self.name, max_size=MAX_FILENAME_SIZE, label="database name")
            + encode_uint64(self.flags)
            + encode_text(self.vfs, max_size=MAX_FILENAME_SIZE, label="vfs name")
        )

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "OpenRequest":
        require_schema(cls, schema)
        view = memoryview(data)
        name, offset = decode_text(view, max_size=MAX_FILENAME_SIZE, label="database name")
        flags = decode_uint64(view[offset:], label="OpenRequest.flags")
        offset += 8
        vfs, consumed = decode_text(view[offset:], max_size=MAX_FILENAME_SIZE, label="vfs name")
        require_consumed(cls, data, offset + consumed)
        return cls(name, flags, vfs)


@final
@dataclass
class DumpRequest(Message):
    """Body: text name, capped at what the C gateway's WAL-name buffer can hold."""

    MSG_TYPE: ClassVar[int] = RequestType.DUMP
    name: str

    def __post_init__(self) -> None:
        _require_str(type(self), "name", self.name)

    @override
    def encode_body(self) -> bytes:
        return encode_text(self.name, max_size=MAX_DUMP_FILENAME_SIZE, label="database name")

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "DumpRequest":
        require_schema(cls, schema)
        name, consumed = decode_text(data, max_size=MAX_DUMP_FILENAME_SIZE, label="database name")
        require_consumed(cls, data, consumed)
        return cls(name)


@final
@dataclass
class AssignRequest(Message):
    """Body: uint64 node_id, uint64 role. Type 13 is shared with the legacy one-word
    PROMOTE, which decodes to role VOTER and re-encodes in the modern shape;
    :meth:`encode_body_legacy` emits the one-word form for callers that opted in
    with ``_legacy_intent``. Unknown roles are rejected rather than folded to VOTER."""

    MSG_TYPE: ClassVar[int] = RequestType.ASSIGN
    node_id: int
    role: NodeRole | int | None = None
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
            _validate_uint64("role", self.role)
            try:
                coerced = NodeRole(self.role)
            except ValueError as e:
                raise EncodeError(f"AssignRequest: unknown role {self.role}") from e
            object.__setattr__(self, "role", coerced)

    @override
    def encode_body(self) -> bytes:
        if self.role is None:
            raise EncodeError(
                "AssignRequest with role=None cannot be encoded via encode_body — "
                "modern dqlite servers and Go-dqlite always send both node_id and role. "
                "Use role=NodeRole.VOTER (or another role) for the modern ASSIGN body, "
                "or call encode_body_legacy() explicitly for the legacy PROMOTE shape."
            )
        return encode_uint64(self.node_id) + encode_uint64(int(self.role))

    def encode_body_legacy(self) -> bytes:
        """The one-word PROMOTE body; the role is dropped."""
        return encode_uint64(self.node_id)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "AssignRequest":
        require_schema(cls, schema)
        if len(data) == 8:
            return cls(decode_uint64(data), NodeRole.VOTER)
        if len(data) == 16:
            raw = decode_uint64(data[8:])
            try:
                role = NodeRole(raw)
            except ValueError as e:
                raise DecodeError(f"AssignRequest: unknown role {raw}") from e
            return cls(decode_uint64(data), role)
        raise DecodeError(
            f"AssignRequest body must be 8 (PROMOTE) or 16 (ASSIGN) bytes, got {len(data)}"
        )


@final
@dataclass
class ClusterRequest(Message):
    """Body: uint64 format. Only format 1 (with roles) can be constructed here, since
    ServersResponse decodes only V1; a decoded format-0 frame carries ``_decoded`` so
    replay tooling can re-encode it."""

    MSG_TYPE: ClassVar[int] = RequestType.CLUSTER
    format: int = 1
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
        return encode_uint64(self.format)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "ClusterRequest":
        require_schema(cls, schema)
        require_length(cls, data, 8)
        format_val = decode_uint64(data)
        if format_val not in (0, 1):
            raise DecodeError(
                f"ClusterRequest format must be 0 (V0) or 1 (V1); upstream "
                f"defines only those two values. Got {format_val}."
            )
        return cls(format=format_val, _decoded=True)


@final
@dataclass
class DescribeRequest(Message):
    """Body: uint64 format; upstream defines only 0. ``decode_body(strict=False)`` admits
    other values (flagged ``_decoded``) for inspecting captured traffic."""

    MSG_TYPE: ClassVar[int] = RequestType.DESCRIBE
    format: int = 0
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
        require_schema(cls, schema)
        require_length(cls, data, 8)
        format_val = decode_uint64(data)
        if format_val != 0:
            if strict:
                raise DecodeError(f"DescribeRequest format must be 0 (V0); got {format_val}")
            return cls(format=format_val, _decoded=True)
        return cls(format_val)
