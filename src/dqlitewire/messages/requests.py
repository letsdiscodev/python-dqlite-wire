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
    """Validate the optional ``_decoded_schema`` round-trip hint shared by
    ExecRequest / QueryRequest / ExecSqlRequest / QuerySqlRequest.

    The field is a private hook used to reproduce wire bytes identically
    on re-encode (mock-server / proxy use cases) — it lets a request
    carry schema=1 with ≤255 params even though the count heuristic
    would pick schema=0. ``None`` (auto-select), ``0``, and ``1`` are the
    only legitimate values; schema=0 caps params at 255 (the V0 tuple
    format's uint8 count byte) and schema=1 caps params at
    ``_MAX_PARAM_COUNT`` (SQLite's ``SQLITE_MAX_VARIABLE_NUMBER``
    engine-side bind cap, see ``tuples.py:_MAX_PARAM_COUNT``; the V1
    on-wire count field is a uint32, not uint16).

    Both caps are enforced here at construction time so a caller building
    a malformed request gets an actionable error at the construction
    site, not deep inside ``encode_params_tuple`` at first encode. The
    encode-time cap inside ``encode_params_tuple`` is retained as
    defense-in-depth in case the caller mutates ``_decoded_schema`` or
    ``params`` after construction.
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
    """Validate the optional ``_decoded_empty_header`` round-trip hint
    shared by ExecRequest / QueryRequest / ExecSqlRequest /
    QuerySqlRequest.

    The field is a private hook used to reproduce empty-params wire
    bytes identically on re-encode. ``None`` (caller-originated → Go-
    style omission), ``False`` (decoded a Go-style frame, no header
    bytes consumed), and ``True`` (decoded a C-style frame with an
    explicit 8-byte zero header) are the only legitimate values, AND
    ``True`` only makes sense with empty ``params`` — the field's
    documented contract is "Set by ``decode_body`` only when
    ``params`` is empty AND the inbound frame had bytes for the
    tuple". A caller constructing a request with
    ``_decoded_empty_header=True`` and non-empty params silently
    produces an internally inconsistent state today (the encoder's
    ``emit_empty_header=True`` arm is a no-op when ``params`` is
    non-empty, so the wire bytes look fine, but the field disagrees
    with the bytes). A future refactor that broadens the
    ``emit_empty_header=True`` arm to "always emit the 8-byte header"
    would then produce malformed wire bytes only when this caller
    misuse pattern occurs. Pin the invariant at construction time so
    the field stays in sync with the wire-byte semantics it claims to
    represent. Mirrors ``_validate_decoded_schema``'s discipline.

    The type narrowing uses ``isinstance(x, bool)`` rather than ``x is
    True`` / ``x is False`` so any non-bool input — ``1``, ``"yes"``,
    ``object()`` — is rejected explicitly. Without the type check the
    downstream ``bool(self._decoded_empty_header)`` at ``encode_body``
    promotes truthy non-bool inputs to the C-style 8-byte empty-header
    emission silently, contradicting the documented ``None | False |
    True`` contract that the sibling ``_validate_decoded_schema``
    pins via membership.
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
    """Request current cluster leader address.

    Body: uint64 (reserved, unused)
    """

    MSG_TYPE: ClassVar[int] = RequestType.LEADER

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(0)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "LeaderRequest":
        if schema != 0:
            raise DecodeError(f"LeaderRequest unsupported schema version {schema}")
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


@final
@dataclass
class ClientRequest(Message):
    """Register as a client.

    Body: uint64 client_id
    """

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
    """Send heartbeat to server (private — not a public wire message).

    .. warning::

        The ``uint64 timestamp`` body shape is **speculative**. Upstream
        (``protocol.h``) reserves the type code ``DQLITE_REQUEST_HEARTBEAT
        = 2`` but defines no schema — ``REQUEST__TYPES`` in
        ``request.h`` omits ``heartbeat``, and ``gateway.c``'s
        dispatcher falls through to ``DQLITE_PARSE`` for this type. The
        Go client's heartbeat code is also commented out.

        Because no upstream peer accepts a heartbeat frame, this class
        is private: it is omitted from the ``REQUEST_TYPES`` registry in
        ``codec.py`` and from ``messages/__init__.py``'s ``__all__``.
        It remains in the source tree only for test-mock / golden-byte
        harnesses that synthesize a type-2 frame for negative-path
        coverage; callers import it via the private symbol
        ``dqlitewire.messages.requests._HeartbeatRequest``.

        This dataclass preserves the historical ``uint64 timestamp``
        layout for that compatibility purpose; if upstream ever defines
        a real schema, this body shape will need to change.

    Body: uint64 timestamp (speculative — not part of any upstream spec)
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
        # memoryview wrap so per-slice cost stays O(1) on large bodies
        # with many embedded text fields, matching the response-side
        # decoders (RowsResponse, FilesResponse, ServersResponse) that
        # already use this pattern.
        view = memoryview(data)
        name, offset = decode_text(view, max_size=_MAX_FILENAME_SIZE, label="database name")
        flags = decode_uint64(view[offset:])
        offset += 8
        vfs, consumed = decode_text(view[offset:], max_size=_MAX_FILENAME_SIZE, label="vfs name")
        offset += consumed
        if offset != len(data):
            raise DecodeError(f"OpenRequest has {len(data) - offset} trailing bytes")
        return cls(name, flags, vfs)


@final
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
        # Pass the same ``max_size`` the decoder uses by default so
        # encode/decode round-trip is symmetric — without this, an
        # over-cap SQL string serialises successfully then fails its
        # own decoder. Mirrors the cap-symmetry pattern already
        # applied to OpenRequest / DumpRequest / AddRequest /
        # _ConnectRequest text fields.
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
    # Preserves the header schema byte seen on decode so a decode →
    # re-encode round-trip emits byte-identical output even when the
    # upstream C client used schema=1 with ≤255 params (which the count
    # heuristic alone would otherwise downgrade to schema=0). Excluded
    # from repr/compare so it stays an internal round-trip hint.
    _decoded_schema: int | None = field(default=None, repr=False, compare=False)
    # Companion round-trip hint to ``_decoded_schema``. Tracks
    # whether the inbound wire frame carried an explicit empty-params
    # header (C-style: 8 zero bytes) or omitted the params tuple
    # entirely (Go-style: 0 bytes). Both shapes are valid for
    # zero-param requests but produce different wire bytes. Set by
    # ``decode_body`` only when ``params`` is empty AND the inbound
    # frame had bytes for the tuple; otherwise ``None`` (caller-
    # originated default → Go-style omission). Excluded from
    # repr/compare so it stays an internal round-trip hint.
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
        # Track whether the inbound frame had an explicit empty-params
        # header (C-style ``consumed > 0`` for ``params == []``) so a
        # decode→re-encode round-trip is byte-identical for both
        # shapes Go and C emit for empty params.
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
    # Companion round-trip hint to ``_decoded_schema``. Tracks
    # whether the inbound wire frame carried an explicit empty-params
    # header (C-style: 8 zero bytes) or omitted the params tuple
    # entirely (Go-style: 0 bytes). Both shapes are valid for
    # zero-param requests but produce different wire bytes. Set by
    # ``decode_body`` only when ``params`` is empty AND the inbound
    # frame had bytes for the tuple; otherwise ``None`` (caller-
    # originated default → Go-style omission). Excluded from
    # repr/compare so it stays an internal round-trip hint.
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
    # Companion round-trip hint to ``_decoded_schema``. Tracks
    # whether the inbound wire frame carried an explicit empty-params
    # header (C-style: 8 zero bytes) or omitted the params tuple
    # entirely (Go-style: 0 bytes). Both shapes are valid for
    # zero-param requests but produce different wire bytes. Set by
    # ``decode_body`` only when ``params`` is empty AND the inbound
    # frame had bytes for the tuple; otherwise ``None`` (caller-
    # originated default → Go-style omission). Excluded from
    # repr/compare so it stays an internal round-trip hint.
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
    # Companion round-trip hint to ``_decoded_schema``. Tracks
    # whether the inbound wire frame carried an explicit empty-params
    # header (C-style: 8 zero bytes) or omitted the params tuple
    # entirely (Go-style: 0 bytes). Both shapes are valid for
    # zero-param requests but produce different wire bytes. Set by
    # ``decode_body`` only when ``params`` is empty AND the inbound
    # frame had bytes for the tuple; otherwise ``None`` (caller-
    # originated default → Go-style omission). Excluded from
    # repr/compare so it stays an internal round-trip hint.
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
    """Interrupt the current operation.

    Body: uint64 db_id (server-ignored)

    The wire schema includes ``db_id`` for historical reasons but the
    upstream server's ``handle_interrupt`` (``dqlite-upstream/src/
    gateway.c``) does not read it — interrupt is per-connection
    routing, not per-database. Defaulting to 0 keeps the wire
    encoding stable and matches the Go reference's idiomatic
    zero-value usage so callers without a meaningful db_id in scope
    don't have to thread one through.
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
    """Establish a Raft transport connection (node-to-node).

    Body: uint64 node_id, text address

    WARNING: CONNECT (type 11) is a Raft-transport frame, NOT a gateway
    request. Upstream ``request.h``'s ``REQUEST__TYPES`` macro omits it,
    and ``gateway.c`` falls through to ``DQLITE_PARSE`` for type-11
    frames — a real dqlite gateway rejects a CONNECT request sent over
    the client-server link. The class is exposed with a private name so
    mock-server / golden-byte harnesses can still synthesise a type-11
    frame for testing, but it is intentionally absent from the public
    dispatcher (``REQUEST_TYPES`` in ``codec.py``) and the public
    ``dqlitewire.messages`` re-export list — mirroring the same
    treatment applied to ``_HeartbeatRequest``.
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
    """Add a node to the cluster.

    Body: uint64 node_id, text address
    """

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

    ASSIGN and PROMOTE share type code 13. They are distinguished by body size:
    - PROMOTE (legacy): uint64 node_id (1 word)
    - ASSIGN: uint64 node_id, uint64 role (2 words)

    **Legacy round-trip asymmetry (intentional).** A peer-emitted
    1-word PROMOTE body decodes to ``AssignRequest(node_id=N,
    role=NodeRole.VOTER)`` because PROMOTE upstream-semantically
    elevates the node to voter. Re-encoding the resulting dataclass
    via :meth:`encode_body` produces the modern 2-word ASSIGN shape
    (16 bytes). This is a deliberate one-way upgrade: the legacy
    PROMOTE wire form has no role field, so on decode we MUST pick
    one (VOTER per upstream semantics) and on re-encode the modern
    form is the only safe shape against current servers. Callers
    that genuinely need to re-emit the legacy 1-word shape (for
    relaying to an old server or for round-trip-identity tests) must
    call :meth:`encode_body_legacy` explicitly.

    **Strict-decode role narrowing diverges from C.** The Python
    decoder rejects role values outside :class:`NodeRole` with
    :class:`DecodeError`. Upstream C silently folds unknown roles to
    ``DQLITE_VOTER`` via the ``default:`` arm of
    ``translateDqliteRole`` (``dqlite-upstream/src/translate.c``,
    commented as "For backward compat with clients that don't set a
    role"). The narrowing is deliberate — silent coercion would mask
    a future role code (e.g. ``OBSERVER = 3``) added by upstream
    during a mixed-version rollout. A mock-server / proxy that needs
    C-style permissiveness can pre-decode the raw uint64 and
    construct ``AssignRequest`` with ``role=NodeRole.VOTER``
    explicitly.

    **Construction rejects bare ``role=None``.** ``AssignRequest(node_id=N)``
    without a role would be silently admitted by the dataclass and
    fail only at ``encode_body()`` — possibly far from the
    construction site. The ``__post_init__`` rejects the bare-typo
    case at construction time. Callers that genuinely need to emit
    the legacy 1-word PROMOTE shape (relays to old servers,
    round-trip-identity tests) opt in with
    ``AssignRequest(node_id=N, role=None, _legacy_intent=True)`` and
    call :meth:`encode_body_legacy`. The sentinel mirrors the
    ``_decoded_schema`` / ``_decoded_empty_header`` pattern on
    ``ExecRequest`` so the wire layer keeps a uniform vocabulary.
    """

    MSG_TYPE: ClassVar[int] = RequestType.ASSIGN

    node_id: int
    role: NodeRole | int | None = None
    # Round-trip / legacy-emit sentinel. ``True`` opts a caller into
    # the legacy 1-word PROMOTE encode path (``encode_body_legacy``)
    # with ``role=None``. ``False`` (the default) makes the bare
    # ``AssignRequest(node_id=N)`` typo fail at construction rather
    # than at encode time. Excluded from ``repr`` and equality so two
    # logically-equal requests compare equal regardless of how they
    # were built. Mirrors ``ExecRequest._decoded_schema`` and
    # ``ClusterRequest._decoded``.
    _legacy_intent: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_uint64("node_id", self.node_id)
        if self.role is None and not self._legacy_intent:
            # Reject bare ``AssignRequest(node_id=N)`` at the
            # construction site rather than at ``encode_body()``. The
            # encoder-side message is preserved as defense-in-depth for
            # callers that mutate ``role`` to ``None`` after
            # construction.
            raise EncodeError(
                "AssignRequest requires role=NodeRole.VOTER/STANDBY/SPARE for the "
                "modern ASSIGN body. Bare AssignRequest(node_id=N) is rejected at "
                "construction so the typo doesn't silently surface as a cryptic "
                "encode-time error. Callers that genuinely need the legacy 1-word "
                "PROMOTE shape must pass _legacy_intent=True and call "
                "encode_body_legacy() explicitly."
            )
        if self.role is not None:
            # Coerce bare ints to the NodeRole enum and reject unknown
            # values. Mirrors the response-side narrowing on
            # ``ServersResponse`` so an outbound assign or a
            # mock-server decode carries a validated role, not an
            # unknown integer that would silently surface in the
            # dataclass.
            if isinstance(self.role, NodeRole):
                _validate_uint64("role", int(self.role))
            else:
                _validate_uint64("role", self.role)
                try:
                    coerced = NodeRole(self.role)
                except ValueError as e:
                    raise EncodeError(f"AssignRequest: unknown role {self.role}") from e
                # ``object.__setattr__`` is the post-init coercion idiom.
                # If this dataclass is ever flipped to
                # ``@dataclass(frozen=True)`` (for hashability or
                # thread-safety), this line raises
                # ``FrozenInstanceError`` — fold the coercion into a
                # custom ``__init__`` that writes attributes once at
                # the end. Sibling ``ServersResponse.NodeInfo`` has the
                # same shape.
                object.__setattr__(self, "role", coerced)

    @override
    def encode_body(self) -> bytes:
        result = encode_uint64(self.node_id)
        if self.role is not None:
            result += encode_uint64(int(self.role))
            return result
        # Legacy PROMOTE shape (1-word body) is emitted only by very
        # old dqlite clients. Go-dqlite's canonical EncodeAssign
        # never produces it; modern servers always expect the
        # 2-word ASSIGN body. Reject explicitly so a typo like
        # ``AssignRequest(node_id=42)`` (forgetting role=) doesn't
        # silently downgrade to the legacy shape and surface as a
        # cryptic protocol error on the wire. Callers that genuinely
        # need legacy emission should construct
        # ``AssignRequest(node_id=42, role=None)`` and call
        # :meth:`encode_body_legacy` instead.
        raise EncodeError(
            "AssignRequest with role=None cannot be encoded via encode_body — "
            "modern dqlite servers and Go-dqlite always send both node_id and role. "
            "Use role=NodeRole.VOTER (or another role) for the modern ASSIGN body, "
            "or call encode_body_legacy() explicitly for the legacy PROMOTE shape."
        )

    def encode_body_legacy(self) -> bytes:
        """Encode as legacy PROMOTE body (1 word: node_id only).

        The legacy wire shape is explicit-opt-in only; ``encode_body``
        rejects ``role=None`` so accidental omission can't silently
        downgrade.

        Unlike ``LeaderResponse.encode_body_legacy``, which raises
        ``EncodeError`` if a non-zero ``node_id`` would be lost in the
        legacy shape, this method silently drops ``role`` (the legacy
        PROMOTE body has no role field — the upstream-semantic VOTER
        elevation is implicit). Callers asking for the legacy shape have
        already opted into that information loss; the class-level
        docstring's "Legacy round-trip asymmetry (intentional)" section
        is the canonical reference.
        """
        return encode_uint64(self.node_id)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "AssignRequest":
        if schema != 0:
            raise DecodeError(f"AssignRequest unsupported schema version {schema}")
        # Upstream emits bodies of exactly 8 (PROMOTE) or 16 (ASSIGN)
        # bytes. Reject anything else rather than silently dropping
        # trailing bytes — parity with the C cursor-cap semantics.
        if len(data) == 8:
            # Legacy PROMOTE: no role on the wire. Map to
            # NodeRole.VOTER per the documented upstream semantics
            # (PROMOTE elevates a non-voter to voter). Without this,
            # the dataclass would carry role=None which round-trips
            # to encode_body and raises EncodeError — the legacy
            # shape would no longer round-trip.
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
    """Remove a node from the cluster.

    Body: uint64 node_id
    """

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
    """Request a database dump.

    Body: text name
    """

    MSG_TYPE: ClassVar[int] = RequestType.DUMP

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise EncodeError(f"DumpRequest.name must be str, got {type(self.name).__name__}")

    @override
    def encode_body(self) -> bytes:
        # The C gateway's ``handle_dump`` uses a 1024-byte stack
        # buffer for the WAL filename, leaving 1019 bytes once the
        # ``-wal`` suffix and NUL terminator are accounted for
        # (dqlite-upstream ``src/gateway.c::handle_dump``). Cap the
        # encoder at the C ceiling so a longer name fails fast on
        # the Python side rather than producing a ``FilesResponse``
        # whose WAL entry refers to a truncated path.
        return encode_text(self.name, max_size=_MAX_DUMP_FILENAME_SIZE, label="database name")

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "DumpRequest":
        if schema != 0:
            raise DecodeError(f"DumpRequest unsupported schema version {schema}")
        # Symmetric with ``encode_body`` — refuse a peer-supplied
        # ``DumpRequest`` whose name exceeds the C ceiling so the
        # decoder cannot accept a request the encoder would reject.
        name, consumed = decode_text(data, max_size=_MAX_DUMP_FILENAME_SIZE, label="database name")
        if consumed != len(data):
            raise DecodeError(f"DumpRequest has {len(data) - consumed} trailing bytes")
        return cls(name)


@final
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
    # Decoded-V0 sentinel. Declared as a regular dataclass field so it
    # participates in the dataclass-generated ``__init__`` (the
    # decoder passes ``_decoded=True`` as a constructor kwarg —
    # uniformly through dataclass machinery, no low-level instance
    # bypass). ``repr=False`` / ``compare=False`` keep it out of
    # equality and display. Mirrors the constructor-kwarg sentinel
    # pattern on the SQL request siblings (``_decoded_schema``,
    # ``_decoded_empty_header``) and on ``AssignRequest._legacy_intent``.
    _decoded: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_uint64("format", self.format)
        # Construction-time rejection still applies to V0 because
        # this client's outbound shape is V1-only (the matched
        # ``ServersResponse`` decoder reads the role fields). Upstream
        # defines only V0=0 and V1=1 (include/dqlite.h); the gateway
        # rejects anything else with DQLITE_PARSE.
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
        if self.format == 0:
            # Encode rejection mirrors the construction-time gate: this
            # client cannot consume the V0 ServersResponse, so emitting
            # a V0 request is a wire-shape inconsistency.
            raise EncodeError(
                "ClusterRequest format=0 (V0) is valid in upstream dqlite but "
                "not implemented in this Python library: ServersResponse only "
                "decodes V1. Use format=1."
            )
        return encode_uint64(self.format)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "ClusterRequest":
        if schema != 0:
            raise DecodeError(f"ClusterRequest unsupported schema version {schema}")
        # Decode-side accepts V0 so a relaying proxy / mock server /
        # captured-traffic replay tool can round-trip a V0 frame
        # without losing the original byte shape. Re-encoding a
        # decoded V0 still raises (encode_body refuses) — only the
        # decode → introspect → drop flow is supported.
        if len(data) != 8:
            raise DecodeError(f"ClusterRequest body must be 8 bytes, got {len(data)}")
        format_val = decode_uint64(data)
        if format_val not in (0, 1):
            raise DecodeError(
                f"ClusterRequest format must be 0 (V0) or 1 (V1); upstream "
                f"defines only those two values. Got {format_val}."
            )
        # Pass ``_decoded=True`` through the dataclass-generated
        # ``__init__`` so the V0 construction-time gate sees the
        # sentinel and short-circuits. Going through the public
        # constructor keeps the decoder uniform with the sibling SQL
        # request family (which uses the same pattern for
        # ``_decoded_schema`` / ``_decoded_empty_header``). The
        # previous implementation used a low-level instance-bypass
        # path which silently broke under frozen=True / slots=True
        # flips or new fields without explicit assignment.
        return cls(format=format_val, _decoded=True)


@final
@dataclass
class TransferRequest(Message):
    """Request leadership transfer.

    Body: uint64 target_node_id
    """

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
    """Request database schema description.

    Body: uint64 format

    Upstream defines only ``DQLITE_REQUEST_DESCRIBE_FORMAT_V0 = 0``
    (``gateway.c`` rejects anything else with ``SQLITE_PROTOCOL``).
    Reject unknown formats client-side so callers get a local
    ``EncodeError`` instead of a confusing server failure.
    """

    MSG_TYPE: ClassVar[int] = RequestType.DESCRIBE

    format: int = 0

    def __post_init__(self) -> None:
        _validate_uint64("format", self.format)
        if self.format != 0:
            raise EncodeError(
                f"DescribeRequest format must be 0 (V0); upstream rejects "
                f"anything else with SQLITE_PROTOCOL. Got {self.format}."
            )

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.format)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "DescribeRequest":
        if schema != 0:
            raise DecodeError(f"DescribeRequest unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"DescribeRequest body must be 8 bytes, got {len(data)}")
        format_val = decode_uint64(data)
        if format_val != 0:
            raise DecodeError(f"DescribeRequest format must be 0 (V0); got {format_val}")
        return cls(format_val)


@final
@dataclass
class WeightRequest(Message):
    """Set node weight for leader election.

    Body: uint64 weight
    """

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
