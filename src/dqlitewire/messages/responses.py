"""Server to client response messages."""

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from dqlitewire.constants import (
    ROW_DONE_MARKER,
    ROW_PART_MARKER,
    WORD_SIZE,
    NodeRole,
    ResponseType,
    ValueType,
)
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.messages.base import Message
from dqlitewire.tuples import (
    _MAX_PARAM_COUNT,
    _ROW_DONE_MARKER,
    _ROW_PART_MARKER,
    RowMarker,
    decode_row_header,
    decode_row_values,
    encode_row_header,
    encode_row_values,
)
from dqlitewire.types import (
    decode_text,
    decode_uint32,
    decode_uint64,
    encode_text,
    encode_uint32,
    encode_uint64,
    encode_value,
)

# Defense-in-depth upper bounds for count fields in response messages.
# These are far above any legitimate use case but prevent CPU/memory
# exhaustion from malicious or corrupted messages.
_MAX_COLUMN_COUNT = 10_000
_MAX_FILE_COUNT = 100
_MAX_NODE_COUNT = 10_000

# Per-field cap on ``FailureResponse.message``. The frame-size cap
# in ``buffer.py`` (64 MiB) bounds total bytes, but error messages in
# practice are short (SQLite's own error strings are under ~200 chars).
# A peer sending megabytes of text is malicious or broken; cap well
# above any realistic message so legitimate cases are never clipped.
_MAX_FAILURE_MESSAGE_SIZE = 64 * 1024

# Per-column-name cap on ``RowsResponse``. SQLite column-name identifiers
# are short by any realistic standard; 4 KiB is orders of magnitude above
# legitimate use and well below any memory-exhaustion concern. Same
# defense-in-depth policy as ``_MAX_FAILURE_MESSAGE_SIZE``.
_MAX_COLUMN_NAME_SIZE = 4096

# Per-filename cap on ``FilesResponse``. dqlite file entries are the
# on-disk page-backed database files (``main``, ``wal``, etc.); POSIX
# PATH_MAX is 4 KiB and mirrors the column-name cap.
_MAX_FILENAME_SIZE = 4096

# Per-address cap on ``LeaderResponse`` / ``ServersResponse`` (and their
# legacy variants). Legitimate cluster addresses are small (hostname
# + port, or IPv6 literal in brackets + port); RFC 1035 sets domain
# names at ≤253 bytes and 256 leaves margin for the port. A multi-MB
# "address" is malicious or broken and would amplify through log /
# exception messages even after ``_sanitize_server_text``.
_MAX_ADDRESS_SIZE = 256

# Sanitize server-supplied text destined for exception messages and
# logs. The C server promises UTF-8 but makes no promise about terminal
# escapes or log-injection characters: a malicious or compromised peer
# can embed ANSI colour/clear sequences, CR/LF to forge log lines, or
# NUL bytes that upset some log backends. Replace C0 controls (except
# tab 0x09 and LF 0x0A) and DEL (0x7F) with a literal "?". CR (0x0D) is
# dropped — it is the log-injection vector alongside LF, and LF alone
# is enough to represent legitimate multi-line server messages in
# journald / file handlers.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _sanitize_server_text(s: str) -> str:
    """Replace C0 control characters and DEL with '?' in server strings.

    Applied at the decoder boundary for text fields that flow directly
    into exception messages and logs (FailureResponse.message,
    LeaderResponse.address, ServersResponse.nodes[*].address). Leaves
    tab and LF untouched so multi-line server diagnostics render
    correctly.
    """
    return _CONTROL_CHARS_RE.sub("?", s)


@dataclass
class FailureResponse(Message):
    """Operation failed.

    Body: uint64 code, text message

    The ``code`` field contains a SQLite error code (or extended error code).
    Common values include ``SQLITE_ERROR`` (1), ``SQLITE_BUSY`` (5), and the
    dqlite-specific extended codes ``SQLITE_IOERR_NOT_LEADER`` (the node is
    not the cluster leader) and ``SQLITE_IOERR_LEADERSHIP_LOST`` (leadership
    was lost during the operation). See the `SQLite result codes documentation
    <https://www.sqlite.org/rescode.html>`_ for the full list.
    """

    MSG_TYPE: ClassVar[int] = ResponseType.FAILURE

    code: int
    message: str

    def encode_body(self) -> bytes:
        if len(self.message) > _MAX_FAILURE_MESSAGE_SIZE:
            raise EncodeError(
                f"Failure message length {len(self.message)} "
                f"exceeds maximum {_MAX_FAILURE_MESSAGE_SIZE}"
            )
        return encode_uint64(self.code) + encode_text(self.message)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "FailureResponse":
        code = decode_uint64(data)
        message, _ = decode_text(data[8:])
        if len(message) > _MAX_FAILURE_MESSAGE_SIZE:
            raise DecodeError(
                f"Failure message length {len(message)} exceeds maximum {_MAX_FAILURE_MESSAGE_SIZE}"
            )
        return cls(code, _sanitize_server_text(message))


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
    def decode_body(cls, data: bytes, schema: int = 0) -> "LeaderResponse":
        """Decode leader response body (modern v1+ format).

        Modern format: uint64 node_id + text address.
        For pre-1.0 (legacy) servers that send only text address without
        node_id, use decode_body_legacy() instead.
        """
        node_id = decode_uint64(data)
        address, _ = decode_text(data[8:])
        if len(address) > _MAX_ADDRESS_SIZE:
            raise DecodeError(
                f"leader address length {len(address)} exceeds maximum {_MAX_ADDRESS_SIZE}"
            )
        return cls(node_id, _sanitize_server_text(address))

    @classmethod
    def decode_body_legacy(cls, data: bytes) -> "LeaderResponse":
        """Decode legacy (pre-1.0) leader response body.

        Legacy format: text address only (no node_id). Returns node_id=0.
        Go reference: DecodeNodeLegacy in internal/protocol/message.go.
        """
        address, _ = decode_text(data)
        if len(address) > _MAX_ADDRESS_SIZE:
            raise DecodeError(
                f"leader address length {len(address)} exceeds maximum {_MAX_ADDRESS_SIZE}"
            )
        return cls(node_id=0, address=_sanitize_server_text(address))


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
    def decode_body(cls, data: bytes, schema: int = 0) -> "WelcomeResponse":
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
        return encode_uint32(self.db_id) + encode_uint32(0)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "DbResponse":
        if len(data) < 8:
            raise DecodeError(f"DbResponse body must be 8 bytes, got {len(data)}")
        db_id = decode_uint32(data)
        reserved = decode_uint32(data[4:])
        if reserved != 0:
            raise DecodeError(f"DbResponse reserved field must be 0, got {reserved}")
        return cls(db_id)


@dataclass
class StmtResponse(Message):
    """Statement prepared response.

    V0 body: uint32 db_id, uint32 stmt_id, uint64 num_params
    V1 body: uint32 db_id, uint32 stmt_id, uint64 num_params, uint64 tail_offset

    Schema selection: ``_get_schema()`` derives the header schema byte from
    ``tail_offset``. ``tail_offset=None`` (default) → schema=0 (V0 body);
    ``tail_offset`` set to any int (including ``0``) → schema=1 (V1 body).
    Mock-server authors must match this to the schema byte of the inbound
    :class:`PrepareRequest`, since upstream dqlite servers dispatch on the
    request's schema byte, not on any reply-side field.

    Note: V1 tail_offset is not present in the canonical Go client
    (go-dqlite). The Go EncodePrepare always uses schema=0 and DecodeStmt
    does not read tail_offset. This feature may be supported by the C
    dqlite server for multi-statement SQL but is not exercised by Go.
    """

    MSG_TYPE: ClassVar[int] = ResponseType.STMT

    db_id: int
    stmt_id: int
    num_params: int
    tail_offset: int | None = None

    def _get_schema(self) -> int:
        return 1 if self.tail_offset is not None else 0

    def encode_body(self) -> bytes:
        result = (
            encode_uint32(self.db_id) + encode_uint32(self.stmt_id) + encode_uint64(self.num_params)
        )
        if self.tail_offset is not None:
            result += encode_uint64(self.tail_offset)
        return result

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "StmtResponse":
        expected = 24 if schema >= 1 else 16
        if len(data) < expected:
            raise DecodeError(
                f"StmtResponse schema={schema} requires at least {expected} bytes, got {len(data)}"
            )
        db_id = decode_uint32(data)
        stmt_id = decode_uint32(data[4:])
        num_params = decode_uint64(data[8:])
        # Parity with the encoder cap in tuples.py: a server declaring a
        # prepared-statement parameter count above _MAX_PARAM_COUNT is
        # either malicious or corrupt. Other count-bearing decode paths
        # (_MAX_COLUMN_COUNT, _MAX_FILE_COUNT, _MAX_NODE_COUNT) already
        # enforce their own caps; this closes the matching gap.
        if num_params > _MAX_PARAM_COUNT:
            raise DecodeError(
                f"StmtResponse num_params {num_params} exceeds maximum ({_MAX_PARAM_COUNT})"
            )
        tail_offset = decode_uint64(data[16:]) if schema >= 1 else None
        return cls(db_id, stmt_id, num_params, tail_offset)


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
    def decode_body(cls, data: bytes, schema: int = 0) -> "ResultResponse":
        last_insert_id = decode_uint64(data)
        rows_affected = decode_uint64(data[8:])
        return cls(last_insert_id, rows_affected)


@dataclass
class RowsResponse(Message):
    """Query result rows.

    Body: uint64 column_count, text[] column_names, then rows...
    Each row: header (types) + values, ending with marker

    Attributes:
        column_names: Column names from the query.
        column_types: Types from the first row's header. SQLite uses dynamic
            typing, so different rows may have different types for the same
            column. Use ``row_types`` for accurate per-row type information.
        row_types: Per-row type lists, one entry per decoded row.
        rows: Decoded row values.
        has_more: True if a PART marker was found (more rows in next message).
    """

    MSG_TYPE: ClassVar[int] = ResponseType.ROWS

    column_names: list[str] = field(default_factory=list)
    column_types: list[ValueType] = field(default_factory=list)
    row_types: list[list[ValueType]] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    has_more: bool = False

    def __post_init__(self) -> None:
        # Defensive copies. Two sources of
        # aliasing motivate this:
        #
        # 1. ``decode_body`` stores ``column_types = types`` where
        #    ``types`` is also stored as ``all_row_types[0]``, so
        #    without a copy ``self.column_types is self.row_types[0]``
        #    and mutating one silently rewrites the other.
        #
        # 2. User code constructing ``RowsResponse`` directly with a
        #    list they intend to keep mutating elsewhere.
        #
        # Copy all list-valued fields uniformly. ``row_types`` is a
        # list-of-lists so it needs both outer and inner copies; the
        # same for ``rows``. Cost is O(n) on the row dimension —
        # dominated by the row payload itself, so negligible.
        self.column_names = list(self.column_names)
        self.column_types = list(self.column_types)
        self.row_types = [list(t) for t in self.row_types]
        self.rows = [list(r) for r in self.rows]

    def _get_row_types(self, row_idx: int, row: list[Any]) -> list[ValueType]:
        """Get types for a row: from row_types, column_types, or inferred.

        The ``column_types`` fallback returns a fresh copy rather than
        ``self.column_types`` itself, so that a caller who mutates the
        return value cannot silently rewrite the message's private
        copy. This preserves the aliasing invariant that
        ``__post_init__`` establishes.

        None values override the declared type to NULL, matching Go's
        per-row type header behavior where the nibble reflects the actual
        value, not the column schema.
        """
        if self.row_types and row_idx < len(self.row_types):
            types = list(self.row_types[row_idx])
        elif self.column_types:
            types = list(self.column_types)
        else:
            # Infer from values
            return [encode_value(v)[1] for v in row]

        # Override type to NULL for None values, matching Go's behavior
        for i, v in enumerate(row):
            if v is None and i < len(types):
                types[i] = ValueType.NULL
        return types

    def encode_body(self) -> bytes:
        col_count = len(self.column_names)
        if self.column_types and len(self.column_types) != col_count:
            raise EncodeError(
                f"column_types length ({len(self.column_types)}) != "
                f"column_names length ({col_count})"
            )
        # ``row_types`` must either be empty (infer per-row from values /
        # column_types) or exactly match ``rows`` one-to-one. A shorter
        # list previously fell through to inference silently for the
        # trailing rows, contradicting the documented invariant.
        if self.row_types and len(self.row_types) != len(self.rows):
            raise EncodeError(
                f"row_types length ({len(self.row_types)}) != "
                f"rows length ({len(self.rows)}); pass an empty row_types "
                f"to infer per-row types from the values"
            )
        # Zero-column rows produce zero bytes per row, so the encoded
        # output is indistinguishable from a zero-row result set — the
        # decoder's zero-column fast path returns no rows. Reject at
        # encode time rather than silently lose row count.
        if col_count == 0 and self.rows:
            raise EncodeError(
                f"RowsResponse with zero columns cannot carry rows "
                f"(got {len(self.rows)} empty row(s))"
            )
        for i, row in enumerate(self.rows):
            if len(row) != col_count:
                raise EncodeError(f"Row {i} has {len(row)} values, expected {col_count}")
            if self.row_types and i < len(self.row_types) and len(self.row_types[i]) != col_count:
                raise EncodeError(
                    f"row_types[{i}] has {len(self.row_types[i])} types, expected {col_count}"
                )

        result = encode_uint64(col_count)

        # Column names
        for name in self.column_names:
            result += encode_text(name)

        # Rows - each row gets its own type header
        for i, row in enumerate(self.rows):
            types = self._get_row_types(i, row)
            result += encode_row_header(types)
            result += encode_row_values(row, types)

        # End marker: full uint64 marker word (matching Go)
        marker = ROW_PART_MARKER if self.has_more else ROW_DONE_MARKER
        result += encode_uint64(marker)

        return result

    DEFAULT_MAX_ROWS = 1_000_000

    @classmethod
    def decode_body(
        cls, data: bytes, schema: int = 0, max_rows: int = DEFAULT_MAX_ROWS
    ) -> "RowsResponse":
        # Wrap in memoryview so per-iteration slices are O(1) rather
        # than O(remaining). Without this, a body with many small rows
        # triggers quadratic-time decode: each
        # ``data[offset:]`` allocates a fresh ``bytes`` copy of the
        # tail. Memoryview slicing is a view, so slicing is free.
        view = memoryview(data)
        offset = 0

        # Column count
        column_count = decode_uint64(view[offset:])
        offset += 8

        if column_count > _MAX_COLUMN_COUNT:
            raise DecodeError(f"Column count {column_count} exceeds maximum {_MAX_COLUMN_COUNT}")

        # Bounds check: each column name is at least 8 bytes (null + padding)
        remaining = len(view) - offset
        if column_count > remaining // 8:
            raise DecodeError(
                f"Column count {column_count} exceeds maximum possible in "
                f"{remaining} bytes of remaining data"
            )

        # Column names
        column_names: list[str] = []
        for _ in range(column_count):
            name, consumed = decode_text(view[offset:])
            if len(name) > _MAX_COLUMN_NAME_SIZE:
                raise DecodeError(
                    f"column name length {len(name)} exceeds maximum {_MAX_COLUMN_NAME_SIZE}"
                )
            column_names.append(name)
            offset += consumed

        # Rows - each row has its own type header
        rows: list[list[Any]] = []
        all_row_types: list[list[ValueType]] = []
        column_types: list[ValueType] = []

        # Zero-column results cannot have row data (each row would be zero
        # bytes), so skip the row loop and consume the end marker directly.
        # Validate the full 8-byte sentinel against DQLITE_RESPONSE_ROWS_DONE
        # / _PART, matching the non-zero path (which goes through
        # decode_row_header). A first-byte-only compare would silently accept
        # torn markers like ``0xff 0x00..``.
        if column_count == 0:
            if offset + WORD_SIZE > len(view):
                raise DecodeError(
                    "RowsResponse body exhausted without end marker (zero-column result)"
                )
            marker = bytes(view[offset : offset + WORD_SIZE])
            if marker == _ROW_DONE_MARKER:
                has_more = False
            elif marker == _ROW_PART_MARKER:
                has_more = True
            else:
                raise DecodeError(
                    f"Expected DONE or PART marker for zero-column result, got 0x{marker.hex()}"
                )
            return cls(
                column_names=[],
                column_types=[],
                row_types=[],
                rows=[],
                has_more=has_more,
            )

        while offset < len(view):
            # Read row header; markers are detected byte-by-byte inside
            result, consumed = decode_row_header(view[offset:], column_count)
            offset += consumed

            if result is RowMarker.DONE:
                return cls(
                    column_names,
                    column_types=column_types,
                    row_types=all_row_types,
                    rows=rows,
                    has_more=False,
                )
            if result is RowMarker.PART:
                return cls(
                    column_names,
                    column_types=column_types,
                    row_types=all_row_types,
                    rows=rows,
                    has_more=True,
                )

            types = result
            if not isinstance(types, list):
                raise DecodeError(f"Expected column types list, got {type(types).__name__}")
            all_row_types.append(types)
            if not column_types:
                column_types = types

            # Read row values
            values, consumed = decode_row_values(view[offset:], types)
            rows.append(values)
            offset += consumed

            if len(rows) >= max_rows:
                raise DecodeError(f"Row count {len(rows)} reached maximum {max_rows}")

        raise DecodeError(
            f"RowsResponse body exhausted without end marker "
            f"(decoded {len(rows)} rows, consumed {offset} of {len(view)} bytes)"
        )


@dataclass
class EmptyResponse(Message):
    """Empty response (for exec with no result).

    Body: uint64 (reserved, unused)
    """

    MSG_TYPE: ClassVar[int] = ResponseType.EMPTY

    def encode_body(self) -> bytes:
        return encode_uint64(0)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "EmptyResponse":
        if len(data) < 8:
            raise DecodeError(f"EmptyResponse body must be 8 bytes, got {len(data)}")
        reserved = decode_uint64(data)
        if reserved != 0:
            raise DecodeError(f"EmptyResponse reserved field must be 0, got {reserved}")
        return cls()


@dataclass
class FilesResponse(Message):
    """Database dump files response.

    Body: uint64 count, then repeated (text filename, uint64 size, raw bytes content)

    Note: neither Go nor this implementation pads file content to word
    boundaries. The C server asserts content is always word-aligned
    (SQLite pages are multiples of 512), so padding is never needed
    in practice.
    """

    MSG_TYPE: ClassVar[int] = ResponseType.FILES

    files: dict[str, bytes] = field(default_factory=dict)

    def encode_body(self) -> bytes:
        result = encode_uint64(len(self.files))
        for name, content in self.files.items():
            # The upstream C server (gateway.c::dumpFile) asserts
            # ``len % 8 == 0`` for every file's content, because per-file
            # entries are written back-to-back with no explicit padding
            # and SQLite pages are always 8-byte aligned multiples of
            # 512. Validate here so a Python-encoded mock-server frame
            # cannot diverge from what a real C peer produces.
            if len(content) % 8 != 0:
                raise EncodeError(
                    f"FilesResponse content for {name!r} must be 8-byte aligned "
                    f"(got {len(content)} bytes); dqlite file entries carry no "
                    "per-file padding"
                )
            result += encode_text(name)
            result += encode_uint64(len(content))
            result += content
        return result

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "FilesResponse":
        # Memoryview for O(1) slicing in the per-file loop.
        view = memoryview(data)
        files: dict[str, bytes] = {}
        offset = 0
        count = decode_uint64(view[offset:])
        offset += 8
        if count > _MAX_FILE_COUNT:
            raise DecodeError(f"File count {count} exceeds maximum {_MAX_FILE_COUNT}")
        # Bounds check: each file is at least 16 bytes (name + size)
        remaining = len(view) - offset
        if count > remaining // 16:
            raise DecodeError(
                f"File count {count} exceeds maximum possible in "
                f"{remaining} bytes of remaining data"
            )
        for _ in range(count):
            name, consumed = decode_text(view[offset:])
            if len(name) > _MAX_FILENAME_SIZE:
                raise DecodeError(
                    f"filename length {len(name)} exceeds maximum {_MAX_FILENAME_SIZE}"
                )
            offset += consumed
            size = decode_uint64(view[offset:])
            offset += 8
            # Mirror of the encode-side invariant: upstream
            # gateway.c::dumpFile asserts ``len % 8 == 0`` for every
            # file's content. Reject non-aligned payloads on decode too
            # so a mock / malicious peer cannot produce bytes the real
            # C server would not emit.
            if size % 8 != 0:
                raise DecodeError(
                    f"FilesResponse content for {name!r} must be 8-byte aligned (got {size} bytes)"
                )
            if offset + size > len(view):
                raise DecodeError(
                    f"FilesResponse file content truncated: expected {size} bytes "
                    f"at offset {offset}, but only {len(view) - offset} bytes available"
                )
            content = bytes(view[offset : offset + size])
            # No padding after content — matches Go's byte-by-byte read.
            offset += size
            files[name] = content
        return cls(files)


@dataclass
class NodeInfo:
    """Information about a cluster node."""

    node_id: int
    address: str
    role: NodeRole


@dataclass
class ServersResponse(Message):
    """Cluster servers response.

    Body: uint64 count, then repeated (uint64 node_id, text address, uint64 role)
    """

    MSG_TYPE: ClassVar[int] = ResponseType.SERVERS

    nodes: list[NodeInfo] = field(default_factory=list)

    def encode_body(self) -> bytes:
        result = encode_uint64(len(self.nodes))
        for node in self.nodes:
            result += encode_uint64(node.node_id)
            result += encode_text(node.address)
            result += encode_uint64(node.role)
        return result

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "ServersResponse":
        # Memoryview for O(1) slicing in the per-node loop.
        view = memoryview(data)
        nodes: list[NodeInfo] = []
        offset = 0
        count = decode_uint64(view[offset:])
        offset += 8
        if count > _MAX_NODE_COUNT:
            raise DecodeError(f"Node count {count} exceeds maximum {_MAX_NODE_COUNT}")
        # Bounds check: each node is at least 24 bytes (id + address + role)
        remaining = len(view) - offset
        if count > remaining // 24:
            raise DecodeError(
                f"Node count {count} exceeds maximum possible in "
                f"{remaining} bytes of remaining data"
            )
        for _ in range(count):
            node_id = decode_uint64(view[offset:])
            offset += 8
            address, consumed = decode_text(view[offset:])
            if len(address) > _MAX_ADDRESS_SIZE:
                raise DecodeError(
                    f"server address length {len(address)} exceeds maximum {_MAX_ADDRESS_SIZE}"
                )
            address = _sanitize_server_text(address)
            offset += consumed
            raw_role = decode_uint64(view[offset:])
            offset += 8
            try:
                role = NodeRole(raw_role)
            except ValueError as exc:
                valid = sorted(r.value for r in NodeRole)
                raise DecodeError(
                    f"Invalid node role {raw_role} at offset {offset - 8}; expected one of {valid}"
                ) from exc
            nodes.append(NodeInfo(node_id, address, role))
        return cls(nodes)


@dataclass
class MetadataResponse(Message):
    """Node metadata response (failure domain and weight).

    Returned in response to a DescribeRequest. Contains the node's
    failure domain and weight, used for cluster topology decisions.

    Body: uint64 failure_domain, uint64 weight
    """

    MSG_TYPE: ClassVar[int] = ResponseType.METADATA

    failure_domain: int
    weight: int

    def encode_body(self) -> bytes:
        return encode_uint64(self.failure_domain) + encode_uint64(self.weight)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "MetadataResponse":
        failure_domain = decode_uint64(data)
        weight = decode_uint64(data[8:])
        return cls(failure_domain, weight)
