"""Server to client response messages."""

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

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
    _validate_uint32,
    _validate_uint64,
    decode_text,
    decode_uint32,
    decode_uint64,
    encode_text,
    encode_uint32,
    encode_uint64,
    encode_value,
)

# Defense-in-depth upper bounds for count fields in response messages.
# Tightened to SQLite's documented hard cap (SQLITE_MAX_COLUMN max =
# 32767 per https://www.sqlite.org/limits.html). The default upstream
# build uses 2000, but custom builds can go up to 32767 — anything
# above that is provably malformed and could only come from a hostile
# or corrupted peer.
_MAX_COLUMN_COUNT: Final[int] = 32767
_MAX_FILE_COUNT: Final[int] = 100
_MAX_NODE_COUNT: Final[int] = 10_000

# Upper bound on ``StmtResponse.tail_offset``. The field is a byte
# offset into the prepared-SQL text; a malicious peer could emit an
# enormous value and Python's slice semantics would silently return
# ``""`` on ``sql[offset:]``, dropping later statements with no
# diagnostic. 1 MiB is far above any realistic multi-statement SQL
# size.
_MAX_TAIL_OFFSET: Final[int] = 1 * 1024 * 1024

# Per-field cap on ``FailureResponse.message``. The frame-size cap
# in ``buffer.py`` (64 MiB) bounds total bytes, but error messages in
# practice are short (SQLite's own error strings are under ~200 chars).
# A peer sending megabytes of text is malicious or broken; cap well
# above any realistic message so legitimate cases are never clipped.
_MAX_FAILURE_MESSAGE_SIZE: Final[int] = 64 * 1024

# Per-column-name cap on ``RowsResponse``. SQLite column-name identifiers
# are short by any realistic standard; 4 KiB is orders of magnitude above
# legitimate use and well below any memory-exhaustion concern. Same
# defense-in-depth policy as ``_MAX_FAILURE_MESSAGE_SIZE``.
_MAX_COLUMN_NAME_SIZE: Final[int] = 4096

# Per-filename cap on ``FilesResponse``. dqlite file entries are the
# on-disk page-backed database files (``main``, ``wal``, etc.); POSIX
# PATH_MAX is 4 KiB and mirrors the column-name cap.
_MAX_FILENAME_SIZE: Final[int] = 4096

# Per-address cap on ``LeaderResponse`` / ``ServersResponse`` (and their
# legacy variants). Legitimate cluster addresses are small (hostname
# + port, or IPv6 literal in brackets + port); RFC 1035 sets domain
# names at ≤253 bytes and 256 leaves margin for the port. A multi-MB
# "address" is malicious or broken and would amplify through log /
# exception messages even after ``_sanitize_server_text``.
_MAX_ADDRESS_SIZE: Final[int] = 256

# Sanitize server-supplied text destined for exception messages and
# logs. The C server promises UTF-8 but makes no promise about terminal
# escapes or log-injection characters: a malicious or compromised peer
# can embed ANSI colour/clear sequences, CR/LF to forge log lines, or
# NUL bytes that upset some log backends. Replace the following with a
# literal "?":
#
# - C0 controls (\x00-\x08, \x0b-\x1f) and DEL (\x7f)
# - C1 controls (\x80-\x9f) — some terminals still interpret these
#   as CSI / DCS / OSC sequences
# - Unicode line / paragraph separators (U+2028, U+2029) — Python
#   logging and journald treat these as line breaks, so a server
#   message with U+2028 can inject arbitrary log lines
# - Unicode bidi formatting controls (U+202A-U+202E, U+2066-U+2069,
#   U+061C) — the Trojan-Source primitives
# - Zero-width / invisible characters (U+200B-U+200F, U+FEFF) that
#   hide content from casual visual inspection while surviving
#   substring matches
#
# Tab (0x09) and LF (0x0A) are left intact so legitimate multi-line
# server diagnostics render correctly. CR (0x0D) is dropped — it
# is the log-injection vector alongside LF.
_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f"
    r"؜"
    r"​-‏"
    r"  "
    r"‪-‮"
    r"⁦-⁩"
    r"﻿"
    r"]"
)


def _sanitize_server_text(s: str) -> str:
    """Replace control / bidi / invisible characters with '?' in server strings.

    Applied at the decoder boundary for text fields that are
    **display-only** (``FailureResponse.message``). Address fields
    (``LeaderResponse.address``, ``ServersResponse.nodes[*].address``)
    are decoded raw — those values flow into TCP routing and
    allowlist-policy comparisons, and mangling them at decode time
    would silently split an operator-configured address set. The
    client layer (``dqliteclient.cluster``) calls this helper at
    log / exception format time instead.

    Leaves tab and LF untouched so multi-line server diagnostics
    render correctly. See the regex above for the full replacement
    class.
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

    def __post_init__(self) -> None:
        # ``code`` is intentionally NOT rejected when 0. Upstream's gateway
        # emits ``failure(req, 0, "empty statement")`` from
        # ``handle_prepare_done_cb`` and ``handle_query_sql_done_cb`` when
        # the SQL parses to no statement (empty / comment-only / no bound
        # parameters). The C source even has a ``/* FIXME Should we use a
        # code other than 0 here? */`` at the emit site. Rejecting code=0
        # here would poison the streaming decoder buffer on a clean
        # operational signal and force reconnect for users issuing
        # ``cursor.execute("-- comment\n")`` against a real cluster.
        _validate_uint64("code", self.code)

    def encode_body(self) -> bytes:
        return encode_uint64(self.code) + encode_text(
            self.message, max_size=_MAX_FAILURE_MESSAGE_SIZE, label="Failure message"
        )

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "FailureResponse":
        # Require at least 9 bytes: 8 for code + 1 minimum for the
        # null-terminator of an empty message. Without that, an
        # exactly-8-byte body passes the size check and surfaces a
        # less-actionable "Failure message not null-terminated"
        # diagnostic from ``decode_text``. Wire-format alignment pads
        # bodies to multiples of 8, so the realistic minimum is 16
        # bytes (8 code + 1 NUL + 7 padding); the < 9 guard catches
        # the degenerate 8-byte case at the boundary.
        if len(data) < 9:
            raise DecodeError(
                f"FailureResponse body too short: need at least 9 bytes "
                f"(8 for code + 1 for null terminator), got {len(data)}"
            )
        code = decode_uint64(data[:8])
        message, consumed = decode_text(
            data[8:], max_size=_MAX_FAILURE_MESSAGE_SIZE, label="Failure message"
        )
        offset = 8 + consumed
        if offset != len(data):
            # Strict-decode parity with sibling decoders: conforming
            # Go/C servers never emit trailing padding on this body.
            raise DecodeError(
                f"FailureResponse has {len(data) - offset} trailing bytes after message"
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

    def __post_init__(self) -> None:
        _validate_uint64("node_id", self.node_id)

    def encode_body(self) -> bytes:
        return encode_uint64(self.node_id) + encode_text(
            self.address, max_size=_MAX_ADDRESS_SIZE, label="leader address"
        )

    def encode_body_legacy(self) -> bytes:
        """Encode as legacy (V0) body: text address only.

        Upstream emits this shape via ``SUCCESS_V0(server_legacy,
        SERVER_LEGACY)`` when the negotiated protocol version is
        ``PROTOCOL_VERSION_LEGACY``. Mirror of
        :meth:`decode_body_legacy`. The legacy format does NOT carry
        ``node_id`` and the decoder hard-codes 0 on the way back. To
        prevent silent information loss, this method rejects any
        non-zero ``node_id`` with ``EncodeError`` — callers with a
        meaningful node_id must use :meth:`encode_body` (modern
        format) on a non-legacy-version protocol negotiation.
        """
        if self.node_id != 0:
            raise EncodeError(
                "LeaderResponse legacy encoding cannot carry node_id; "
                f"got node_id={self.node_id}. Use encode_body() for the "
                "modern format."
            )
        address = self.address or ""
        return encode_text(address, max_size=_MAX_ADDRESS_SIZE, label="leader address")

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "LeaderResponse":
        """Decode leader response body (modern v1+ format).

        Modern format: uint64 node_id + text address.
        For pre-1.0 (legacy) servers that send only text address without
        node_id, use decode_body_legacy() instead.

        **Version selection is caller-locked.** ``MessageDecoder``
        chooses between this method and ``decode_body_legacy`` based
        on its constructor-time ``version`` argument; the decoder
        does NOT auto-detect the body shape from the bytes themselves.
        A misaligned encoder/decoder version pair (e.g. encoder set to
        ``PROTOCOL_VERSION_LEGACY`` so the server replies in legacy
        shape, but decoder constructed with ``PROTOCOL_VERSION``
        default) silently misdecodes the legacy body's leading 8
        bytes of address text as ``node_id`` and returns the address
        tail past the misaligned cursor — both fields are then
        garbage and the address routes downstream into a non-
        existent host. The only safe recovery for a mixed-version
        cluster is to reconstruct the ``MessageDecoder`` with the
        constructor ``version`` matching the per-connection
        negotiated value (the same value the encoder used).
        """
        node_id = decode_uint64(data)
        address, consumed = decode_text(
            data[8:], max_size=_MAX_ADDRESS_SIZE, label="leader address"
        )
        offset = 8 + consumed
        if offset != len(data):
            # Strict-decode parity with sibling decoders: conforming
            # Go/C servers never emit trailing padding on this body.
            raise DecodeError(
                f"LeaderResponse has {len(data) - offset} trailing bytes after address"
            )
        # Address is stored raw — it flows into TCP routing and
        # allowlist comparisons downstream. Sanitisation happens at
        # log / exception format time, not at decode.
        return cls(node_id, address)

    @classmethod
    def decode_body_legacy(cls, data: bytes) -> "LeaderResponse":
        """Decode legacy (pre-1.0) leader response body.

        Legacy format: text address only (no node_id). Returns node_id=0.
        Go reference: DecodeNodeLegacy in internal/protocol/message.go.
        """
        address, consumed = decode_text(data, max_size=_MAX_ADDRESS_SIZE, label="leader address")
        if consumed != len(data):
            raise DecodeError(
                f"LeaderResponse (legacy) has {len(data) - consumed} trailing bytes after address"
            )
        # Raw address — see the modern decoder for rationale.
        return cls(node_id=0, address=address)


@dataclass
class WelcomeResponse(Message):
    """Client registration acknowledgment.

    Body: uint64 heartbeat_timeout

    Attributes:
        heartbeat_timeout: Server-advertised heartbeat interval, in
            **milliseconds** (upstream default 15000 = 15 s, set from
            ``config->heartbeat_timeout`` in ``config.c`` and copied
            verbatim by ``gateway.c``'s WELCOME handler). The unit is a
            protocol-level invariant; callers passing this value to
            seconds-based APIs (``asyncio.wait_for``, ``time.sleep``)
            must divide by 1000 first. Prefer
            :attr:`heartbeat_timeout_seconds` to avoid the divisor.

            A value of 0 is accepted (the wire layer does not enforce
            a minimum) but is semantically ambiguous: upstream
            ``config.c`` defaults to 15000 and never emits 0, so a 0
            from the wire is either a misconfigured peer or a non-
            conforming server. The in-tree client treats it as "no
            heartbeat-driven timeout extension" — ``trust_server_heartbeat``
            falls back to the static ``_read_timeout`` floor.
    """

    MSG_TYPE: ClassVar[int] = ResponseType.WELCOME

    heartbeat_timeout: int

    def __post_init__(self) -> None:
        _validate_uint64("heartbeat_timeout", self.heartbeat_timeout)

    @property
    def heartbeat_timeout_seconds(self) -> float:
        """Heartbeat interval converted to seconds (milliseconds / 1000)."""
        return self.heartbeat_timeout / 1000.0

    def encode_body(self) -> bytes:
        return encode_uint64(self.heartbeat_timeout)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "WelcomeResponse":
        if len(data) != 8:
            raise DecodeError(f"WelcomeResponse body must be exactly 8 bytes, got {len(data)}")
        heartbeat_timeout = decode_uint64(data)
        return cls(heartbeat_timeout)


@dataclass
class DbResponse(Message):
    """Database opened response.

    Body: uint32 db_id, uint32 reserved
    """

    MSG_TYPE: ClassVar[int] = ResponseType.DB

    db_id: int

    def __post_init__(self) -> None:
        _validate_uint32("db_id", self.db_id)

    def encode_body(self) -> bytes:
        return encode_uint32(self.db_id) + encode_uint32(0)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "DbResponse":
        if len(data) != 8:
            raise DecodeError(f"DbResponse body must be exactly 8 bytes, got {len(data)}")
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
    # Optional schema-byte override for mock servers / proxies that
    # must echo the inbound ``PrepareRequest.schema`` byte
    # independently of whether ``tail_offset`` is set. Upstream C
    # dispatches reply shape on the REQUEST's schema byte
    # (gateway.c::handle_prepare_done_cb): schema=0 emits V0 (16 B,
    # no tail_offset), schema=1 emits V1 (24 B, with tail_offset).
    # Without this override, the auto-select cannot emit "V1 with
    # tail_offset=0" — a valid upstream reply shape. ``None`` keeps
    # the historical auto-select semantics intact.
    schema: int | None = None

    def __post_init__(self) -> None:
        # Range and bool-rejection validation, parity with the sibling
        # responses (FailureResponse / LeaderResponse / HeartbeatResponse
        # at lines ~155, ~213, ~312). Without this, ``StmtResponse(...)``
        # silently accepts negative ints, ints exceeding the wire field
        # width, and bool-as-int (``True`` → 1) — every other Response
        # rejects these.
        _validate_uint32("db_id", self.db_id)
        _validate_uint32("stmt_id", self.stmt_id)
        _validate_uint64("num_params", self.num_params)
        if self.tail_offset is not None:
            _validate_uint64("tail_offset", self.tail_offset)
        if self.schema is not None and self.schema not in (0, 1):
            raise ValueError(f"StmtResponse.schema must be 0 or 1, got {self.schema}")
        if self.schema == 0 and self.tail_offset is not None:
            raise ValueError(
                "StmtResponse: schema=0 (V0 body) cannot carry tail_offset; pass schema=1 for V1"
            )
        # Normalise the V1-implicit-zero form so encode/decode round-
        # trip preserves dataclass equality. ``StmtResponse(schema=1,
        # tail_offset=None)`` and ``StmtResponse(schema=1,
        # tail_offset=0)`` produce identical wire bytes; without this
        # the decoder reconstructs as the latter and the dataclass
        # comparison fails. Construction-time normalisation matches
        # ``AssignRequest.__post_init__``'s int→NodeRole coercion
        # for the same reason.
        if self.schema == 1 and self.tail_offset is None:
            self.tail_offset = 0

    def _get_schema(self) -> int:
        if self.schema is not None:
            return self.schema
        return 1 if self.tail_offset is not None else 0

    def encode_body(self) -> bytes:
        if self.num_params > _MAX_PARAM_COUNT:
            raise EncodeError(
                f"StmtResponse num_params {self.num_params} exceeds maximum ({_MAX_PARAM_COUNT})"
            )
        result = (
            encode_uint32(self.db_id) + encode_uint32(self.stmt_id) + encode_uint64(self.num_params)
        )
        if self._get_schema() == 1:
            # V1: always emit tail_offset (default 0 when None so an
            # explicit schema=1 without tail_offset still produces a
            # 24-byte body matching the inbound PrepareRequest.schema).
            tail_offset = self.tail_offset or 0
            if tail_offset > _MAX_TAIL_OFFSET:
                raise EncodeError(
                    f"StmtResponse tail_offset {tail_offset} exceeds maximum ({_MAX_TAIL_OFFSET})"
                )
            result += encode_uint64(tail_offset)
        return result

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "StmtResponse":
        expected = 24 if schema >= 1 else 16
        if len(data) != expected:
            raise DecodeError(
                f"StmtResponse schema={schema} body must be exactly {expected} bytes, "
                f"got {len(data)}"
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
        # Defense-in-depth cap: ``tail_offset`` is the byte offset of
        # the unparsed SQL tail; a hostile peer could emit ``2**63`` and
        # Python's slice semantics would silently return "" on
        # ``sql[offset:]``, dropping later statements without any
        # diagnostic. Cap against the outbound SQL size cap so the
        # ceiling for inbound and outbound stays aligned.
        if tail_offset is not None and tail_offset > _MAX_TAIL_OFFSET:
            raise DecodeError(
                f"StmtResponse tail_offset {tail_offset} exceeds maximum ({_MAX_TAIL_OFFSET})"
            )
        # Preserve the incoming header schema byte on the dataclass so
        # round-trip encode emits exactly what came in. Without this,
        # a V1 body with ``tail_offset=0`` would round-trip through
        # ``_get_schema()`` as V0 (``tail_offset is None``) on any
        # future equality-based manipulation, losing the
        # "peer advertised V1" signal.
        return cls(db_id, stmt_id, num_params, tail_offset, schema=schema)


@dataclass
class ResultResponse(Message):
    """Statement execution result.

    Body: uint64 last_insert_id, uint64 rows_affected

    Note on signedness: the wire is uint64 per the dqlite protocol spec.
    The upstream C server casts ``sqlite3_last_insert_rowid()``
    (``sqlite3_int64``) through ``(uint64_t)`` before sending, so a
    negative SQLite rowid arrives here as ``2**64 - abs(rowid)``.
    Downstream layers that surface this to PEP 249's
    ``cursor.lastrowid`` (which mirrors ``sqlite3.Connection.lastrowid``,
    a signed value) may want to re-cast as int64 for stdlib parity.
    """

    MSG_TYPE: ClassVar[int] = ResponseType.RESULT

    last_insert_id: int
    rows_affected: int

    def __post_init__(self) -> None:
        _validate_uint64("last_insert_id", self.last_insert_id)
        _validate_uint64("rows_affected", self.rows_affected)

    def encode_body(self) -> bytes:
        return encode_uint64(self.last_insert_id) + encode_uint64(self.rows_affected)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "ResultResponse":
        if len(data) != 16:
            raise DecodeError(f"ResultResponse body must be exactly 16 bytes, got {len(data)}")
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
        column_types: Types from the first decoded row's per-column type
            tag. SQLite uses dynamic typing, so different rows may have
            different types for the same column — use ``row_types`` for
            accurate per-row type information. **Empty when the frame
            carries no rows**; prefer ``column_names`` to detect the
            column count in that case.

            Caveat: when row 0 contains NULL in column X, the per-row
            type tag for that cell is encoded as NULL on the wire
            (Go-parity, see ``encode_body``), so on decode
            ``column_types[X]`` will be ``ValueType.NULL`` even if the
            encoder was given an explicit non-NULL ``column_types``.
            Consumers needing the schema-declared type should not rely
            on ``column_types[X]`` when row 0 may carry NULL — use a
            later non-NULL row's ``row_types`` entry, or consult the
            schema separately.
        row_types: Per-row type lists, one entry per decoded row.
        rows: Decoded row values.
        has_more: True if a PART marker was found (more rows in next message).

    Encode-side type inference: ``encode_body`` consults ``row_types``
    first, then ``column_types``; if **both are empty** and ``rows`` are
    supplied, per-cell types are inferred from Python types via
    ``encode_value``. Inference cannot distinguish the dqlite-specific
    encodings (UNIXTIME / ISO8601 / BOOLEAN / SERVER_TIME) from the
    primitives they share a byte layout with (INTEGER / TEXT / INTEGER
    / INTEGER respectively): a Python ``int`` always encodes as
    ``ValueType.INTEGER``. Callers building frames for a column with a
    declared dqlite-specific type (mock servers, golden-byte harnesses)
    MUST pass ``column_types`` or ``row_types`` explicitly — otherwise
    re-encoding a round-tripped frame emits the wrong type nibble even
    though the payload bytes are identical.
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
        if col_count > _MAX_COLUMN_COUNT:
            raise EncodeError(
                f"RowsResponse column count {col_count} exceeds maximum ({_MAX_COLUMN_COUNT})"
            )
        # Per-name byte cap is enforced inside encode_text below;
        # keeping the loop here would re-check against codepoints
        # rather than UTF-8 bytes (the unit the decoder caps on),
        # admitting non-ASCII names near the boundary that the
        # decoder rejects.
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
            result += encode_text(name, max_size=_MAX_COLUMN_NAME_SIZE, label="Column name")

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

        # Bounds check: each column name is at least 8 bytes (null +
        # padding) AND the mandatory 8-byte DONE/PART row-end marker
        # follows the names (or the row stream). Reserve 8 bytes for
        # the marker so a ``column_count`` that would consume every
        # remaining byte fails here with a clear diagnostic rather than
        # later via "body exhausted without end marker".
        remaining = len(view) - offset
        max_columns = max(0, remaining - WORD_SIZE) // 8
        if column_count > max_columns:
            raise DecodeError(
                f"Column count {column_count} exceeds maximum possible in "
                f"{remaining} bytes of remaining data (reserving {WORD_SIZE} "
                f"bytes for the row end marker)"
            )

        # Column names
        column_names: list[str] = []
        for _ in range(column_count):
            name, consumed = decode_text(
                view[offset:], max_size=_MAX_COLUMN_NAME_SIZE, label="column name"
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
            # The zero-column fast path is Python-specific (upstream C never
            # emits zero-column result sets); enforce buffer exhaustion to
            # match the strict-decode pattern used by every sibling decoder.
            end = offset + WORD_SIZE
            if end != len(view):
                raise DecodeError(
                    f"RowsResponse zero-column body has {len(view) - end} "
                    "trailing bytes after DONE/PART marker"
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
        if len(data) != 8:
            raise DecodeError(f"EmptyResponse body must be exactly 8 bytes, got {len(data)}")
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
        if len(self.files) > _MAX_FILE_COUNT:
            raise EncodeError(
                f"FilesResponse count {len(self.files)} exceeds maximum ({_MAX_FILE_COUNT})"
            )
        # Per-filename byte cap is enforced inside encode_text below.
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
            result += encode_text(name, max_size=_MAX_FILENAME_SIZE, label="filename")
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
            name, consumed = decode_text(
                view[offset:], max_size=_MAX_FILENAME_SIZE, label="filename"
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
            # Reject duplicate filenames: the wire format is a positional
            # sequence of N records; silently overwriting via dict would
            # make ``len(files) < count`` after decode and break
            # re-encode symmetry. Upstream's ``handle_dump`` only ever
            # emits distinct names (``main`` and ``main-wal``), so this
            # catches only malicious or misframed peers.
            if name in files:
                raise DecodeError(f"FilesResponse: duplicate filename {name!r}")
            files[name] = content
        # Upstream client enforces `cursor.cap == fs[i].size` at each
        # iteration; on the last file that amounts to "body must be
        # exhausted." Mirror the strictness so corrupt / malicious
        # trailing bytes cannot vanish silently.
        if offset != len(view):
            raise DecodeError(
                f"FilesResponse has {len(view) - offset} trailing bytes after last file"
            )
        return cls(files)


@dataclass(frozen=True, slots=True)
class NodeInfo:
    """Information about a cluster node.

    Frozen + slotted to match ``dqliteclient.node_store.NodeInfo``. The
    class holds wire-decoded values that are handed off for routing
    decisions; mutation would invalidate caller-held references, and
    hashability lets instances live in sets / dict keys.
    """

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
        if len(self.nodes) > _MAX_NODE_COUNT:
            raise EncodeError(
                f"ServersResponse node count {len(self.nodes)} exceeds maximum ({_MAX_NODE_COUNT})"
            )
        result = encode_uint64(len(self.nodes))
        for node in self.nodes:
            result += encode_uint64(node.node_id)
            result += encode_text(node.address, max_size=_MAX_ADDRESS_SIZE, label="server address")
            result += encode_uint64(node.role)
        return result

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "ServersResponse":
        """Decode the modern V1 cluster body shape (id + address + role).

        The decoder unconditionally parses the V1 (3-field-per-node)
        layout. The upstream C gateway's ``handle_cluster``
        (``src/gateway.c``) historically supported a V0 cluster
        request whose response carried only ``id + address`` (no
        role). Our outbound ``ClusterRequest.__post_init__``
        (``requests.py``) rejects ``format=0`` so an in-tree client
        cannot ask for the V0 shape — making the V0 response a
        contract no in-tree call site triggers.

        For hostile / malformed peer traffic that emits a V0 body
        anyway, the existing ``count > remaining // 24`` bounds check
        and the trailing-bytes reject below catch the misalignment in
        the common case (V0 nodes are 16 bytes; the V1 parser would
        read 24 and either fall short on the first node or accumulate
        a trailing-bytes mismatch). The narrow silent-misparse residual
        — V0 bytes whose layout happens to align with V1's
        ``id + addr + role`` and yield raw_role values in {0,1,2} — is
        not addressed here; format-aware decoding would ripple
        ``format=`` plumbing into the client layer for a shape no
        in-tree path emits.
        """
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
            address, consumed = decode_text(
                view[offset:], max_size=_MAX_ADDRESS_SIZE, label="server address"
            )
            # Raw address — sanitisation happens at log / exception
            # format time so the value used for TCP routing and
            # allowlist comparisons stays authentic. See
            # ``LeaderResponse.decode_body`` for rationale.
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
        if offset != len(view):
            # Strict-decode parity with sibling variable-length
            # decoders: conforming Go/C servers never emit trailing
            # padding on this response.
            raise DecodeError(
                f"ServersResponse has {len(view) - offset} trailing bytes after {count} nodes"
            )
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

    def __post_init__(self) -> None:
        _validate_uint64("failure_domain", self.failure_domain)
        _validate_uint64("weight", self.weight)

    def encode_body(self) -> bytes:
        return encode_uint64(self.failure_domain) + encode_uint64(self.weight)

    @classmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "MetadataResponse":
        if len(data) != 16:
            raise DecodeError(f"MetadataResponse body must be exactly 16 bytes, got {len(data)}")
        failure_domain = decode_uint64(data)
        weight = decode_uint64(data[8:])
        return cls(failure_domain, weight)
