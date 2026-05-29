"""Server to client response messages."""

import logging
import re
import struct
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, final, override

__all__ = [
    "DbResponse",
    "EmptyResponse",
    "FailureResponse",
    "FilesResponse",
    "LeaderResponse",
    "MetadataResponse",
    "NodeInfo",
    "ResultResponse",
    "RowsResponse",
    "ServersResponse",
    "StmtResponse",
    "WelcomeResponse",
    "sanitize_for_log",
    "sanitize_server_text",
]

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
    WireValue,
    _infer_value_type,
    _validate_uint32,
    _validate_uint64,
    decode_text,
    decode_uint32,
    decode_uint64,
    encode_text,
    encode_uint32,
    encode_uint64,
)

logger = logging.getLogger(__name__)

# Defence-in-depth cap; the wire field is an uncapped uint64. Matches
# SQLite's documented SQLITE_MAX_COLUMN default so wide analytics SELECTs
# still decode. https://www.sqlite.org/limits.html#max_column
_MAX_COLUMN_COUNT: Final[int] = 2000
_MAX_FILE_COUNT: Final[int] = 100
# Defence-in-depth cap; the wire field is an uncapped uint64. Raft latency
# precludes realistic clusters far above ~100 nodes.
_MAX_NODE_COUNT: Final[int] = 10_000

# Cap on ``StmtResponse.tail_offset`` (a byte offset into prepared SQL):
# an enormous value would make ``sql[offset:]`` silently return "",
# dropping later statements. Matches the message envelope, not the
# text payload cap, so it stays a structural ceiling above the
# load-bearing _MAX_TEXT_VALUE_SIZE reject.
_MAX_TAIL_OFFSET: Final[int] = 64 * 1024 * 1024

# Per-field caps; the 64 MiB frame envelope bounds total bytes, but each
# variable-length field gets a tighter cap so one hostile entry can't
# claim the whole budget. Failure messages and identifiers are short in
# practice.
_MAX_FAILURE_MESSAGE_SIZE: Final[int] = 64 * 1024
_MAX_COLUMN_NAME_SIZE: Final[int] = 4096
_MAX_FILENAME_SIZE: Final[int] = 4096

# Cap on ``DumpRequest.name``: the C gateway's ``handle_dump`` uses a
# 1024-byte stack buffer and reserves room for the ``-wal`` suffix + NUL
# (1024 - 4 - 1). A longer name silently truncates the WAL filename C-side,
# yielding a torn dump whose ``-wal`` entry points at a different path.
_MAX_DUMP_FILENAME_SIZE: Final[int] = 1019

# Aligned with ``_MAX_BLOB_SIZE`` (64 MiB minus framing). Dumps above this
# require raising both ``max_message_size`` and this constant.
_MAX_FILE_CONTENT_SIZE: Final[int] = 64 * 1024 * 1024 - 64

# RFC 1035 caps domain names at 253 bytes; 256 leaves room for the port.
# A multi-MB "address" is hostile and amplifies through log/exception text.
_MAX_ADDRESS_SIZE: Final[int] = 256

# Default ``RowsResponse`` row cap; mirrors ``MessageDecoder``'s max_rows
# default. Module-level so both the ClassVar alias and decode_body share it.
_DEFAULT_MAX_ROWS: Final[int] = 1_000_000

# Replace control / bidi / invisible characters with "?" in server-supplied
# text. The server promises UTF-8 but not absence of terminal escapes or
# log-injection chars. Tab and LF are kept so multi-line diagnostics render;
# CR is dropped (log-injection vector). Codepoints spelled as \uXXXX so the
# source survives a wrong-encoding file load and stays greppable.
_CONTROL_CHARS_RE: Final[re.Pattern[str]] = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f"
    r"\u061c"  # Arabic letter mark
    r"\u1680"  # Ogham space mark (whitespace-masking defense)
    r"\u180e"  # Mongolian vowel separator (historically zero-width)
    r"\u200b-\u200f"  # zero-width space, ZWNJ, ZWJ, LRM, RLM
    r"\u2028\u2029"  # line / paragraph separator
    r"\u202a-\u202e"  # bidi override (LRE/RLE/PDF/LRO/RLO)
    r"\u202f"  # narrow no-break space (invisible whitespace)
    r"\u2060"  # word joiner (same family as ZWSP)
    r"\u2066-\u2069"  # bidi isolate (LRI/RLI/FSI/PDI)
    r"\ufeff"  # BOM / zero-width no-break space
    r"]"
)


def sanitize_server_text(s: str) -> str:
    """Replace control / bidi / invisible characters with '?' in server strings.

    Applied only to display-only fields. Address fields are decoded raw
    (they flow into TCP routing / allowlist comparisons) and sanitised at
    log/exception format time instead. Leaves tab and LF intact; for log
    output use :func:`sanitize_for_log`, which also escapes LF.
    """
    return _CONTROL_CHARS_RE.sub("?", s)


# Backwards-compatible alias; kept until downstream packages migrate off it.
_sanitize_server_text = sanitize_server_text


def sanitize_for_log(s: str) -> str:
    """Like :func:`sanitize_server_text` but also escapes LF/tab as ``\\n``/``\\t``.

    LF and tab pass through ``sanitize_server_text`` intact; escaping them
    here stops a hostile field from injecting fake log lines or breaking
    TSV/field-separator-based log parsers. CR is already replaced upstream.
    """
    return sanitize_server_text(s).replace("\n", "\\n").replace("\t", "\\t")


@final
@dataclass(frozen=True, slots=True)
class FailureResponse(Message):
    """Operation failed.

    Body: uint64 code (a SQLite or dqlite extended error code), text message.
    """

    MSG_TYPE: ClassVar[int] = ResponseType.FAILURE

    code: int
    message: str

    def __post_init__(self) -> None:
        # code=0 is intentionally accepted: upstream emits failure(req, 0,
        # "empty statement") for comment-only/empty SQL. Rejecting it would
        # force a reconnect on a clean operational signal.
        _validate_uint64("code", self.code)

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.code) + encode_text(
            self.message, max_size=_MAX_FAILURE_MESSAGE_SIZE, label="Failure message"
        )

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "FailureResponse":
        if schema != 0:
            raise DecodeError(f"FailureResponse unsupported schema version {schema}")
        # 8 for code + >=1 for the message NUL terminator. Catches the
        # degenerate 8-byte body before decode_text's vaguer diagnostic.
        if len(data) < 9:
            raise DecodeError(
                f"FailureResponse body too short: a conforming peer "
                f"emits at least 16 bytes (8 for code + 8 for the "
                f"padded empty text block, per BytePad64). The 9-byte "
                f"minimum here catches the degenerate truncated case "
                f"before ``decode_text``'s less-actionable "
                f"'not null-terminated' error. Got {len(data)}."
            )
        code = decode_uint64(data[:8])
        message, consumed = decode_text(
            data[8:], max_size=_MAX_FAILURE_MESSAGE_SIZE, label="Failure message"
        )
        offset = 8 + consumed
        if offset != len(data):
            # The server appends the genuine failure record after an
            # un-rewound partial rows header when a row-returning statement
            # raises mid-step, framing the whole buffer as one failure body:
            #     [col_count][col_name_1 .. col_name_N][code][message]
            # So code/message above are really the column count and first
            # name; the real diagnostic is the trailing record. Frames are
            # length-delimited, so trailing bytes are benign, not a desync.
            recovered = cls._recover_trailing_failure_record(data, code)
            if recovered is not None:
                code, message = recovered
        return cls(code, _sanitize_server_text(message))

    @staticmethod
    def _recover_trailing_failure_record(data: bytes, column_count: int) -> tuple[int, str] | None:
        """Skip ``column_count`` names then decode the trailing (code, text)
        the server appended after an un-rewound partial rows header.

        Returns None when the body doesn't match this shape so the caller
        falls back to the first record, matching the reference Go client.
        """
        if column_count < 0 or column_count > _MAX_COLUMN_COUNT:
            return None
        try:
            offset = 8  # past the leading column-count uint64
            for _ in range(column_count):
                _name, consumed = decode_text(
                    data[offset:],
                    max_size=_MAX_COLUMN_NAME_SIZE,
                    label="column name",
                )
                offset += consumed
            if len(data) - offset < 9:  # 8 code + >=1 terminator
                return None
            code = decode_uint64(data[offset : offset + 8])
            message, consumed = decode_text(
                data[offset + 8 :],
                max_size=_MAX_FAILURE_MESSAGE_SIZE,
                label="Failure message",
            )
            offset += 8 + consumed
        except DecodeError:
            return None
        if offset != len(data):
            return None
        # Raw message; caller sanitises once on the value it surfaces.
        return code, message


@final
@dataclass(frozen=True, slots=True)
class LeaderResponse(Message):
    """Leader address response.

    Body: uint64 node_id, text address
    """

    MSG_TYPE: ClassVar[int] = ResponseType.LEADER

    node_id: int
    address: str

    def __post_init__(self) -> None:
        _validate_uint64("node_id", self.node_id)

    @override
    def encode_body(self) -> bytes:
        # Raft-leader atomicity invariant (mirrors decode_body): (0, "")
        # means "no leader", (N>=1, addr) a known one; (0, non-empty) is
        # malformed. Enforced here, not __post_init__, because
        # decode_body_legacy legitimately builds (0, addr) from legacy bytes.
        if self.node_id == 0 and self.address:
            # Sanitise + truncate: !r escapes LF/CR but not U+2028/bidi/ZWSP.
            sanitised = sanitize_server_text(self.address)
            display_addr = sanitised if len(sanitised) <= 64 else sanitised[:64] + "…"
            raise EncodeError(
                f"LeaderResponse: node_id=0 must be paired with an "
                f"empty address (raft-leader atomicity invariant); "
                f"got address={display_addr!r}"
            )
        return encode_uint64(self.node_id) + encode_text(
            self.address, max_size=_MAX_ADDRESS_SIZE, label="leader address"
        )

    def encode_body_legacy(self) -> bytes:
        """Encode as legacy (V0) body: text address only, no node_id.

        Rejects non-zero node_id since the legacy shape can't carry it
        (the decoder hard-codes 0); use :meth:`encode_body` for that.
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
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "LeaderResponse":
        """Decode the modern (v1+) body: uint64 node_id + text address.

        Version selection is caller-locked: ``MessageDecoder`` picks this
        vs ``decode_body_legacy`` by its constructor ``version``, never by
        sniffing bytes. A version mismatch silently misdecodes a legacy
        body's address text as node_id and routes garbage downstream.
        """
        if schema != 0:
            raise DecodeError(f"LeaderResponse unsupported schema version {schema}")
        # Response-level length guard for a self-describing diagnostic.
        if len(data) < 8:
            raise DecodeError(
                f"LeaderResponse body too short; need 8 bytes for node_id, got {len(data)}"
            )
        node_id = decode_uint64(data)
        address, consumed = decode_text(
            data[8:], max_size=_MAX_ADDRESS_SIZE, label="leader address"
        )
        offset = 8 + consumed
        if offset != len(data):
            raise DecodeError(
                f"LeaderResponse has {len(data) - offset} trailing bytes after address"
            )
        # (0, "") = no leader; (N>=1, addr) = known. (0, non-empty) is
        # malformed and rejected. (N, "") is reachable on a follower after
        # a RAFT_NOMEM on the leader-update path; treat it as "leader
        # unknown" (matching the Go/C clients) and warn for forensics.
        if node_id == 0 and address:
            # !r escapes LF/CR but not U+2028/bidi/ZWSP; sanitise first.
            sanitised = sanitize_server_text(address)
            display_addr = sanitised if len(sanitised) <= 64 else sanitised[:64] + "…"
            raise DecodeError(
                "LeaderResponse: malformed (node_id, address) — "
                f"got id={node_id!r} addr={display_addr!r}; "
                "node_id=0 must be paired with an empty address"
            )
        if node_id != 0 and not address:
            logger.warning(
                "LeaderResponse: follower reports leader_id=%d with empty "
                "address; treating as 'leader unknown' (likely RAFT_NOMEM "
                "transient on the follower's leader-update path)",
                node_id,
            )
        # Address stored raw (feeds TCP routing / allowlist comparisons);
        # sanitised at log/exception format time, not here.
        return cls(node_id, address)

    @classmethod
    def decode_body_legacy(cls, data: bytes) -> "LeaderResponse":
        """Decode legacy (pre-1.0) body: text address only; returns node_id=0."""
        # Response-level guard. 8 is the smallest padded TEXT field (NUL
        # terminator + 7 padding bytes to the 8-byte boundary).
        if len(data) < 8:
            raise DecodeError(
                f"LeaderResponse (legacy) body too short; need at least "
                f"8 bytes for the NUL-terminated address "
                f"(1-byte NUL terminator + 7-byte padding to the "
                f"8-byte boundary), got {len(data)}"
            )
        address, consumed = decode_text(data, max_size=_MAX_ADDRESS_SIZE, label="leader address")
        if consumed != len(data):
            raise DecodeError(
                f"LeaderResponse (legacy) has {len(data) - consumed} trailing bytes after address"
            )
        # Raw address; see the modern decoder for rationale.
        return cls(node_id=0, address=address)


@final
@dataclass(frozen=True, slots=True)
class WelcomeResponse(Message):
    """Client registration acknowledgment.

    Body: uint64 heartbeat_timeout, in MILLISECONDS (upstream default 15000).
    Prefer :attr:`heartbeat_timeout_seconds` over hand-dividing by 1000.
    """

    MSG_TYPE: ClassVar[int] = ResponseType.WELCOME

    heartbeat_timeout: int

    def __post_init__(self) -> None:
        _validate_uint64("heartbeat_timeout", self.heartbeat_timeout)

    @property
    def heartbeat_timeout_seconds(self) -> float:
        """Heartbeat interval converted to seconds (milliseconds / 1000)."""
        return self.heartbeat_timeout / 1000.0

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.heartbeat_timeout)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "WelcomeResponse":
        if schema != 0:
            raise DecodeError(f"WelcomeResponse unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"WelcomeResponse body must be exactly 8 bytes, got {len(data)}")
        heartbeat_timeout = decode_uint64(data)
        if heartbeat_timeout == 0:
            # Upstream never emits 0, so this signals a peer bug; warn but
            # accept (downstream falls back to the static read_timeout floor).
            logger.warning(
                "WelcomeResponse: heartbeat_timeout=0 — peer is "
                "misconfigured or non-conforming (upstream config.c "
                "defaults to 15000ms and never emits 0); downstream "
                "consumers fall back to the static read_timeout floor"
            )
        return cls(heartbeat_timeout)


@final
@dataclass(frozen=True, slots=True)
class DbResponse(Message):
    """Database opened response.

    Body: uint32 db_id, uint32 reserved
    """

    MSG_TYPE: ClassVar[int] = ResponseType.DB

    db_id: int

    def __post_init__(self) -> None:
        _validate_uint32("db_id", self.db_id)

    @override
    def encode_body(self) -> bytes:
        return encode_uint32(self.db_id) + encode_uint32(0)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "DbResponse":
        if schema != 0:
            raise DecodeError(f"DbResponse unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"DbResponse body must be exactly 8 bytes, got {len(data)}")
        db_id = decode_uint32(data)
        # Read-and-discard the reserved/pad uint32, matching Go. Permissive
        # accept of non-zero is intentional (unlike the strict header
        # reserved bytes): this body-level pad is a future extension slot.
        return cls(db_id)


@final
@dataclass
class StmtResponse(Message):
    """Statement prepared response.

    V0 body: uint32 db_id, uint32 stmt_id, uint64 num_params
    V1 body: V0 + uint64 tail_offset

    Schema is derived from ``tail_offset`` (None -> V0, set -> V1) unless the
    ``schema`` override is given. Upstream dispatches reply shape on the
    inbound PrepareRequest's schema byte, so mock servers must match it.
    The go-dqlite client never uses V1.
    """

    MSG_TYPE: ClassVar[int] = ResponseType.STMT

    db_id: int
    stmt_id: int
    num_params: int
    tail_offset: int | None = None
    # Schema-byte override so mock servers can emit "V1 with tail_offset=0"
    # (a valid upstream shape the auto-select can't reach). None = auto-select.
    schema: int | None = None

    def __post_init__(self) -> None:
        _validate_uint32("db_id", self.db_id)
        _validate_uint32("stmt_id", self.stmt_id)
        _validate_uint64("num_params", self.num_params)
        # Cap at construction so the encode/decode caps stay defense-in-depth.
        if self.num_params > _MAX_PARAM_COUNT:
            raise EncodeError(
                f"StmtResponse num_params {self.num_params} exceeds maximum ({_MAX_PARAM_COUNT})"
            )
        if self.tail_offset is not None:
            _validate_uint64("tail_offset", self.tail_offset)
            # Construction-time cap mirroring encode/decode_body.
            if self.tail_offset > _MAX_TAIL_OFFSET:
                raise EncodeError(
                    f"StmtResponse tail_offset {self.tail_offset} exceeds maximum "
                    f"({_MAX_TAIL_OFFSET})"
                )
        if self.schema is not None and self.schema not in (0, 1):
            raise EncodeError(f"StmtResponse.schema must be 0 or 1, got {self.schema}")
        if self.schema == 0 and self.tail_offset is not None:
            raise EncodeError(
                "StmtResponse: schema=0 (V0 body) cannot carry tail_offset; pass schema=1 for V1"
            )
        # Normalise schema=1 + tail_offset=None to 0 so it equals the
        # schema=1 + tail_offset=0 the decoder reconstructs (same wire bytes).
        if self.schema == 1 and self.tail_offset is None:
            # object.__setattr__ would need to become a custom __init__ if
            # this dataclass ever becomes frozen.
            object.__setattr__(self, "tail_offset", 0)

    @override
    def _get_schema(self) -> int:
        if self.schema is not None:
            return self.schema
        return 1 if self.tail_offset is not None else 0

    @override
    def encode_body(self) -> bytes:
        if self.num_params > _MAX_PARAM_COUNT:
            raise EncodeError(
                f"StmtResponse num_params {self.num_params} exceeds maximum ({_MAX_PARAM_COUNT})"
            )
        result = (
            encode_uint32(self.db_id) + encode_uint32(self.stmt_id) + encode_uint64(self.num_params)
        )
        if self._get_schema() == 1:
            # V1 always emits tail_offset (0 when None) for a 24-byte body.
            tail_offset = self.tail_offset or 0
            if tail_offset > _MAX_TAIL_OFFSET:
                raise EncodeError(
                    f"StmtResponse tail_offset {tail_offset} exceeds maximum ({_MAX_TAIL_OFFSET})"
                )
            result += encode_uint64(tail_offset)
        return result

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "StmtResponse":
        # Upstream defines only V0=0 and V1=1; the codec dispatcher already
        # caps inbound at 1, so this guards direct decode_body callers.
        if schema not in (0, 1):
            raise DecodeError(f"StmtResponse unsupported schema version {schema}; expected 0 or 1")
        expected = 24 if schema == 1 else 16
        if len(data) != expected:
            raise DecodeError(
                f"StmtResponse schema={schema} body must be exactly {expected} bytes, "
                f"got {len(data)}"
            )
        db_id = decode_uint32(data)
        stmt_id = decode_uint32(data[4:])
        num_params = decode_uint64(data[8:])
        if num_params > _MAX_PARAM_COUNT:
            raise DecodeError(
                f"StmtResponse num_params {num_params} exceeds maximum ({_MAX_PARAM_COUNT})"
            )
        tail_offset = decode_uint64(data[16:]) if schema == 1 else None
        if tail_offset is not None and tail_offset > _MAX_TAIL_OFFSET:
            raise DecodeError(
                f"StmtResponse tail_offset {tail_offset} exceeds maximum ({_MAX_TAIL_OFFSET})"
            )
        # Preserve the incoming schema byte so round-trip encode emits the
        # same shape (a V1 body with tail_offset=0 would otherwise look V0).
        return cls(db_id, stmt_id, num_params, tail_offset, schema=schema)


# Defensive cap on ``rows_affected``. Upstream returns sqlite3_changes
# (C int), so a real cluster never exceeds INT_MAX; raise this when the
# server migrates to sqlite3_changes64.
_MAX_ROWS_AFFECTED: Final[int] = (1 << 31) - 1


@final
@dataclass(frozen=True, slots=True)
class ResultResponse(Message):
    """Statement execution result.

    Body: uint64 last_insert_id, uint64 rows_affected. A negative SQLite
    rowid arrives as 2**64 - abs(rowid); use :attr:`last_insert_id_signed`
    for stdlib-sqlite parity.
    """

    MSG_TYPE: ClassVar[int] = ResponseType.RESULT

    last_insert_id: int
    rows_affected: int

    def __post_init__(self) -> None:
        _validate_uint64("last_insert_id", self.last_insert_id)
        _validate_uint64("rows_affected", self.rows_affected)
        # Construction-time cap mirroring decode_body so an over-INT_MAX
        # value can't be built into bytes the same decoder then rejects.
        if self.rows_affected > _MAX_ROWS_AFFECTED:
            raise EncodeError(
                f"ResultResponse rows_affected {self.rows_affected} exceeds maximum "
                f"({_MAX_ROWS_AFFECTED}); a real dqlite cluster cannot emit a value "
                "above INT_MAX (sqlite3_changes returns C int)"
            )

    @property
    def last_insert_id_signed(self) -> int:
        """``last_insert_id`` re-cast as int64 (the wire field is uint64)."""
        signed: int = struct.unpack("<q", struct.pack("<Q", self.last_insert_id))[0]
        return signed

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.last_insert_id) + encode_uint64(self.rows_affected)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "ResultResponse":
        if schema != 0:
            raise DecodeError(f"ResultResponse unsupported schema version {schema}")
        if len(data) != 16:
            raise DecodeError(f"ResultResponse body must be exactly 16 bytes, got {len(data)}")
        last_insert_id = decode_uint64(data)
        rows_affected = decode_uint64(data[8:])
        if rows_affected > _MAX_ROWS_AFFECTED:
            raise DecodeError(
                f"ResultResponse rows_affected {rows_affected} exceeds maximum "
                f"({_MAX_ROWS_AFFECTED}); a real dqlite cluster cannot emit a value "
                "above INT_MAX (sqlite3_changes returns C int)"
            )
        return cls(last_insert_id, rows_affected)


@final
@dataclass
class RowsResponse(Message):
    """Query result rows.

    Body: uint64 column_count, text[] column_names, then per-row
    (type header + values), ending with a DONE/PART marker.

    ``column_types`` reflects row 0's per-cell type tags (SQLite is
    dynamically typed; use ``row_types`` for accurate per-row types) and is
    empty when the frame carries no rows. A NULL in row 0 forces that
    column's tag to NULL even if a non-NULL type was supplied.

    Encode-side type inference (when both row_types and column_types are
    empty) cannot distinguish dqlite-specific encodings (UNIXTIME / ISO8601
    / BOOLEAN) from the primitives they share a layout with, so callers
    needing those MUST pass column_types or row_types explicitly.
    """

    MSG_TYPE: ClassVar[int] = ResponseType.ROWS
    DEFAULT_MAX_ROWS: ClassVar[int] = _DEFAULT_MAX_ROWS

    column_names: list[str] = field(default_factory=list)
    column_types: list[ValueType] = field(default_factory=list)
    row_types: list[list[ValueType]] = field(default_factory=list, repr=False)
    rows: list[list[WireValue]] = field(default_factory=list, repr=False)
    has_more: bool = False

    def __repr__(self) -> str:
        # Summarise unbounded rows/row_types; the dataclass-generated repr
        # walked every cell and could produce a multi-MB string.
        return (
            f"{type(self).__name__}("
            f"column_names={self.column_names!r}, "
            f"column_types={self.column_types!r}, "
            f"row_types=<{len(self.row_types)} items>, "
            f"rows=<{len(self.rows)} items>, "
            f"has_more={self.has_more}"
            f")"
        )

    def __post_init__(self) -> None:
        # Defensive deep-copies so caller-held lists can't bleed in and
        # column_types can't alias row_types[0]. The decoder skips this via
        # _from_decoded (it owns its lists; the copy was the dominant cost).
        self.column_names = list(self.column_names)
        self.column_types = list(self.column_types)
        self.row_types = [list(t) for t in self.row_types]
        self.rows = [list(r) for r in self.rows]

    @classmethod
    def _from_decoded(
        cls,
        *,
        column_names: list[str],
        column_types: list[ValueType],
        row_types: list[list[ValueType]],
        rows: list[list[WireValue]],
        has_more: bool,
    ) -> "RowsResponse":
        """Construct a :class:`RowsResponse` skipping __post_init__'s deep-copy.

        Private: every call site must own its lists outright and must break
        the column_types/row_types[0] alias itself (see ``decode_body``).
        """
        obj = object.__new__(cls)
        obj.column_names = column_names
        obj.column_types = column_types
        obj.row_types = row_types
        obj.rows = rows
        obj.has_more = has_more
        return obj

    def _get_row_types(self, row_idx: int, row: list[WireValue]) -> list[ValueType]:
        """Get types for a row: from row_types, column_types, or inferred.

        Returns a fresh list (never aliases self.column_types). None values
        override the declared type to NULL, matching Go's per-row header.
        """
        if self.row_types and row_idx < len(self.row_types):
            types = list(self.row_types[row_idx])
        elif self.column_types:
            types = list(self.column_types)
        else:
            # Type-only inference (the isinstance ladder), not the full
            # encode pipeline; oversize values still reject in encode_row_values.
            types = [_infer_value_type(v) for v in row]

        # Override to NULL for None, applied uniformly across all three paths
        # so it isn't silently load-bearing on _infer_value_type(None)==NULL.
        for i, v in enumerate(row):
            if v is None and i < len(types):
                types[i] = ValueType.NULL
        return types

    @override
    def encode_body(self) -> bytes:
        col_count = len(self.column_names)
        if col_count > _MAX_COLUMN_COUNT:
            raise EncodeError(
                f"RowsResponse column count {col_count} exceeds maximum ({_MAX_COLUMN_COUNT})"
            )
        # Per-name byte cap is enforced inside encode_text (capping here
        # would check codepoints, not the UTF-8 bytes the decoder caps on).
        if self.column_types and len(self.column_types) != col_count:
            raise EncodeError(
                f"column_types length ({len(self.column_types)}) != "
                f"column_names length ({col_count})"
            )
        # row_types must be empty (infer per-row) or match rows one-to-one.
        if self.row_types and len(self.row_types) != len(self.rows):
            raise EncodeError(
                f"row_types length ({len(self.row_types)}) != "
                f"rows length ({len(self.rows)}); pass an empty row_types "
                f"to infer per-row types from the values"
            )
        # Zero-column rows encode to zero bytes, indistinguishable from a
        # zero-row set; reject rather than silently lose the row count.
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

        # bytearray accumulation avoids the O(N^2) memcopy of bytes += bytes.
        result = bytearray()
        result.extend(encode_uint64(col_count))

        for name in self.column_names:
            result.extend(encode_text(name, max_size=_MAX_COLUMN_NAME_SIZE, label="column name"))

        # Each row carries its own type header.
        for i, row in enumerate(self.rows):
            types = self._get_row_types(i, row)
            result.extend(encode_row_header(types))
            result.extend(encode_row_values(row, types))

        marker = ROW_PART_MARKER if self.has_more else ROW_DONE_MARKER
        result.extend(encode_uint64(marker))

        return bytes(result)

    @classmethod
    @override
    def decode_body(
        cls,
        data: bytes,
        schema: int = 0,
        max_rows: int = _DEFAULT_MAX_ROWS,
        *,
        text_errors: str = "strict",
    ) -> "RowsResponse":
        if schema != 0:
            raise DecodeError(f"RowsResponse unsupported schema version {schema}")
        # memoryview so per-iteration slices are O(1), not O(remaining).
        view = memoryview(data)
        offset = 0

        column_count = decode_uint64(view[offset:])
        offset += 8

        if column_count > _MAX_COLUMN_COUNT:
            raise DecodeError(f"Column count {column_count} exceeds maximum {_MAX_COLUMN_COUNT}")

        # Each name is >=8 bytes and an 8-byte marker follows; reserving the
        # marker gives a clear diagnostic instead of "body exhausted" later.
        remaining = len(view) - offset
        max_columns = max(0, remaining - WORD_SIZE) // 8
        if column_count > max_columns:
            raise DecodeError(
                f"Column count {column_count} exceeds maximum possible in "
                f"{remaining} bytes of remaining data (reserving {WORD_SIZE} "
                f"bytes for the row end marker)"
            )

        column_names: list[str] = []
        for _ in range(column_count):
            name, consumed = decode_text(
                view[offset:], max_size=_MAX_COLUMN_NAME_SIZE, label="column name"
            )
            column_names.append(name)
            offset += consumed

        rows: list[list[Any]] = []
        all_row_types: list[list[ValueType]] = []
        column_types: list[ValueType] = []

        # Zero-column results carry no row data; skip the loop and validate
        # the full 8-byte marker (a first-byte compare would accept torn ones).
        if column_count == 0:
            if offset + WORD_SIZE > len(view):
                raise DecodeError(
                    "RowsResponse body exhausted without end marker (zero-column result)"
                )
            # memoryview supports buffer-protocol equality, so compare the
            # prefix directly; only materialise to bytes in the error arm.
            marker_view = view[offset : offset + WORD_SIZE]
            if marker_view == _ROW_DONE_MARKER:
                has_more = False
            elif marker_view == _ROW_PART_MARKER:
                has_more = True
            else:
                got_hex = bytes(marker_view).hex()
                raise DecodeError(
                    f"Expected DONE or PART marker for zero-column result, got 0x{got_hex}"
                )
            # Strict-decode: buffer must be exhausted (Python-specific path;
            # upstream C never emits zero-column result sets).
            end = offset + WORD_SIZE
            if end != len(view):
                raise DecodeError(
                    f"RowsResponse zero-column body has {len(view) - end} "
                    "trailing bytes after DONE/PART marker"
                )
            return cls._from_decoded(
                column_names=column_names,
                column_types=[],
                row_types=[],
                rows=[],
                has_more=has_more,
            )

        while offset < len(view):
            result, consumed = decode_row_header(view[offset:], column_count)
            offset += consumed

            if result is RowMarker.DONE or result is RowMarker.PART:
                # Strict-decode: trailing bytes after the marker are malformed.
                if offset != len(view):
                    raise DecodeError(
                        f"RowsResponse body has {len(view) - offset} trailing bytes "
                        f"after {'DONE' if result is RowMarker.DONE else 'PART'} marker "
                        f"(decoded {len(rows)} rows)"
                    )
                # Break the column_types/row_types[0] alias here, since
                # _from_decoded bypasses __post_init__'s defensive copy.
                if all_row_types and column_types is all_row_types[0]:
                    column_types = list(column_types)
                return cls._from_decoded(
                    column_names=column_names,
                    column_types=column_types,
                    row_types=all_row_types,
                    rows=rows,
                    has_more=result is RowMarker.PART,
                )

            types = result
            if not isinstance(types, list):
                raise DecodeError(f"Expected column types list, got {type(types).__name__}")
            all_row_types.append(types)
            if not column_types:
                column_types = types

            values, consumed = decode_row_values(view[offset:], types, text_errors=text_errors)
            rows.append(values)
            offset += consumed

            if len(rows) >= max_rows:
                # Cap is exclusive: reaching max_rows is rejected.
                raise DecodeError(
                    f"Row count {len(rows)} reached limit {max_rows} "
                    f"(cap is exclusive — raise max_rows if larger "
                    f"result sets are expected)"
                )

        raise DecodeError(
            f"RowsResponse body exhausted without end marker "
            f"(decoded {len(rows)} rows, consumed {offset} of {len(view)} bytes)"
        )


@final
@dataclass(frozen=True, slots=True)
class EmptyResponse(Message):
    """Empty response (for exec with no result).

    Body: uint64 (reserved, unused)
    """

    MSG_TYPE: ClassVar[int] = ResponseType.EMPTY

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(0)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "EmptyResponse":
        if schema != 0:
            raise DecodeError(f"EmptyResponse unsupported schema version {schema}")
        if len(data) != 8:
            raise DecodeError(f"EmptyResponse body must be exactly 8 bytes, got {len(data)}")
        # Read-and-discard the reserved uint64, matching Go.
        decode_uint64(data)
        return cls()


@final
@dataclass
class FilesResponse(Message):
    """Database dump files response.

    Body: uint64 count, then repeated (text filename, uint64 size, raw content).
    Content is not word-padded; the C server asserts it is always aligned.
    """

    MSG_TYPE: ClassVar[int] = ResponseType.FILES

    files: dict[str, bytes] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        # Summary repr (see RowsResponse.__repr__); content can be ~64 MiB.
        head = list(self.files.items())[:3]
        more = len(self.files) - len(head)
        sample = ", ".join(f"{name!r}: <{len(content)} bytes>" for name, content in head)
        tail = f", +{more} more" if more > 0 else ""
        return f"{type(self).__name__}(files={{{sample}{tail}}})"

    @override
    def encode_body(self) -> bytes:
        if len(self.files) > _MAX_FILE_COUNT:
            raise EncodeError(
                f"FilesResponse count {len(self.files)} exceeds maximum ({_MAX_FILE_COUNT})"
            )
        # bytearray accumulation avoids the O(N^2) memcopy of bytes += bytes.
        result = bytearray()
        result.extend(encode_uint64(len(self.files)))
        for name, content in self.files.items():
            # Upstream writes entries back-to-back with no padding and asserts
            # content is 8-byte aligned; match that so frames don't diverge.
            if len(content) % 8 != 0:
                raise EncodeError(
                    f"FilesResponse content for {name!r} must be 8-byte aligned "
                    f"(got {len(content)} bytes); dqlite file entries carry no "
                    "per-file padding"
                )
            # Per-file content cap mirroring the decode-side guard below.
            if len(content) > _MAX_FILE_CONTENT_SIZE:
                raise EncodeError(
                    f"FilesResponse content for {name!r} length {len(content)} "
                    f"exceeds maximum ({_MAX_FILE_CONTENT_SIZE})"
                )
            result.extend(encode_text(name, max_size=_MAX_FILENAME_SIZE, label="filename"))
            result.extend(encode_uint64(len(content)))
            result.extend(content)
        return bytes(result)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "FilesResponse":
        if schema != 0:
            raise DecodeError(f"FilesResponse unsupported schema version {schema}")
        view = memoryview(data)  # O(1) slicing in the per-file loop
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
            # Mirror the encode-side 8-byte alignment invariant.
            if size % 8 != 0:
                raise DecodeError(
                    f"FilesResponse content for {name!r} must be 8-byte aligned (got {size} bytes)"
                )
            # Per-file content cap. Checked before the over-read bounds-check
            # so an oversize claim gets a specific (not "truncated") diagnostic.
            if size > _MAX_FILE_CONTENT_SIZE:
                raise DecodeError(
                    f"FilesResponse content for {name!r} length {size} "
                    f"exceeds maximum ({_MAX_FILE_CONTENT_SIZE})"
                )
            if offset + size > len(view):
                raise DecodeError(
                    f"FilesResponse file content truncated: expected {size} bytes "
                    f"at offset {offset}, but only {len(view) - offset} bytes available"
                )
            content = bytes(view[offset : offset + size])
            offset += size  # no padding after content
            # Reject duplicate filenames: overwriting via dict would make
            # len(files) < count and break re-encode symmetry.
            if name in files:
                raise DecodeError(f"FilesResponse: duplicate filename {name!r}")
            files[name] = content
        # Strict-decode: body must be fully consumed.
        if offset != len(view):
            raise DecodeError(
                f"FilesResponse has {len(view) - offset} trailing bytes after last file"
            )
        return cls(files)


@final
@dataclass(frozen=True, slots=True)
class NodeInfo:
    """Information about a cluster node (frozen + hashable for set/dict keys)."""

    node_id: int
    address: str
    role: NodeRole

    def __post_init__(self) -> None:
        _validate_uint64("node_id", self.node_id)
        # Raft-configuration invariant (mirrors ServersResponse.decode_body):
        # real config rows always have node_id >= 1 (id=0 is the "no leader"
        # sentinel) and a non-empty address. Validated here so a bad caller-
        # built NodeInfo can't encode to bytes the decoder then refuses.
        if self.node_id == 0:
            raise EncodeError(
                f"NodeInfo: node_id must be >= 1 (raft-configuration "
                f"invariant); got node_id={self.node_id}"
            )
        if not self.address:
            raise EncodeError(
                f"NodeInfo: address must be non-empty for node_id="
                f"{self.node_id} (raft-configuration invariant)"
            )
        if isinstance(self.role, NodeRole):
            return
        # Check uint64 range first so a giant int gets the "exceeds uint64"
        # diagnostic rather than the enum's "unknown role"; then coerce.
        _validate_uint64("role", self.role)
        try:
            coerced = NodeRole(self.role)
        except ValueError as e:
            raise EncodeError(
                f"NodeInfo: unknown role {self.role!r}; valid roles are "
                f"0 (VOTER), 1 (STANDBY), 2 (SPARE)"
            ) from e
        # object.__setattr__ would need a custom __init__ if this dataclass
        # ever becomes frozen.
        object.__setattr__(self, "role", coerced)


@final
@dataclass
class ServersResponse(Message):
    """Cluster servers response.

    Body: uint64 count, then repeated (uint64 node_id, text address, uint64 role)
    """

    MSG_TYPE: ClassVar[int] = ResponseType.SERVERS

    nodes: list[NodeInfo] = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        # Summary repr (see RowsResponse.__repr__); nodes can be up to 10_000.
        head_repr = ", ".join(repr(n) for n in self.nodes[:3])
        more = len(self.nodes) - 3
        tail = f", +{more} more" if more > 0 else ""
        return f"{type(self).__name__}(nodes=[{head_repr}{tail}])"

    @override
    def encode_body(self) -> bytes:
        if len(self.nodes) > _MAX_NODE_COUNT:
            raise EncodeError(
                f"ServersResponse node count {len(self.nodes)} exceeds maximum ({_MAX_NODE_COUNT})"
            )
        # bytearray accumulation, same O(N^2) avoidance as the sibling encoders.
        result = bytearray()
        result.extend(encode_uint64(len(self.nodes)))
        for node in self.nodes:
            result.extend(encode_uint64(node.node_id))
            result.extend(
                encode_text(node.address, max_size=_MAX_ADDRESS_SIZE, label="server address")
            )
            result.extend(encode_uint64(node.role))
        return bytes(result)

    @classmethod
    @override
    def decode_body(
        cls,
        data: bytes,
        schema: int = 0,
        *,
        unknown_role_policy: str = "reject",
    ) -> "ServersResponse":
        """Decode the modern V1 cluster body (id + address + role per node).

        Only V1 is parsed; the legacy V0 (id + address) shape is no longer
        requestable in-tree, and a V0 body from a hostile peer is caught by
        the bounds/trailing-bytes checks in the common case.

        ``unknown_role_policy`` handles roles this client predates:
        - "reject" (default): raise DecodeError.
        - "warn": substitute NodeRole.SPARE (safe default) and log.
        - "accept": substitute SPARE silently.
        """
        if schema != 0:
            raise DecodeError(f"ServersResponse unsupported schema version {schema}")
        if unknown_role_policy not in ("reject", "warn", "accept"):
            raise DecodeError(
                "unknown_role_policy must be 'reject', 'warn', or "
                f"'accept'; got {unknown_role_policy!r}"
            )
        view = memoryview(data)  # O(1) slicing in the per-node loop
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
        # Collect unknown roles so warn-mode emits one WARNING per response,
        # not one per node.
        unknown_seen: list[int] = []
        # Strict-decode: reject duplicate node_id (would let consumers keying
        # by id silently overwrite a prior entry).
        seen_ids: set[int] = set()
        for _ in range(count):
            node_id = decode_uint64(view[offset:])
            offset += 8
            address, consumed = decode_text(
                view[offset:], max_size=_MAX_ADDRESS_SIZE, label="server address"
            )
            # Raw address (feeds TCP routing / allowlist comparisons);
            # sanitised at format time. See LeaderResponse.decode_body.
            offset += consumed
            # Raft-config invariant: only (id>=1, non-empty) is legitimate.
            # (0, *) or (>=1, "") would propagate phantom/routeless nodes.
            if node_id == 0 or not address:
                # !r escapes LF/CR but not U+2028/bidi/ZWSP; sanitise first.
                sanitised = sanitize_server_text(address)
                display_addr = sanitised if len(sanitised) <= 64 else sanitised[:64] + "…"
                raise DecodeError(
                    "ServersResponse: malformed (node_id, address) entry — "
                    f"got id={node_id!r} addr={display_addr!r}; "
                    "expected node_id>=1 with non-empty address "
                    "(raft configuration invariant)"
                )
            if node_id in seen_ids:
                raise DecodeError(f"Duplicate node_id {node_id} in ServersResponse")
            seen_ids.add(node_id)
            raw_role = decode_uint64(view[offset:])
            offset += 8
            try:
                role = NodeRole(raw_role)
            except ValueError as exc:
                if unknown_role_policy == "reject":
                    valid = sorted(r.value for r in NodeRole)
                    raise DecodeError(
                        f"Invalid node role {raw_role} at offset {offset - 8}; "
                        f"expected one of {valid}"
                    ) from exc
                # warn / accept: substitute SPARE (safe default).
                if unknown_role_policy == "warn":
                    unknown_seen.append(raw_role)
                role = NodeRole.SPARE
            nodes.append(NodeInfo(node_id, address, role))
        if unknown_seen:
            unique_roles = sorted(set(unknown_seen))
            logger.warning(
                "ServersResponse: substituted SPARE for %d node(s) with "
                "unknown role(s) %s under unknown_role_policy='warn'",
                len(unknown_seen),
                unique_roles,
            )
        if offset != len(view):
            raise DecodeError(
                f"ServersResponse has {len(view) - offset} trailing bytes after {count} nodes"
            )
        return cls(nodes)


@final
@dataclass(frozen=True, slots=True)
class MetadataResponse(Message):
    """Node metadata response (reply to a DescribeRequest).

    Body: uint64 failure_domain, uint64 weight
    """

    MSG_TYPE: ClassVar[int] = ResponseType.METADATA

    failure_domain: int
    weight: int

    def __post_init__(self) -> None:
        _validate_uint64("failure_domain", self.failure_domain)
        _validate_uint64("weight", self.weight)

    @override
    def encode_body(self) -> bytes:
        return encode_uint64(self.failure_domain) + encode_uint64(self.weight)

    @classmethod
    @override
    def decode_body(cls, data: bytes, schema: int = 0) -> "MetadataResponse":
        if schema != 0:
            raise DecodeError(f"MetadataResponse unsupported schema version {schema}")
        if len(data) != 16:
            raise DecodeError(f"MetadataResponse body must be exactly 16 bytes, got {len(data)}")
        failure_domain = decode_uint64(data)
        weight = decode_uint64(data[8:])
        return cls(failure_domain, weight)
