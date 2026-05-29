"""Pure Python wire protocol implementation for dqlite."""

import logging as _logging
import os as _os
import sys as _sys

# Refuse free-threaded CPython (PEP 703): ReadBuffer/WriteBuffer rely on the
# GIL's bytearray atomicity, so concurrent access SIGSEGVs or hangs the process
# rather than raising. Override with DQLITEWIRE_ALLOW_FREE_THREADED=1.
if hasattr(_sys, "_is_gil_enabled") and not _sys._is_gil_enabled():
    if _os.environ.get("DQLITEWIRE_ALLOW_FREE_THREADED") != "1":
        raise ImportError(
            "dqlitewire does not support free-threaded Python "
            "(python3.13t / no-GIL). The ReadBuffer and WriteBuffer classes "
            "rely on bytearray mutation semantics that cause SIGSEGV and "
            "process hangs under free-threading. "
            "To override at your own risk, set "
            "DQLITEWIRE_ALLOW_FREE_THREADED=1."
        )
    import warnings as _warnings

    _warnings.warn(
        "dqlitewire is running under free-threaded Python with "
        "DQLITEWIRE_ALLOW_FREE_THREADED=1. ReadBuffer and WriteBuffer are "
        "known to crash or hang under concurrent access on this runtime. "
        "Use strictly single-owner-per-instance.",
        RuntimeWarning,
        stacklevel=2,
    )

from typing import Final as _Final

from dqlitewire import messages, tuples, types
from dqlitewire.buffer import ReadBuffer, WriteBuffer
from dqlitewire.codec import MessageDecoder, MessageEncoder, decode_message, encode_message
from dqlitewire.constants import (
    BARE_DATABASE_ERROR_CODES,
    DEFAULT_MAX_CONTINUATION_FRAMES,
    DEFAULT_MAX_TOTAL_ROWS,
    DQLITE_NOTFOUND,
    DQLITE_PARSE,
    DQLITE_PROTO,
    HEADER_SIZE,
    LEADER_ERROR_CODES,
    LEADER_LOST_DB_LOOKUP_SUBSTRING,
    NO_TRANSACTION_MESSAGE_SUBSTRINGS,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_LEGACY,
    ROW_DONE_BYTE,
    ROW_DONE_MARKER,
    ROW_PART_BYTE,
    ROW_PART_MARKER,
    SQLITE_ABORT,
    SQLITE_AUTH,
    SQLITE_BUSY,
    SQLITE_CONSTRAINT,
    SQLITE_CONSTRAINT_CHECK,
    SQLITE_CONSTRAINT_COMMITHOOK,
    SQLITE_CONSTRAINT_FOREIGNKEY,
    SQLITE_CONSTRAINT_FUNCTION,
    SQLITE_CONSTRAINT_NOTNULL,
    SQLITE_CONSTRAINT_PINNED,
    SQLITE_CONSTRAINT_PRIMARYKEY,
    SQLITE_CONSTRAINT_ROWID,
    SQLITE_CONSTRAINT_TRIGGER,
    SQLITE_CONSTRAINT_UNIQUE,
    SQLITE_CONSTRAINT_VTAB,
    SQLITE_CORRUPT,
    SQLITE_ERROR,
    SQLITE_FORMAT,
    SQLITE_FULL,
    SQLITE_INTERNAL,
    SQLITE_INTERRUPT,
    SQLITE_IOERR,
    SQLITE_IOERR_LEADERSHIP_LOST,
    SQLITE_IOERR_LEADERSHIP_LOST_LEGACY,
    SQLITE_IOERR_NOT_LEADER,
    SQLITE_IOERR_NOT_LEADER_LEGACY,
    SQLITE_MISMATCH,
    SQLITE_MISUSE,
    SQLITE_NOLFS,
    SQLITE_NOMEM,
    SQLITE_NOTADB,
    SQLITE_NOTFOUND,
    SQLITE_NOTICE,
    SQLITE_PRIMARY_CODE_MASK,
    SQLITE_PROTOCOL,
    SQLITE_RANGE,
    SQLITE_TOOBIG,
    SQLITE_WARNING,
    TX_AUTO_ROLLBACK_PRIMARY_CODES,
    WIRE_DECODE_FAILED_PREFIX,
    WORD_SIZE,
    NodeRole,
    RequestType,
    ResponseType,
    ValueType,
    is_dqlite_namespace_code,
    primary_sqlite_code,
)
from dqlitewire.exceptions import (
    ContinuationError,
    DecodeError,
    EncodeError,
    HandshakeError,
    PoisonedError,
    ProtocolError,
    ServerFailure,
    StreamError,
)
from dqlitewire.messages.base import Header, Message
from dqlitewire.messages.responses import NodeInfo, sanitize_for_log, sanitize_server_text
from dqlitewire.truncate import DEFAULT_MAX_RAW_MESSAGE, cap_raw_message
from dqlitewire.tuples import RowMarker
from dqlitewire.types import WireInput, WireValue

__version__: _Final[str] = "0.2.2"

__all__ = [
    "BARE_DATABASE_ERROR_CODES",
    "DEFAULT_MAX_CONTINUATION_FRAMES",
    "DEFAULT_MAX_TOTAL_ROWS",
    "DQLITE_NOTFOUND",
    "DQLITE_PARSE",
    "DQLITE_PROTO",
    "HEADER_SIZE",
    "LEADER_ERROR_CODES",
    "LEADER_LOST_DB_LOOKUP_SUBSTRING",
    "NO_TRANSACTION_MESSAGE_SUBSTRINGS",
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_LEGACY",
    "ROW_DONE_BYTE",
    "ROW_DONE_MARKER",
    "ROW_PART_BYTE",
    "ROW_PART_MARKER",
    "SQLITE_ABORT",
    "SQLITE_AUTH",
    "SQLITE_BUSY",
    "SQLITE_CONSTRAINT",
    "SQLITE_CONSTRAINT_CHECK",
    "SQLITE_CONSTRAINT_COMMITHOOK",
    "SQLITE_CONSTRAINT_FOREIGNKEY",
    "SQLITE_CONSTRAINT_FUNCTION",
    "SQLITE_CONSTRAINT_NOTNULL",
    "SQLITE_CONSTRAINT_PINNED",
    "SQLITE_CONSTRAINT_PRIMARYKEY",
    "SQLITE_CONSTRAINT_ROWID",
    "SQLITE_CONSTRAINT_TRIGGER",
    "SQLITE_CONSTRAINT_UNIQUE",
    "SQLITE_CONSTRAINT_VTAB",
    "SQLITE_CORRUPT",
    "SQLITE_ERROR",
    "SQLITE_FORMAT",
    "SQLITE_FULL",
    "SQLITE_INTERNAL",
    "SQLITE_INTERRUPT",
    "SQLITE_IOERR",
    "SQLITE_IOERR_LEADERSHIP_LOST",
    "SQLITE_IOERR_LEADERSHIP_LOST_LEGACY",
    "SQLITE_IOERR_NOT_LEADER",
    "SQLITE_IOERR_NOT_LEADER_LEGACY",
    "SQLITE_MISMATCH",
    "SQLITE_MISUSE",
    "SQLITE_NOLFS",
    "SQLITE_NOMEM",
    "SQLITE_NOTADB",
    "SQLITE_NOTFOUND",
    "SQLITE_NOTICE",
    "SQLITE_PRIMARY_CODE_MASK",
    "SQLITE_PROTOCOL",
    "SQLITE_RANGE",
    "SQLITE_TOOBIG",
    "SQLITE_WARNING",
    "TX_AUTO_ROLLBACK_PRIMARY_CODES",
    "WIRE_DECODE_FAILED_PREFIX",
    "WORD_SIZE",
    "ContinuationError",
    "DEFAULT_MAX_RAW_MESSAGE",
    "DecodeError",
    "EncodeError",
    "HandshakeError",
    "Header",
    "Message",
    "MessageDecoder",
    "MessageEncoder",
    "NodeInfo",
    "NodeRole",
    "PoisonedError",
    "ProtocolError",
    "ReadBuffer",
    "RequestType",
    "ResponseType",
    "RowMarker",
    "ServerFailure",
    "StreamError",
    "ValueType",
    "WireInput",
    "WireValue",
    "WriteBuffer",
    "__version__",
    "cap_raw_message",
    "decode_message",
    "encode_message",
    "is_dqlite_namespace_code",
    "messages",
    "primary_sqlite_code",
    "sanitize_for_log",
    "sanitize_server_text",
    "tuples",
    "types",
]

# Library NullHandler: suppress lastResort stderr when the app hasn't
# configured logging.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())
