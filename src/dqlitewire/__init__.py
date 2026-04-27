"""Pure Python wire protocol implementation for dqlite."""

import os as _os
import sys as _sys

# Refuse to run under free-threaded CPython (python3.13t / PEP 703).
#
# ReadBuffer and WriteBuffer are backed by bytearray and rely on the GIL's
# C-level atomicity for bytearray.extend(), slicing, and attribute rebinding
# inside _maybe_compact(). Under a free-threaded build, concurrent access
# produces SIGSEGV in the read path and a process-level hang in the write
# path. The single-owner-per-instance contract alone
# is not enough to keep users safe, because "share across threads" is an
# easy mistake and the failure mode is an interpreter crash rather than a
# Python exception.
#
# The check can be overridden with DQLITEWIRE_ALLOW_FREE_THREADED=1 — a
# RuntimeWarning is still emitted in that case so test configurations that
# promote warnings to errors surface the risk.
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

from dqlitewire.buffer import ReadBuffer, WriteBuffer
from dqlitewire.codec import MessageDecoder, MessageEncoder, decode_message, encode_message
from dqlitewire.constants import (
    DEFAULT_MAX_CONTINUATION_FRAMES,
    DEFAULT_MAX_TOTAL_ROWS,
    DQLITE_NOTFOUND,
    DQLITE_PARSE,
    LEADER_ERROR_CODES,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_LEGACY,
    ROW_DONE_BYTE,
    ROW_DONE_MARKER,
    ROW_PART_BYTE,
    ROW_PART_MARKER,
    SQLITE_ABORT,
    SQLITE_BUSY,
    SQLITE_CORRUPT,
    SQLITE_ERROR,
    SQLITE_FULL,
    SQLITE_INTERRUPT,
    SQLITE_IOERR,
    SQLITE_IOERR_LEADERSHIP_LOST,
    SQLITE_IOERR_NOT_LEADER,
    SQLITE_NOMEM,
    SQLITE_NOTFOUND,
    SQLITE_PRIMARY_CODE_MASK,
    SQLITE_PROTOCOL,
    TX_AUTO_ROLLBACK_PRIMARY_CODES,
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
from dqlitewire.types import WireInput, WireValue

__version__ = "0.1.3"

__all__ = [
    "DEFAULT_MAX_CONTINUATION_FRAMES",
    "DEFAULT_MAX_TOTAL_ROWS",
    "DQLITE_NOTFOUND",
    "DQLITE_PARSE",
    "LEADER_ERROR_CODES",
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_LEGACY",
    "ROW_DONE_BYTE",
    "ROW_DONE_MARKER",
    "ROW_PART_BYTE",
    "ROW_PART_MARKER",
    "SQLITE_ABORT",
    "SQLITE_BUSY",
    "SQLITE_CORRUPT",
    "SQLITE_ERROR",
    "SQLITE_FULL",
    "SQLITE_INTERRUPT",
    "SQLITE_IOERR",
    "SQLITE_IOERR_LEADERSHIP_LOST",
    "SQLITE_NOMEM",
    "SQLITE_IOERR_NOT_LEADER",
    "SQLITE_NOTFOUND",
    "SQLITE_PRIMARY_CODE_MASK",
    "SQLITE_PROTOCOL",
    "TX_AUTO_ROLLBACK_PRIMARY_CODES",
    "ContinuationError",
    "DecodeError",
    "EncodeError",
    "HandshakeError",
    "Header",
    "Message",
    "MessageDecoder",
    "MessageEncoder",
    "NodeRole",
    "PoisonedError",
    "ProtocolError",
    "ReadBuffer",
    "RequestType",
    "ResponseType",
    "ServerFailure",
    "StreamError",
    "ValueType",
    "WireInput",
    "WireValue",
    "WriteBuffer",
    "__version__",
    "decode_message",
    "encode_message",
    "is_dqlite_namespace_code",
    "primary_sqlite_code",
]
