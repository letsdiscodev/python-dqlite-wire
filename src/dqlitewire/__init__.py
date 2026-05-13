"""Pure Python wire protocol implementation for dqlite.

Schema coverage matrix (Python vs canonical Go vs C server):

==================  ==========  ==========  ============
Request / Response  Go canon    Python      C server
==================  ==========  ==========  ============
Exec                0, 1        0, 1        0, 1
Query               0, 1        0, 1        0, 1
ExecSQL             0, 1        0, 1        0, 1
QuerySQL            0, 1        0, 1        0, 1
Prepare             0           0, 1 (*)    0, 1
Stmt response       0           0, 1 (*)    0, 1
==================  ==========  ==========  ============

(*) Python supports schema=1 for Prepare / Stmt — not Go-canonical
but interoperates with the C server's V1 shape (carries
``tail_offset`` to the response so a multi-statement script's tail
can be re-prepared without round-tripping the original SQL again).
A Python ``PrepareRequest(schema=1)`` against a Go-canonical client
would be rejected; against a real C server it works.

Cap defaults that affect interop and hostile-server resistance:

- ``ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE``: 64 MiB. Go has no upper
  bound on message size (``recvBody`` doubles indefinitely until
  ``n`` fits). A legitimate dqlite server emitting a Files response
  from a 1+ GB database would be silently accepted by Go and rejected
  by Python. The cap is configurable on ``ReadBuffer`` / ``MessageDecoder``
  via the ``max_message_size`` kwarg, but the upstream
  ``dqliteclient`` / ``dqlitedbapi`` / ``sqlalchemydqlite`` packages
  do NOT plumb the override through their public surfaces — large-
  result workloads must either drive the wire layer directly or
  accept the 64 MiB default at the higher layers.
- ``_MAX_BLOB_SIZE`` and ``_MAX_TEXT_VALUE_SIZE``: 64 MiB each
  (aligned with the message-size cap so a value fitting in the
  frame envelope is not rejected at the inner cap).
- ``_MAX_COLUMN_COUNT``: 255 (matches the C server's
  ``STMT__MAX_COLUMNS``; any column count above 255 from a real
  cluster is provably malformed).

Empty-params encoding: empty ``Tuple`` parameters encode as zero
bytes by default (Go-style); the C server's ``tuple_encoder``
emits 8 bytes for the same shape — a zero count (1 byte in V0,
4 bytes in V1) followed by zero-padding to the 8-byte boundary.
The encoded byte content is the same eight zero bytes for both
schemas. The decoder accepts both forms;
``encode_params_tuple(emit_empty_header=True)`` produces the
C-style 8-byte empty-params body for callers that need byte-
identity round-trip with C-origin captures.

UNIXTIME columns decode to ``int`` (seconds since epoch), unlike
Go which returns ``time.Time``. Conversion to ``datetime`` is the
dbapi layer's job; bare wire-layer consumers must convert
themselves.

``RowsResponse._get_row_types`` falls through to ``encode_value(v)[1]``
inference when both ``row_types`` and ``column_types`` are empty.
This inference cannot pick UNIXTIME (server-only), ISO8601 (string
indistinguishable from TEXT), or BOOLEAN (only Python ``bool``
triggers, not int 0/1 from a BOOLEAN-decltype column). Pass
``column_types`` or ``row_types`` explicitly when constructing
``RowsResponse`` for byte-shape parity with real C-server emission.

Disconnect-classification constants
-----------------------------------

Two string constants anchor the four-package disconnect-classification
chain (wire failure → ``client.ProtocolError`` →
``dbapi.OperationalError(code=None)`` → SA's
``_dqlite_disconnect_messages`` substring scan):

- ``WIRE_DECODE_FAILED_PREFIX = "wire decode failed"`` — the canonical
  lowercase phrase emitted by every site that wraps a wire-decode
  error into a client / dbapi ``OperationalError``. SA's substring
  scan lowercases the rendered exception text before matching, so
  emission-side casing is flexible; the constant value is the
  canonical form. Five emitter sites across ``dqliteclient.protocol``
  (3), ``dqliteclient.connection`` (handshake-rewrap), and
  ``dqlitedbapi.connection`` (connect-rewrap) reference the constant;
  ``sqlalchemydqlite.base`` is the consumer.

- ``LEADER_LOST_DB_LOOKUP_SUBSTRING = "no database opened"`` — the
  verbatim wording upstream ``gateway.c::LOOKUP_DB`` emits with
  code ``SQLITE_NOTFOUND`` (=12) after a Raft demotion. Substring-
  gates the leader-flip arm so the orthogonal ``LOOKUP_STMT``
  emission ("no statement with the given id ...") stays as a benign
  ``InternalError`` instead of triggering pool invalidation. Mirrors
  Go's ``driverError`` ``errNotFound → ErrBadConn`` arm. Consumed by
  ``dqliteclient.connection._run_protocol`` and
  ``sqlalchemydqlite.base::is_disconnect``.

Both constants are exported and load-bearing across packages — a
rename ripples through grep on the constant name.
"""

import logging as _logging
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

from typing import Final

from dqlitewire import messages, tuples, types
from dqlitewire.buffer import ReadBuffer, WriteBuffer
from dqlitewire.codec import MessageDecoder, MessageEncoder, decode_message, encode_message
from dqlitewire.constants import (
    DEFAULT_MAX_CONTINUATION_FRAMES,
    DEFAULT_MAX_TOTAL_ROWS,
    DQLITE_NOTFOUND,
    DQLITE_PARSE,
    DQLITE_PROTO,
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
from dqlitewire.messages.responses import sanitize_for_log, sanitize_server_text
from dqlitewire.types import WireInput, WireValue

__version__: Final[str] = "0.1.5"

__all__ = [
    "DEFAULT_MAX_CONTINUATION_FRAMES",
    "DEFAULT_MAX_TOTAL_ROWS",
    "DQLITE_NOTFOUND",
    "DQLITE_PARSE",
    "DQLITE_PROTO",
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
    "messages",
    "primary_sqlite_code",
    "sanitize_for_log",
    "sanitize_server_text",
    "tuples",
    "types",
]

# Convention from the Python logging HOWTO: every library adds a
# ``NullHandler`` to its top-level logger so an application that
# hasn't configured logging doesn't see the ``lastResort`` stderr
# emission, and downstream code that wants to silence the library
# with ``getLogger("dqlitewire").propagate = False`` works cleanly.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())
