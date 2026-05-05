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
  by Python. Override via the ``max_message_size`` kwarg on
  ``ReadBuffer`` / ``MessageDecoder`` for large-result workloads.
- ``_MAX_BLOB_SIZE`` and ``_MAX_TEXT_VALUE_SIZE``: 64 MiB each
  (aligned with the message-size cap so a value fitting in the
  frame envelope is not rejected at the inner cap).
- ``_MAX_COLUMN_COUNT``: 255 (matches the C server's
  ``STMT__MAX_COLUMNS``; any column count above 255 from a real
  cluster is provably malformed).

Empty-params encoding: empty ``Tuple`` parameters encode as zero
bytes by default (Go-style); the C server's ``tuple_encoder``
emits 8 bytes (count=0 + 7 padding) for the same shape. The
decoder accepts both forms; ``encode_params_tuple(emit_empty_header=
True)`` produces the C-style 8-byte empty-params body for callers
that need byte-identity round-trip with C-origin captures.

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
"""

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

from dqlitewire.buffer import ReadBuffer, WriteBuffer
from dqlitewire.codec import MessageDecoder, MessageEncoder, decode_message, encode_message
from dqlitewire.constants import (
    DEFAULT_MAX_CONTINUATION_FRAMES,
    DEFAULT_MAX_TOTAL_ROWS,
    DQLITE_NOTFOUND,
    DQLITE_PARSE,
    DQLITE_PROTO,
    LEADER_ERROR_CODES,
    NO_TRANSACTION_MESSAGE_SUBSTRINGS,
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
    SQLITE_FORMAT,
    SQLITE_FULL,
    SQLITE_INTERRUPT,
    SQLITE_IOERR,
    SQLITE_IOERR_LEADERSHIP_LOST,
    SQLITE_IOERR_NOT_LEADER,
    SQLITE_NOMEM,
    SQLITE_NOTADB,
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
from dqlitewire import messages
from dqlitewire.messages.base import Header, Message
from dqlitewire.types import WireInput, WireValue

__version__: Final[str] = "0.1.3"

__all__ = [
    "DEFAULT_MAX_CONTINUATION_FRAMES",
    "DEFAULT_MAX_TOTAL_ROWS",
    "DQLITE_NOTFOUND",
    "DQLITE_PARSE",
    "DQLITE_PROTO",
    "LEADER_ERROR_CODES",
    "NO_TRANSACTION_MESSAGE_SUBSTRINGS",
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
    "SQLITE_FORMAT",
    "SQLITE_FULL",
    "SQLITE_INTERRUPT",
    "SQLITE_IOERR",
    "SQLITE_IOERR_LEADERSHIP_LOST",
    "SQLITE_IOERR_NOT_LEADER",
    "SQLITE_NOMEM",
    "SQLITE_NOTADB",
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
    "messages",
    "primary_sqlite_code",
]
