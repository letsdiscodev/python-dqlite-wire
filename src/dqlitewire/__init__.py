"""Pure Python wire protocol implementation for dqlite."""

import os as _os
import sys as _sys

# Refuse to run under free-threaded CPython (python3.13t / PEP 703).
#
# ReadBuffer and WriteBuffer are backed by bytearray and rely on the GIL's
# C-level atomicity for bytearray.extend(), slicing, and attribute rebinding
# inside _maybe_compact(). Under a free-threaded build, concurrent access
# produces SIGSEGV in the read path and a process-level hang in the write
# path (see issues/033-free-threaded-segfault-in-readbuffer.md for the full
# reproduction and rationale). The single-owner-per-instance contract alone
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
            "process hangs under free-threading. See "
            "issues/033-free-threaded-segfault-in-readbuffer.md. "
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
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_LEGACY,
    ROW_DONE_BYTE,
    ROW_DONE_MARKER,
    ROW_PART_BYTE,
    ROW_PART_MARKER,
    RequestType,
    ResponseType,
    ValueType,
)
from dqlitewire.exceptions import DecodeError, EncodeError, ProtocolError

__all__ = [
    "DecodeError",
    "EncodeError",
    "MessageDecoder",
    "MessageEncoder",
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_LEGACY",
    "ProtocolError",
    "ROW_DONE_BYTE",
    "ROW_DONE_MARKER",
    "ROW_PART_BYTE",
    "ROW_PART_MARKER",
    "ReadBuffer",
    "RequestType",
    "ResponseType",
    "ValueType",
    "WriteBuffer",
    "decode_message",
    "encode_message",
]

__version__ = "0.1.0"
