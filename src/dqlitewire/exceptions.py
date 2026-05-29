"""Exceptions for dqlite wire protocol; all inherit from ``ProtocolError``."""

__all__ = [
    "ContinuationError",
    "DecodeError",
    "EncodeError",
    "HandshakeError",
    "PoisonedError",
    "ProtocolError",
    "ServerFailure",
    "StreamError",
]


class ProtocolError(Exception):
    """Base wire-layer protocol exception.

    Sibling packages subclass this via multiple inheritance, so catching
    ``dqlitewire.ProtocolError`` also catches the client-layer subclass.
    """


class EncodeError(ProtocolError):
    pass


class DecodeError(ProtocolError):
    pass


class StreamError(ProtocolError):
    """The stream is at an unknown offset; the connection must be rebuilt.

    Exception: the wrong-mid-stream-type variant from
    ``decode_continuation`` does NOT poison the buffer — the caller can
    resync, and ``frame_count`` / ``total_rows`` carry the partial
    accumulation (both 0 when raised outside the continuation drain).
    """

    frame_count: int
    total_rows: int

    def __init__(self, msg: str, *, frame_count: int = 0, total_rows: int = 0) -> None:
        super().__init__(msg)
        self.frame_count = frame_count
        self.total_rows = total_rows


class PoisonedError(StreamError):
    """Buffer was previously poisoned by a decode failure or signal
    interruption; the original cause is available via ``__cause__``.
    """


class HandshakeError(ProtocolError):
    """Handshake missing, unsupported version, or re-performed. Call
    ``decode_handshake()`` once before decoding any messages.
    """


class ContinuationError(ProtocolError):
    """The ROWS continuation state machine was used incorrectly.

    Recoverable: call the correct method (``decode_continuation()`` or
    ``reset()``) and proceed.
    """


class ServerFailure(ProtocolError):
    """The peer returned a FAILURE response (server-side error, not wire
    corruption). Raised only from ``decode_continuation()`` when FAILURE
    arrives mid-stream; the normal ``decode()`` path returns a
    ``FailureResponse`` object instead.

    A well-formed FAILURE leaves the wire coherent — treat as a normal
    operational error, no ``reset()`` needed. A malformed body poisons
    the buffer via ``DecodeError`` and requires ``reset()`` + reconnect.
    """

    code: int
    message: str

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        # Pass as separate args so ``self.args == (code, message)``;
        # otherwise pickle/deepcopy reconstruct via ``ServerFailure(*args)``
        # with one positional arg and raise ``TypeError``.
        super().__init__(code, message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
