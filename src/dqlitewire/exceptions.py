"""Exceptions for dqlite wire protocol.

All exceptions inherit from ``ProtocolError`` so callers can keep a
single broad ``except ProtocolError`` catch. Callers who need to
distinguish recoverable server errors from fatal stream desync can
catch the more specific subclasses below.
"""


class ProtocolError(Exception):
    """Base exception for protocol errors."""


class EncodeError(ProtocolError):
    """Error during message encoding."""


class DecodeError(ProtocolError):
    """Error during message decoding."""


class StreamError(ProtocolError):
    """The stream is at an unknown offset; the connection must be rebuilt.

    Callers catching this should close and re-establish the underlying
    socket. The decoder and buffer are not recoverable — any remaining
    buffered bytes are at an unknown protocol offset.
    """


class PoisonedError(StreamError):
    """Raised by ``ReadBuffer._check_poisoned`` when the buffer was
    previously poisoned by a decode failure or signal interruption.

    The original exception that caused the poisoning is available via
    ``__cause__``.
    """


class HandshakeError(ProtocolError):
    """The handshake has not been completed, has been completed with
    an unsupported version, or an attempt was made to re-perform an
    already-completed handshake. Caller should call ``decode_handshake()``
    before attempting to decode messages, and should only call it once.
    """


class ContinuationError(ProtocolError):
    """The ROWS continuation state machine was used incorrectly — e.g.
    ``decode()`` was called while a multipart ``RowsResponse`` was in
    progress, or ``decode_continuation()`` was called when no
    continuation was pending.

    Recoverable: call the correct method (``decode_continuation()`` or
    ``reset()``) and proceed.
    """


class ServerFailure(ProtocolError):
    """The peer returned a FAILURE response.

    Semantically a server-side error (e.g. SQL constraint violation,
    busy database, or the dqlite-specific "not leader" response) rather
    than a wire-level corruption. The SQLite error code is exposed
    programmatically so callers can distinguish recoverable conditions
    (like ``SQLITE_IOERR_NOT_LEADER``, which callers may handle by
    redirecting to the cluster leader) from terminal ones.

    Connection state: in the normal ``decode()`` path the server
    returns a ``FailureResponse`` *object* (not an exception), so no
    state is disturbed. This exception is raised only from
    ``decode_continuation()`` when FAILURE arrives mid-stream during
    a multi-part ``RowsResponse``. In that path the buffer is also
    poisoned: the remaining continuation frames the
    caller was expecting will never arrive, and the next decode must
    start fresh. Callers catching ``ServerFailure`` in that path
    should treat the decoder as requiring ``reset()`` and reconnect.

    Attributes:
        code: SQLite error code (or extended dqlite code). Common codes
            include ``SQLITE_BUSY`` (5), ``SQLITE_ERROR`` (1), and the
            dqlite-specific ``SQLITE_IOERR_NOT_LEADER``. See the SQLite
            documentation for the full list.
        message: Human-readable error message from the server.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
