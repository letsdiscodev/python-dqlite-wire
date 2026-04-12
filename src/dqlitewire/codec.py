"""Message encoder and decoder for dqlite wire protocol."""

from dqlitewire.buffer import ReadBuffer
from dqlitewire.constants import (
    HEADER_SIZE,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_LEGACY,
    RequestType,
    ResponseType,
)
from dqlitewire.exceptions import DecodeError, ProtocolError
from dqlitewire.messages.base import Header, Message
from dqlitewire.messages.requests import (
    AddRequest,
    AssignRequest,
    ClientRequest,
    ClusterRequest,
    ConnectRequest,
    DescribeRequest,
    DumpRequest,
    ExecRequest,
    ExecSqlRequest,
    FinalizeRequest,
    HeartbeatRequest,
    InterruptRequest,
    LeaderRequest,
    OpenRequest,
    PrepareRequest,
    QueryRequest,
    QuerySqlRequest,
    RemoveRequest,
    TransferRequest,
    WeightRequest,
)
from dqlitewire.messages.responses import (
    DbResponse,
    EmptyResponse,
    FailureResponse,
    FilesResponse,
    LeaderResponse,
    MetadataResponse,
    ResultResponse,
    RowsResponse,
    ServersResponse,
    StmtResponse,
    WelcomeResponse,
)

# Mapping from type codes to message classes
REQUEST_TYPES: dict[int, type[Message]] = {
    RequestType.LEADER: LeaderRequest,
    RequestType.CLIENT: ClientRequest,
    RequestType.HEARTBEAT: HeartbeatRequest,
    RequestType.OPEN: OpenRequest,
    RequestType.PREPARE: PrepareRequest,
    RequestType.EXEC: ExecRequest,
    RequestType.QUERY: QueryRequest,
    RequestType.FINALIZE: FinalizeRequest,
    RequestType.EXEC_SQL: ExecSqlRequest,
    RequestType.QUERY_SQL: QuerySqlRequest,
    RequestType.INTERRUPT: InterruptRequest,
    RequestType.CONNECT: ConnectRequest,
    RequestType.ADD: AddRequest,
    RequestType.ASSIGN: AssignRequest,
    RequestType.REMOVE: RemoveRequest,
    RequestType.DUMP: DumpRequest,
    RequestType.CLUSTER: ClusterRequest,
    RequestType.TRANSFER: TransferRequest,
    RequestType.DESCRIBE: DescribeRequest,
    RequestType.WEIGHT: WeightRequest,
}

RESPONSE_TYPES: dict[int, type[Message]] = {
    ResponseType.FAILURE: FailureResponse,
    ResponseType.LEADER: LeaderResponse,
    ResponseType.WELCOME: WelcomeResponse,
    ResponseType.SERVERS: ServersResponse,
    ResponseType.DB: DbResponse,
    ResponseType.STMT: StmtResponse,
    ResponseType.RESULT: ResultResponse,
    ResponseType.ROWS: RowsResponse,
    ResponseType.EMPTY: EmptyResponse,
    ResponseType.FILES: FilesResponse,
    ResponseType.METADATA: MetadataResponse,
}


# Maximum supported schema version per message type (default is 0)
_MAX_SCHEMA: dict[int, int] = {
    RequestType.PREPARE: 1,
    RequestType.EXEC: 1,
    RequestType.QUERY: 1,
    RequestType.EXEC_SQL: 1,
    RequestType.QUERY_SQL: 1,
    ResponseType.STMT: 1,
}


_SUPPORTED_VERSIONS = {PROTOCOL_VERSION, PROTOCOL_VERSION_LEGACY}


class MessageEncoder:
    """Encodes messages to wire protocol format.

    Thread-safety: NOT thread-safe. A single ``MessageEncoder``
    instance must be owned by one thread (or one asyncio coroutine)
    at a time. The encoder is effectively stateless after
    construction (it only caches a protocol ``_version``), but the
    single-owner contract matches the rest of the package — see
    issue 021 and the class docstring on ``MessageDecoder`` /
    ``ReadBuffer``.
    """

    def __init__(self, version: int = PROTOCOL_VERSION) -> None:
        """Initialize encoder.

        Args:
            version: Protocol version to use in handshake. Defaults to
                     PROTOCOL_VERSION (1). Use PROTOCOL_VERSION_LEGACY
                     (0x86104dd760433fe5) for pre-1.0 dqlite servers.
        """
        self._version = version

    def encode(self, message: Message) -> bytes:
        """Encode a message to bytes."""
        return message.encode()

    def encode_handshake(self) -> bytes:
        """Encode the protocol version handshake.

        Must be sent before any other message.
        """
        return self._version.to_bytes(8, "little")


class MessageDecoder:
    """Decodes messages from wire protocol format.

    Thread-safety: NOT thread-safe. A single ``MessageDecoder``
    instance must be owned by one thread (or one asyncio coroutine)
    at a time. The single-owner contract matches Go's
    ``driver.Conn`` layer in go-dqlite; see issue 021 for the full
    analysis.

    Concurrent misuse from multiple threads produces **silent data
    corruption**, not exceptions. The underlying ``ReadBuffer``
    suffers from lost-update races on ``_pos`` and torn
    ``_data``/``_pos`` snapshots across ``_maybe_compact()``
    calls; these produce valid-looking byte slices that decode
    cleanly to wrong (or duplicated) messages. Fuzz testing
    (issue 050) confirms this reliably on every trial.

    The ``is_poisoned`` flag does NOT detect concurrent misuse.
    Poison is designed to catch single-owner torn state from
    interrupted signal delivery (see issues 037, 041, 045). It
    cannot observe lost-update races or torn reads that produce
    valid-looking output. If you need concurrent access, wrap every
    call site in an ``asyncio.Lock`` or ``threading.Lock`` at the
    layer that owns the socket.
    """

    def __init__(self, is_request: bool = False, version: int = PROTOCOL_VERSION) -> None:
        """Initialize decoder.

        Args:
            is_request: If True, decode as request messages.
                       If False (default), decode as response messages.
            version: Protocol version to assume for client-side decoders.
                    Defaults to PROTOCOL_VERSION (1). Use PROTOCOL_VERSION_LEGACY
                    for pre-1.0 dqlite servers (affects LeaderResponse format).
                    Ignored for request decoders (version comes from handshake).
        """
        self._buffer = ReadBuffer()
        self._is_request = is_request
        self._type_map = REQUEST_TYPES if is_request else RESPONSE_TYPES
        # For client-side decoders, version is set from the constructor parameter.
        # For server-side decoders, version is set by decode_handshake().
        self._version: int | None = version if not is_request else None
        # Request decoders (server-side) must receive the protocol version
        # handshake before decoding any messages. Response decoders (client-side)
        # don't receive an inbound handshake, so they skip this check.
        self._handshake_done = not is_request
        self._continuation_expected = False

    @property
    def version(self) -> int | None:
        """Protocol version from handshake, or None if not yet received."""
        return self._version

    @property
    def is_poisoned(self) -> bool:
        """True if the decoder has hit an unrecoverable mid-stream error.

        Once poisoned, every ``decode*`` method raises ``ProtocolError`` until
        ``reset()`` is called. This protects callers from silently continuing
        to decode from an unknown offset after a parse failure desynchronized
        the stream.
        """
        return self._buffer.is_poisoned

    def reset(self) -> None:
        """Reset decoder state to a fresh, un-poisoned condition.

        Clears the read buffer and, for server-side decoders, returns the
        handshake state to "not yet received". Use this after a reconnect.
        """
        self._buffer.reset()
        self._continuation_expected = False
        if self._is_request:
            self._handshake_done = False
            self._version = None

    def feed(self, data: bytes) -> None:
        """Feed data to the decoder."""
        self._buffer.feed(data)

    def has_message(self) -> bool:
        """Check if a complete message is available."""
        return self._buffer.has_message()

    def skip_message(self) -> bool:
        """Skip the current message in the buffer.

        Use this to recover after has_message() or decode() raises
        DecodeError for an oversized message. For oversized messages,
        returns False until the full message has been discarded
        (check is_skipping). For normal-sized messages, waits until
        the full message is available before skipping.

        Raises ProtocolError if a ROWS continuation is in progress —
        use ``decode_continuation()`` to drain, or ``reset()`` to
        abandon the stream.
        """
        if self._continuation_expected:
            raise ProtocolError(
                "Cannot skip a message while a ROWS continuation is "
                "in progress. Call decode_continuation() to drain, "
                "or reset() to abandon the stream."
            )
        return self._buffer.skip_message()

    @property
    def is_skipping(self) -> bool:
        """True if still discarding bytes from an oversized message."""
        return self._buffer.is_skipping

    def decode_continuation(self) -> RowsResponse | None:
        """Decode a ROWS continuation message from the buffer.

        After receiving a RowsResponse with has_more=True, call this
        method to decode each subsequent ROWS frame. The C dqlite server
        sends continuation frames with the same body layout as the
        initial frame (column_count + column_names + rows + marker),
        so this method uses ``RowsResponse.decode_body`` — the same
        decoder used for the initial frame.

        Returns None if no complete message is available.
        Raises ProtocolError if no continuation is in progress
        (use ``decode()`` for the initial message).
        Raises ProtocolError if the server sends a FailureResponse
        instead of a ROWS continuation (e.g., mid-stream I/O error).
        """
        self._buffer._check_poisoned()
        if not self._continuation_expected:
            raise ProtocolError(
                "decode_continuation() called but no ROWS continuation "
                "is in progress. Use decode() for the initial message."
            )
        data = self._buffer.read_message()
        if data is None:
            return None

        try:
            header = Header.decode(data[:HEADER_SIZE])
            body = data[HEADER_SIZE : HEADER_SIZE + header.size_words * 8]

            if header.msg_type == ResponseType.FAILURE:
                failure = FailureResponse.decode_body(body, schema=header.schema)
                raise ProtocolError(
                    f"Server error during ROWS continuation: [{failure.code}] {failure.message}"
                )
            if header.msg_type != ResponseType.ROWS:
                raise ProtocolError(
                    f"Expected ROWS continuation (type {ResponseType.ROWS}), "
                    f"got type {header.msg_type}"
                )

            result = RowsResponse.decode_body(body, schema=header.schema)
            if not result.has_more:
                self._continuation_expected = False
            return result
        except BaseException as e:
            self._buffer.poison(
                e
                if isinstance(e, Exception)
                else RuntimeError(f"decode_continuation interrupted: {type(e).__name__}")
            )
            raise

    def decode(self) -> Message | None:
        """Decode the next message from the buffer.

        Returns None if no complete message is available.
        Raises ProtocolError if called on a request decoder before decode_handshake().
        Raises ProtocolError if the decoder is poisoned.
        Raises ProtocolError if a ROWS continuation is in progress
        (call ``decode_continuation()`` until ``has_more`` is ``False``,
        or ``reset()`` to abandon the stream).
        """
        self._buffer._check_poisoned()
        if self._continuation_expected:
            raise ProtocolError(
                "Cannot decode a new message while a ROWS continuation "
                "is in progress. Call decode_continuation() until "
                "has_more is False, or call reset() to abandon the stream."
            )
        if not self._handshake_done:
            raise ProtocolError(
                "Protocol handshake not yet received. Call decode_handshake() before decode()."
            )
        # read_message may raise DecodeError for an oversized header. That
        # error is recoverable via skip_message() — the bytes have not been
        # consumed — so we deliberately do NOT poison on it.
        data = self._buffer.read_message()
        if data is None:
            return None

        # Bytes have been consumed. ANY failure now leaves the buffer
        # at an unknown offset; poison so subsequent calls fail fast.
        # Catch BaseException so that signal-delivered
        # KeyboardInterrupt (issue 045) also poisons before
        # propagating. `decode_body` implementations can raise
        # struct.error, ValueError, UnicodeDecodeError, IndexError,
        # etc., and all of them mean the stream is desynchronized.
        try:
            msg = self.decode_bytes(data)
        except BaseException as e:
            # poison() stores Exception | None; wrap non-Exception
            # BaseException subclasses so the poison cause is still a
            # real Exception we can inspect.
            self._buffer.poison(
                e
                if isinstance(e, Exception)
                else RuntimeError(f"decode interrupted: {type(e).__name__}")
            )
            raise

        if isinstance(msg, RowsResponse) and msg.has_more:
            self._continuation_expected = True

        return msg

    def decode_bytes(self, data: bytes) -> Message:
        """Decode a message from bytes.

        Raises ProtocolError if the decoder is poisoned.
        Raises ProtocolError if called on a request decoder before decode_handshake().
        """
        self._buffer._check_poisoned()
        if not self._handshake_done:
            raise ProtocolError(
                "Protocol handshake not yet received. "
                "Call decode_handshake() before decode_bytes()."
            )
        if len(data) < HEADER_SIZE:
            raise DecodeError(f"Message too short: {len(data)} bytes")

        header = Header.decode(data[:HEADER_SIZE])
        body_size = header.size_words * 8
        if len(data) < HEADER_SIZE + body_size:
            raise DecodeError(
                f"Message body too short: header says {body_size} bytes, "
                f"got {len(data) - HEADER_SIZE}"
            )
        body = data[HEADER_SIZE : HEADER_SIZE + body_size]

        msg_class = self._type_map.get(header.msg_type)
        if msg_class is None:
            raise DecodeError(f"Unknown message type: {header.msg_type}")

        max_schema = _MAX_SCHEMA.get(header.msg_type, 0)
        if header.schema > max_schema:
            raise DecodeError(
                f"Unsupported schema version {header.schema} for message type "
                f"{header.msg_type} (max supported: {max_schema})"
            )

        # LeaderResponse has a version-dependent format: legacy servers
        # send only text address (no node_id prefix).
        if (
            header.msg_type == ResponseType.LEADER
            and self._version == PROTOCOL_VERSION_LEGACY
            and msg_class is LeaderResponse
        ):
            return LeaderResponse.decode_body_legacy(body)

        return msg_class.decode_body(body, schema=header.schema)

    def decode_handshake(self) -> int | None:
        """Decode protocol version handshake.

        Returns the protocol version or None if not enough data.
        Must be called before decode() on request decoders.
        Raises ProtocolError if the version is not recognized or if
        the handshake was already completed.

        On an unsupported version, the 8 handshake bytes are left in the
        buffer untouched so that a retry is deterministic (same bytes, same
        error) rather than silently consuming the next 8 bytes of real data.

        Signal-safety (issue 041): the commit order is
        ``_version``/``_handshake_done`` FIRST, then ``read_bytes(8)``.
        If an async exception (``KeyboardInterrupt``) lands between the
        state commit and the buffer consume, the except block reverts
        ``_handshake_done`` and ``_version`` so the buffer is still
        coherent: the 8 handshake bytes are still there and a retry
        repeats the peek/validate/commit cycle. This replaces the
        previous "consume then commit" order, which allowed a signal
        to leave the bytes consumed but the state not yet marked —
        retry would then re-peek 8 bytes of real message data as a
        handshake and almost always raise a misleading "Unsupported
        protocol version" error.
        """
        if self._handshake_done:
            raise ProtocolError("Handshake already completed")
        # Peek first so we only commit on a valid version. An invalid version
        # leaves the bytes in place — a retry is deterministic rather than
        # silently advancing into real message data.
        peek = self._buffer.peek_bytes(8)
        if peek is None:
            return None
        version = int.from_bytes(peek, "little")
        if version not in _SUPPORTED_VERSIONS:
            raise ProtocolError(f"Unsupported protocol version: {version:#x}")
        # Commit state BEFORE consuming bytes. If the consume is
        # interrupted by an async exception, revert so the peek/commit
        # pair becomes atomic from the caller's perspective.
        self._version = version
        self._handshake_done = True
        try:
            self._buffer.read_bytes(8)
        except BaseException:
            self._handshake_done = False
            self._version = None
            raise
        return version


def decode_message(
    data: bytes, is_request: bool = False, version: int = PROTOCOL_VERSION
) -> Message:
    """Convenience function to decode a single message.

    This is a stateless decode — it does not enforce protocol handshake
    since there is no connection context. Use MessageDecoder for full
    protocol-aware streaming decode with handshake enforcement.

    Args:
        data: Raw message bytes (header + body).
        is_request: If True, decode as a request message.
        version: Protocol version to assume. Use PROTOCOL_VERSION_LEGACY
                 to decode legacy-format messages (e.g., LeaderResponse
                 without node_id).
    """
    decoder = MessageDecoder(is_request=is_request, version=version)
    if is_request:
        # Request decoders start with handshake_done=False; bypass for
        # stateless single-message decoding.
        decoder._handshake_done = True
        decoder._version = version
    return decoder.decode_bytes(data)


def encode_message(message: Message) -> bytes:
    """Convenience function to encode a single message."""
    return message.encode()
