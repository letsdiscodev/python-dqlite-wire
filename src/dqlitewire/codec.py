"""Message encoder and decoder for dqlite wire protocol."""

from typing import NoReturn

from dqlitewire.buffer import ReadBuffer
from dqlitewire.constants import (
    DEFAULT_MAX_CONTINUATION_FRAMES,
    DEFAULT_MAX_TOTAL_ROWS,
    HEADER_SIZE,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_LEGACY,
    RequestType,
    ResponseType,
)
from dqlitewire.exceptions import (
    ContinuationError,
    DecodeError,
    HandshakeError,
    ServerFailure,
    StreamError,
)
from dqlitewire.messages.base import Header, Message
from dqlitewire.messages.requests import (
    AddRequest,
    AssignRequest,
    ClientRequest,
    ClusterRequest,
    DescribeRequest,
    DumpRequest,
    ExecRequest,
    ExecSqlRequest,
    FinalizeRequest,
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

# Mapping from type codes to message classes.
#
# ``RequestType.HEARTBEAT`` and ``RequestType.CONNECT`` are intentionally
# absent: upstream C's ``REQUEST__TYPES`` (``request.h``) omits both and
# ``gateway.c`` falls through to ``DQLITE_PARSE`` for type-2 and type-11
# frames, so no real gateway accepts them. HEARTBEAT is a transport
# ping that Go/C clients never send; CONNECT is a Raft-transport frame
# used only for inter-node connections, never for client-gateway
# traffic. A type-2 or type-11 request decodes to the unknown-type
# error path here, matching upstream's reject. The historical classes
# live on as the private ``_HeartbeatRequest`` and ``_ConnectRequest``
# in ``messages.requests`` for test-mock / golden-byte harnesses that
# synthesize the frames.
REQUEST_TYPES: dict[int, type[Message]] = {
    RequestType.LEADER: LeaderRequest,
    RequestType.CLIENT: ClientRequest,
    RequestType.OPEN: OpenRequest,
    RequestType.PREPARE: PrepareRequest,
    RequestType.EXEC: ExecRequest,
    RequestType.QUERY: QueryRequest,
    RequestType.FINALIZE: FinalizeRequest,
    RequestType.EXEC_SQL: ExecSqlRequest,
    RequestType.QUERY_SQL: QuerySqlRequest,
    RequestType.INTERRUPT: InterruptRequest,
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


# Maximum supported schema version per message type (default is 0).
#
# Keyed separately by direction because ``RequestType`` and ``ResponseType``
# share numeric codes (e.g. ``RequestType.QUERY=6`` collides with
# ``ResponseType.RESULT=6``). A single-dict lookup would conflate the two —
# a hostile or buggy server could then emit, say, a ``FILES`` response
# (code 9) with ``schema=1`` and slip through a ceiling meant for
# ``QUERY_SQL`` (also code 9). The ``MessageDecoder`` selects the
# appropriate dict using ``is_request`` at construction time.
_REQUEST_MAX_SCHEMA: dict[int, int] = {
    RequestType.PREPARE: 1,
    RequestType.EXEC: 1,
    RequestType.QUERY: 1,
    RequestType.EXEC_SQL: 1,
    RequestType.QUERY_SQL: 1,
}
_RESPONSE_MAX_SCHEMA: dict[int, int] = {
    ResponseType.STMT: 1,
}


_SUPPORTED_VERSIONS = frozenset({PROTOCOL_VERSION, PROTOCOL_VERSION_LEGACY})


class MessageEncoder:
    """Encodes messages to wire protocol format.

    Thread-safety: NOT thread-safe. A single ``MessageEncoder``
    instance must be owned by one thread (or one asyncio coroutine)
    at a time. The encoder is effectively stateless after
    construction (it only caches a protocol ``_version``), but the
    single-owner contract matches the rest of the package — see
    the class docstring on ``MessageDecoder`` / ``ReadBuffer``.
    """

    def __reduce__(self) -> NoReturn:
        # The class is "effectively stateless after construction" per
        # the docstring above (it only caches a protocol ``_version``
        # int). The rejection is preserved as a wire-package single-
        # owner-discipline structural pin (forward-compat if the
        # encoder ever grows per-stream state); the message is
        # rewritten to align with the docstring rather than claim a
        # non-existent connection binding. ``type(self).__name__`` so
        # subclasses inherit the right name.
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — wire-"
            f"package single-owner discipline; share by re-creating "
            f"in the target process."
        )

    def __init__(self, version: int = PROTOCOL_VERSION) -> None:
        """Initialize encoder.

        Args:
            version: Protocol version to use in handshake. Defaults to
                     PROTOCOL_VERSION (1). Use PROTOCOL_VERSION_LEGACY
                     (0x86104dd760433fe5) for pre-1.0 dqlite servers.

        Note: ``version`` is used for the handshake bytes only.
        Body shape is the caller's responsibility — ``encode(message)``
        always invokes the modern ``message.encode()``. To emit a
        legacy-shape ``LeaderResponse``, call
        ``LeaderResponse.encode_body_legacy()`` directly. Asymmetric
        with ``MessageDecoder`` which auto-dispatches body shape on
        ``self._version``; the asymmetry is documented and intentional
        (mock-server / proxy authors are the only callers needing
        legacy emission).
        """
        if version not in _SUPPORTED_VERSIONS:
            raise HandshakeError(
                f"Unsupported protocol version: {version:#x}. "
                f"Supported: {', '.join(f'{v:#x}' for v in sorted(_SUPPORTED_VERSIONS))}"
            )
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
    ``driver.Conn`` layer in go-dqlite.

    Concurrent misuse from multiple threads produces **silent data
    corruption**, not exceptions. The underlying ``ReadBuffer``
    suffers from lost-update races on ``_pos`` and torn
    ``_data``/``_pos`` snapshots across ``_maybe_compact()``
    calls; these produce valid-looking byte slices that decode
    cleanly to wrong (or duplicated) messages. Fuzz testing
    confirms this reliably on every trial.

    The ``is_poisoned`` flag does NOT detect concurrent misuse.
    Poison is designed to catch single-owner torn state from
    interrupted signal delivery. It cannot observe lost-update
    races or torn reads that produce valid-looking output. If you
    need concurrent access, wrap every call site in an
    ``asyncio.Lock`` or ``threading.Lock`` at the layer that owns
    the socket.
    """

    def __reduce__(self) -> NoReturn:
        # The class owns a ``ReadBuffer`` (with per-stream-position
        # state) plus continuation counters and a version int — there
        # is no socket or connection reference. The rejection is
        # policy-correct (per-stream-position state under single-owner
        # discipline; sharing across processes makes no sense because
        # consumers would have a divergent view of the stream cursor)
        # but the message describes the actual reason rather than
        # inventing a connection binding. ``type(self).__name__`` so
        # subclasses inherit the right name.
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — instances "
            f"own a ReadBuffer with per-stream-position state and "
            f"continuation counters under a single-owner discipline; "
            f"share by re-creating in the target process."
        )

    def __init__(
        self,
        is_request: bool = False,
        version: int = PROTOCOL_VERSION,
        max_message_size: int = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE,
        max_rows: int = RowsResponse.DEFAULT_MAX_ROWS,
        max_continuation_frames: int | None = DEFAULT_MAX_CONTINUATION_FRAMES,
        max_total_rows: int | None = DEFAULT_MAX_TOTAL_ROWS,
        unknown_role_policy: str = "reject",
    ) -> None:
        """Initialize decoder.

        Args:
            is_request: If True, decode as request messages.
                       If False (default), decode as response messages.
            version: Protocol version to assume for client-side decoders.
                    Defaults to PROTOCOL_VERSION (1). Use PROTOCOL_VERSION_LEGACY
                    for pre-1.0 dqlite servers (affects LeaderResponse format).
                    Ignored for request decoders (version comes from handshake).
            max_message_size: Maximum allowed message size in bytes.
                    Defaults to 64 MiB. Messages exceeding this limit are
                    rejected with DecodeError.
            max_rows: Maximum number of rows permitted in a single
                    ``RowsResponse`` frame (including continuation frames).
                    Defaults to ``RowsResponse.DEFAULT_MAX_ROWS``. The cap
                    is exclusive: a frame whose row count reaches
                    ``max_rows`` raises ``DecodeError``. To permit up to N
                    rows in one frame, set ``max_rows = N + 1``.
            max_continuation_frames: Maximum number of continuation
                    frames permitted in a single ROWS stream. A
                    slow-dripping server emitting many 1-row frames
                    can pin the client on Python decode work; the cap
                    bounds total CPU. Defaults to
                    :data:`DEFAULT_MAX_CONTINUATION_FRAMES`. ``None``
                    disables the cap (operator opt-out for trusted
                    deployments).
            max_total_rows: Cumulative row cap across continuation
                    frames for a single ROWS stream. Bounds total
                    memory irrespective of per-frame size. Defaults
                    to :data:`DEFAULT_MAX_TOTAL_ROWS`. ``None``
                    disables the cap.
            unknown_role_policy: Policy for ``ServersResponse``
                    decoded entries whose ``role`` byte is outside
                    the known {VOTER=0, STANDBY=1, SPARE=2} set.
                    Forwarded to ``ServersResponse.decode_body``;
                    accepted values are ``"reject"`` (default,
                    raise ``DecodeError``), ``"warn"`` (substitute
                    ``NodeRole.SPARE`` and emit ``logger.warning``),
                    and ``"accept"`` (substitute silently). The
                    knob is reachable through the streaming
                    ``MessageDecoder`` so production cluster-info
                    consumers can opt into forward-compat tolerance
                    without bypassing the decoder.
        """
        if not is_request and version not in _SUPPORTED_VERSIONS:
            raise HandshakeError(
                f"Unsupported protocol version: {version:#x}. "
                f"Supported: {', '.join(f'{v:#x}' for v in sorted(_SUPPORTED_VERSIONS))}"
            )
        if max_rows < 1:
            raise ValueError(f"max_rows must be >= 1, got {max_rows}")
        if max_continuation_frames is not None and max_continuation_frames < 1:
            raise ValueError(
                f"max_continuation_frames must be >= 1 or None, got {max_continuation_frames}"
            )
        if max_total_rows is not None and max_total_rows < 1:
            raise ValueError(f"max_total_rows must be >= 1 or None, got {max_total_rows}")
        if unknown_role_policy not in ("reject", "warn", "accept"):
            raise ValueError(
                f"unknown_role_policy must be one of 'reject', 'warn', 'accept'; "
                f"got {unknown_role_policy!r}"
            )
        self._unknown_role_policy = unknown_role_policy
        self._buffer = ReadBuffer(max_message_size=max_message_size)
        self._is_request = is_request
        self._type_map = REQUEST_TYPES if is_request else RESPONSE_TYPES
        self._max_schema = _REQUEST_MAX_SCHEMA if is_request else _RESPONSE_MAX_SCHEMA
        # For client-side decoders, version is set from the constructor parameter.
        # For server-side decoders, version is set by decode_handshake().
        self._version: int | None = version if not is_request else None
        # Request decoders (server-side) must receive the protocol version
        # handshake before decoding any messages. Response decoders (client-side)
        # don't receive an inbound handshake, so they skip this check.
        self._handshake_done = not is_request
        self._continuation_expected = False
        self._max_rows = max_rows
        self._max_continuation_frames: int | None = max_continuation_frames
        self._max_total_rows: int | None = max_total_rows
        # Per-stream counters reset whenever ``_continuation_expected``
        # transitions to True via the initial ROWS frame in ``decode``.
        self._continuation_frame_count = 0
        self._continuation_total_rows = 0
        # Column count snapshotted on the initial frame. Continuation
        # frames must agree — a server emitting different column_count
        # values mid-stream is corrupt; without this check the decoder
        # would silently truncate (column_count=0) or overrun
        # (column_count > original).
        self._continuation_column_count: int | None = None
        # Cross-frame column-name consistency: identical to the count
        # check above, but on the names tuple. The user-facing
        # ``column_names`` accessor reports the initial frame's names
        # only (the protocol-layer drain merges only ``rows``); a
        # buggy / hostile peer that holds count constant but rotates
        # the names list per continuation frame would silently
        # mis-attribute the per-row data the rest of the way. Matches
        # the strict-decode posture of column-count drift, BOOLEAN
        # narrowing, NULL-zero distinction, and Header.reserved == 0.
        self._continuation_column_names: tuple[str, ...] | None = None

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
        self._finalize_continuation_state()

    def _finalize_continuation_state(self) -> None:
        """Reset all per-stream continuation counters and flags.

        Called from every termination arm of ``decode_continuation`` —
        clean (FAILURE, EMPTY, wrong-type, has_more=False) and dirty
        (oversized read_message, broad-BaseException catch) — and from
        ``reset()``. Centralising the discipline keeps the asymmetry
        between clean-exit and poison-exit arms from drifting back into
        partial clears, and makes the pattern grep-able for any future
        sixth termination path.

        Counter staleness is functionally inaccessible behind the
        buffer's poison flag (``_check_poisoned`` short-circuits every
        post-poison decoder call), but a future telemetry accessor or a
        partial reset would surface the stale data; the helper closes
        that hazard now.
        """
        self._continuation_expected = False
        self._continuation_frame_count = 0
        self._continuation_total_rows = 0
        self._continuation_column_count = None
        self._continuation_column_names = None
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
            raise ContinuationError(
                "Cannot skip a message while a ROWS continuation is "
                "in progress. Call decode_continuation() to drain, "
                "or reset() to abandon the stream."
            )
        return self._buffer.skip_message()

    @property
    def is_skipping(self) -> bool:
        """True if still discarding bytes from an oversized message."""
        return self._buffer.is_skipping

    def decode_continuation(self) -> RowsResponse | EmptyResponse | None:
        """Decode a ROWS continuation message from the buffer.

        After receiving a RowsResponse with has_more=True, call this
        method to decode each subsequent ROWS frame. The C dqlite server
        sends continuation frames with the same body layout as the
        initial frame (column_count + column_names + rows + marker),
        so this method uses ``RowsResponse.decode_body`` — the same
        decoder used for the initial frame.

        An ``EmptyResponse`` mid-continuation is also a conforming
        terminator: the upstream C server emits
        ``EmptyResponse`` instead of a final ROWS frame when the
        in-flight query was cancelled by an INTERRUPT request
        (``gateway.c::handle_query_done_cb``). The reference Go client
        loops reading continuations until it sees ``ResponseEmpty``.
        Returning the ``EmptyResponse`` here mirrors that behaviour and
        clears the continuation-expected flag so subsequent traffic on
        the connection can resume normally.

        Returns None if no complete message is available.
        Raises ProtocolError if no continuation is in progress
        (use ``decode()`` for the initial message).
        Raises ServerFailure if the server sends a FailureResponse
        instead of a ROWS continuation (e.g., mid-stream I/O error).
        Raises StreamError for any other message type — that indicates
        genuine wire desync and the buffer is poisoned.
        """
        self._buffer._check_poisoned()
        if not self._continuation_expected:
            raise ContinuationError(
                "decode_continuation() called but no ROWS continuation "
                "is in progress. Use decode() for the initial message."
            )
        try:
            data = self._buffer.read_message()
        except DecodeError:
            # Oversized continuation frame — no skip_message() recovery
            # available during continuation mode. Finalise counters and
            # poison so the caller knows reset() is required.
            self._finalize_continuation_state()
            self._buffer.poison(
                DecodeError("Oversized ROWS continuation frame; call reset() to recover")
            )
            raise
        if data is None:
            return None

        try:
            header = Header.decode(data[:HEADER_SIZE])
            body = data[HEADER_SIZE : HEADER_SIZE + header.size_words * 8]

            # Type-recognition check FIRST, then schema cap. Mirrors
            # ``decode_bytes``'s ordering at the bottom of this file.
            # An unknown msg_type combined with schema=1 would
            # otherwise raise "unsupported schema" — misleading; the
            # actual root cause is "unknown type". Recognised types
            # for the continuation path: FAILURE, EMPTY, ROWS.
            if header.msg_type not in (
                ResponseType.FAILURE,
                ResponseType.EMPTY,
                ResponseType.ROWS,
            ):
                self._finalize_continuation_state()
                raise StreamError(
                    f"Expected ROWS continuation (type {ResponseType.ROWS}), "
                    f"got type {header.msg_type}"
                )

            max_schema = self._max_schema.get(header.msg_type, 0)
            if header.schema > max_schema:
                raise DecodeError(
                    f"Unsupported schema version {header.schema} for message type "
                    f"{header.msg_type} (max supported: {max_schema})"
                )

            if header.msg_type == ResponseType.FAILURE:
                # Decode the body BEFORE clearing the flag so a malformed
                # body still goes through the broad ``except`` below and
                # poisons the buffer (genuine wire desync). A well-formed
                # FailureResponse mid-stream is a clean operational signal
                # — clear the flag and re-raise outside the broad except
                # so the buffer offset remains coherent for the next
                # request on the connection.
                failure = FailureResponse.decode_body(body, schema=header.schema)
                self._finalize_continuation_state()
                raise ServerFailure(failure.code, failure.message)
            if header.msg_type == ResponseType.EMPTY:
                # The upstream C server emits an ``EmptyResponse`` when
                # the in-flight query was cancelled by an INTERRUPT
                # request mid-stream. A WELL-FORMED EmptyResponse body
                # (8 zero bytes from a conforming peer) is treated as a
                # clean terminator, mirroring go-dqlite's
                # ``Protocol.Interrupt`` drain loop, and the buffer is
                # NOT poisoned. A malformed EmptyResponse body (wrong
                # length or non-zero reserved field) still poisons the
                # buffer via the standard DecodeError path raised from
                # ``EmptyResponse.decode_body`` — see
                # ``test_decode_continuation_malformed_empty_still_poisons``.
                self._finalize_continuation_state()
                return EmptyResponse.decode_body(body, schema=header.schema)
            # ``msg_type`` is ROWS here (the early type-recognition
            # check above narrowed the universe to FAILURE/EMPTY/ROWS).

            result = RowsResponse.decode_body(body, schema=header.schema, max_rows=self._max_rows)
            # Per-stream caps: bound the number of continuation frames
            # and the cumulative row count. A slow-dripping server
            # emitting many 1-row frames would otherwise pin the
            # client on Python decode work within whatever wall-clock
            # budget the caller's outer deadline allows. The frame
            # cap bounds CPU; the cumulative-row cap bounds memory.
            self._continuation_frame_count += 1
            if (
                self._max_continuation_frames is not None
                and self._continuation_frame_count > self._max_continuation_frames
            ):
                raise DecodeError(
                    f"ROWS continuation exceeded max_continuation_frames cap "
                    f"({self._max_continuation_frames}); server may be "
                    f"slow-dripping rows"
                )
            self._continuation_total_rows += len(result.rows)
            if (
                self._max_total_rows is not None
                and self._continuation_total_rows > self._max_total_rows
            ):
                raise DecodeError(
                    f"ROWS continuation exceeded max_total_rows cap "
                    f"({self._max_total_rows}); cumulative row count "
                    f"{self._continuation_total_rows} across "
                    f"{self._continuation_frame_count} frames"
                )
            # Cross-frame column-count consistency: the initial frame
            # established the schema; subsequent frames MUST agree.
            # A server that drops to column_count=0 mid-stream would
            # silently truncate the result; a server that grows
            # column_count would overrun the row buffer with phantom
            # NULL values. Either is a corrupt-stream signal.
            if (
                self._continuation_column_count is not None
                and len(result.column_names) != self._continuation_column_count
            ):
                raise DecodeError(
                    f"ROWS continuation column count drift: initial frame had "
                    f"{self._continuation_column_count} columns, frame "
                    f"{self._continuation_frame_count} has "
                    f"{len(result.column_names)}"
                )
            # Cross-frame column-name consistency: the user-facing
            # ``column_names`` accessor reports the initial frame's
            # names only (the protocol-layer drain merges only
            # ``rows``). A buggy / hostile peer that holds count
            # constant but rotates the names list per continuation
            # frame would silently mis-attribute the per-row data the
            # rest of the way. Sibling-shaped to the count check
            # above; same strict-decode posture.
            if (
                self._continuation_column_names is not None
                and tuple(result.column_names) != self._continuation_column_names
            ):
                raise DecodeError(
                    f"ROWS continuation column name drift: initial frame had "
                    f"{list(self._continuation_column_names)}, frame "
                    f"{self._continuation_frame_count} has "
                    f"{list(result.column_names)}"
                )
            if not result.has_more:
                self._finalize_continuation_state()
            return result
        except ServerFailure:
            # Clean server-emitted failure with a well-formed body. The
            # buffer offset is at the start of the next response and the
            # flag is cleared above; the connection is still wire-coherent
            # so subsequent requests (after the user handles the
            # OperationalError) decode normally. Do NOT poison.
            raise
        except BaseException as e:
            # Counter-finalise BEFORE poisoning so the discipline is
            # symmetric with the clean-exit arms — staleness is
            # functionally inaccessible behind the poison flag, but a
            # future telemetry accessor or partial reset would
            # otherwise surface stale per-stream data.
            self._finalize_continuation_state()
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
            raise ContinuationError(
                "Cannot decode a new message while a ROWS continuation "
                "is in progress. Call decode_continuation() until "
                "has_more is False, or call reset() to abandon the stream."
            )
        if not self._handshake_done:
            raise HandshakeError(
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
        # KeyboardInterrupt also poisons before propagating. `decode_body` implementations can raise
        # struct.error, ValueError, UnicodeDecodeError, IndexError,
        # etc., and all of them mean the stream is desynchronized.
        try:
            msg = self.decode_bytes(data)
            # The flag store MUST be inside the try block: the bytes
            # have been consumed, so any asynchronously-delivered
            # exception here (KeyboardInterrupt, PyErr_SetAsyncExc)
            # between the successful decode and the flag store would
            # otherwise leave the stream desynchronized without
            # poisoning — the next ``decode()`` would mis-frame the
            # continuation frame as a top-level message.
            if isinstance(msg, RowsResponse) and msg.has_more:
                self._continuation_expected = True
                # Initialize per-stream cap counters with the first
                # frame's row count so the cumulative-row check in
                # decode_continuation can compare against the running
                # total.
                self._continuation_frame_count = 1
                self._continuation_total_rows = len(msg.rows)
                # Snapshot column count and names for cross-frame
                # consistency. A continuation frame whose names tuple
                # differs from this snapshot is rejected as a corrupt
                # stream — see decode_continuation.
                self._continuation_column_count = len(msg.column_names)
                self._continuation_column_names = tuple(msg.column_names)
                if (
                    self._max_total_rows is not None
                    and self._continuation_total_rows > self._max_total_rows
                ):
                    raise DecodeError(
                        f"ROWS initial frame already exceeded max_total_rows cap "
                        f"({self._max_total_rows}); first frame had "
                        f"{self._continuation_total_rows} rows"
                    )
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

        return msg

    def decode_bytes(self, data: bytes | bytearray | memoryview) -> Message:
        """Decode a message from bytes-like input.

        Raises ProtocolError if the decoder is poisoned.
        Raises ProtocolError if called on a request decoder before decode_handshake().
        """
        self._buffer._check_poisoned()
        if not self._handshake_done:
            raise HandshakeError(
                "Protocol handshake not yet received. "
                "Call decode_handshake() before decode_bytes()."
            )
        # Materialise once so the per-message decoders that take
        # ``bytes`` can rely on the type. The widened input shape is
        # for the convenience of zero-copy callers; the per-message
        # decoders still own a ``bytes``-typed contract.
        if not isinstance(data, bytes):
            data = bytes(data)
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

        max_schema = self._max_schema.get(header.msg_type, 0)
        if header.schema > max_schema:
            raise DecodeError(
                f"Unsupported schema version {header.schema} for message type "
                f"{header.msg_type} (max supported: {max_schema})"
            )

        # LeaderResponse has a version-dependent format: legacy servers
        # send only text address (no node_id prefix). The selector is
        # caller-locked via the constructor-time ``self._version``;
        # there is no auto-fallback on the inbound bytes themselves.
        # A misaligned encoder/decoder version (one of them at
        # ``PROTOCOL_VERSION_LEGACY``, the other at the modern
        # default) silently misdecodes the LEADER body — the legacy
        # text address's first 8 bytes are interpreted as a uint64
        # ``node_id`` and the address-text decode resumes past the
        # misaligned cursor, both fields garbage. Callers must
        # construct ``MessageDecoder`` with the same ``version``
        # negotiated on the wire.
        if (
            header.msg_type == ResponseType.LEADER
            and self._version == PROTOCOL_VERSION_LEGACY
            and msg_class is LeaderResponse
        ):
            return LeaderResponse.decode_body_legacy(body)

        # RowsResponse takes an extra ``max_rows`` cap; ServersResponse
        # takes ``unknown_role_policy`` for forward-compat tolerance of
        # role bytes outside {0,1,2}; other classes share the generic
        # (body, schema) signature.
        if msg_class is RowsResponse:
            return RowsResponse.decode_body(body, schema=header.schema, max_rows=self._max_rows)
        if msg_class is ServersResponse:
            return ServersResponse.decode_body(
                body,
                schema=header.schema,
                unknown_role_policy=self._unknown_role_policy,
            )

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

        Signal-safety: the commit order is
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
            raise HandshakeError("Handshake already completed")
        # Peek first so we only commit on a valid version. An invalid version
        # leaves the bytes in place — a retry is deterministic rather than
        # silently advancing into real message data.
        peek = self._buffer.peek_bytes(8)
        if peek is None:
            return None
        version = int.from_bytes(peek, "little")
        if version not in _SUPPORTED_VERSIONS:
            raise HandshakeError(f"Unsupported protocol version: {version:#x}")
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
    data: bytes | bytearray | memoryview,
    is_request: bool = False,
    version: int = PROTOCOL_VERSION,
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
