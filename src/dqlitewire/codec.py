"""Message encoder and decoder for dqlite wire protocol."""

from typing import Final, NoReturn

__all__ = [
    "MessageDecoder",
    "MessageEncoder",
    "decode_message",
    "encode_message",
]

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
    EncodeError,
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
REQUEST_TYPES: Final[dict[int, type[Message]]] = {
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

RESPONSE_TYPES: Final[dict[int, type[Message]]] = {
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
_REQUEST_MAX_SCHEMA: Final[dict[int, int]] = {
    RequestType.PREPARE: 1,
    RequestType.EXEC: 1,
    RequestType.QUERY: 1,
    RequestType.EXEC_SQL: 1,
    RequestType.QUERY_SQL: 1,
}
_RESPONSE_MAX_SCHEMA: Final[dict[int, int]] = {
    # STMT bumped to schema=1 to accept the V1 stmt-id+tail-offset
    # response shape; see :class:`StmtResponse`.
    ResponseType.STMT: 1,
    # ROWS, FAILURE, EMPTY, LEADER, WELCOME, SERVERS, DB, RESULT,
    # FILES, METADATA: schema=0 only as of dqlite upstream commit
    # f30fc9936a2a39674f3ec665217aff9b96e3b286. When upstream bumps a
    # response (most plausibly ROWS for a future row-encoding
    # extension), add ``ResponseType.<NAME>: <max-schema>`` here AND
    # extend the matching ``decode_body`` to dispatch on ``schema``.
    # The sibling ``_REQUEST_MAX_SCHEMA`` above (which records the
    # PREPARE/EXEC/QUERY/EXEC_SQL/QUERY_SQL bumps) is the canonical
    # pattern; mirror it on this side when the time comes.
    #
    # Until then, the ``.get(..., 0)`` fallback in the dispatch sites
    # at ``decode_continuation`` and ``decode_bytes`` rejects any
    # non-zero ``header.schema`` with a poisoning ``DecodeError`` —
    # the deliberate choice today is to fail fast on a forward-compat
    # mismatch rather than silently accept unknown bytes. A future
    # protocol revision that introduces a ROWS schema bump will
    # require lockstep client updates; document the bump here when it
    # happens.
}


_SUPPORTED_VERSIONS: Final[frozenset[int]] = frozenset({PROTOCOL_VERSION, PROTOCOL_VERSION_LEGACY})


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

    def __init__(
        self,
        version: int = PROTOCOL_VERSION,
        max_message_size: int = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE,
    ) -> None:
        """Initialize encoder.

        Args:
            version: Protocol version to use in handshake. Defaults to
                     PROTOCOL_VERSION (1). Use PROTOCOL_VERSION_LEGACY
                     (0x86104dd760433fe5) for pre-1.0 dqlite servers.
            max_message_size: Maximum allowed total frame size in bytes
                     (header + body). Defaults to
                     ``ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE`` (64 MiB),
                     matching the decoder default so a caller using
                     both defaults stays symmetric. A frame exceeding
                     this cap raises ``EncodeError``. Per-field caps
                     (``_MAX_BLOB_SIZE``, ``_MAX_TEXT_VALUE_SIZE``,
                     ``_MAX_FILE_CONTENT_SIZE``, ``_MAX_FILE_COUNT``,
                     ``_MAX_NODE_COUNT``, column count <= 2000)
                     partially bound the envelope, but a composite
                     frame (e.g. a ``FilesResponse`` with many files
                     near the per-file content cap) can still overflow
                     the matching decoder's envelope; the envelope cap
                     here closes that asymmetry. Operators with a
                     genuinely larger cluster (``raft.log.entry_size_max``
                     above 64 MiB) must raise the cap on BOTH sides.

                     Go-parity asymmetry: Go's encoder helpers in
                     ``go-dqlite/internal/protocol/message.go`` do
                     NOT enforce a per-frame cap. Go's cap lives
                     server-side as ``raft.log.entry_size_max`` and
                     applies per-Raft-log-entry rather than per-wire-
                     frame. This Python encoder narrows that posture
                     to match the matching Python decoder default. An
                     operator with a raised
                     ``raft.log.entry_size_max`` must pass the same
                     higher cap to BOTH encoder and decoder — the
                     encoder's per-frame check will otherwise reject
                     large legitimate frames the cluster would
                     accept.

        Raises:
            HandshakeError: If ``version`` is not in
                ``_SUPPORTED_VERSIONS`` (currently
                ``{PROTOCOL_VERSION, PROTOCOL_VERSION_LEGACY}``).
            ValueError: If ``max_message_size`` is less than 1.

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
        if max_message_size < 1:
            raise ValueError(f"max_message_size must be >= 1, got {max_message_size}")
        if max_message_size > ReadBuffer.MAX_MESSAGE_SIZE_CEILING:
            # Defense-in-depth: see ReadBuffer.MAX_MESSAGE_SIZE_CEILING
            # for the rationale (C server's per-frame UINT32_MAX
            # ceiling at dqlite-upstream/src/conn.c:169).
            raise ValueError(
                f"max_message_size must be <= {ReadBuffer.MAX_MESSAGE_SIZE_CEILING} "
                f"(UINT32_MAX bytes; the C server's per-frame ceiling at "
                f"dqlite-upstream/src/conn.c:169 rejects any single frame above this "
                f"bound), got {max_message_size}"
            )
        self._version = version
        self._max_message_size = max_message_size

    def encode(self, message: Message) -> bytes:
        """Encode a message to bytes.

        The resulting frame is checked against ``max_message_size``
        and rejected with ``EncodeError`` if it exceeds the cap. The
        diagnostic mirrors ``MessageDecoder``'s shape so a caller
        catching ``"exceeds maximum"`` sees the same wording on both
        sides of the symmetry.
        """
        frame = message.encode()
        if len(frame) > self._max_message_size:
            # Same diagnostic shape as ``MessageDecoder.decode_bytes``
            # / ``ReadBuffer.read_message`` so a caller catching either
            # side's ``"exceeds maximum"`` reads the same wording.
            raise EncodeError(
                f"Message size {len(frame):#x} bytes exceeds maximum {self._max_message_size}"
            )
        return frame

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
        text_errors: str = "strict",
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
                    rejected with DecodeError. Go-parity asymmetry: Go's
                    decoder (``go-dqlite/internal/protocol/message.go``)
                    has no per-frame cap; the server-side cap lives in
                    ``raft.log.entry_size_max`` and applies per Raft
                    log entry rather than per wire frame. Operators
                    raising ``raft.log.entry_size_max`` must pass the
                    same higher cap to BOTH encoder and decoder to
                    keep round-trip symmetric.
            max_rows: Maximum number of rows permitted in a single
                    ``RowsResponse`` frame (including continuation frames).
                    Defaults to ``RowsResponse.DEFAULT_MAX_ROWS``. The cap
                    is exclusive: a frame whose row count reaches
                    ``max_rows`` raises ``DecodeError``. To permit up to N
                    rows in one frame, set ``max_rows = N + 1``.
            max_continuation_frames: Maximum number of ROWS frames
                    permitted in a single streamed result set,
                    **including the initial frame**. A slow-dripping
                    server emitting many 1-row frames can pin the
                    client on Python decode work; the cap bounds total
                    CPU. A value of N permits at most N-1 continuation
                    frames after the initial one — e.g. ``N=1``
                    rejects the very first continuation. Defaults to
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
            text_errors: UTF-8 decode error policy for per-row
                    TEXT and ISO8601 cells. Forwarded to
                    ``RowsResponse.decode_body`` →
                    ``decode_row_values`` → ``decode_value`` →
                    ``decode_text``. The default ``"strict"`` matches
                    the dqlite wire-spec contract; a legacy / mixed-
                    encoding cluster's single non-UTF-8 cell otherwise
                    fails the entire ``RowsResponse`` decode and
                    poisons the streaming buffer. Permissive modes
                    accepted by :func:`str.decode` (e.g.
                    ``"replace"``, ``"backslashreplace"``,
                    ``"surrogateescape"``) keep the stream alive and
                    mirror Go's ``getString`` shape, but bring caveats
                    documented on :func:`dqlitewire.types.decode_text`
                    — surrogateescape is not round-trippable via
                    :func:`encode_text` and the replacement codepoint
                    bypasses ``sanitize_for_log``.

        Raises:
            HandshakeError: If ``version`` is not in
                ``_SUPPORTED_VERSIONS``. Applied uniformly for both
                request- and response-side constructors; see the
                inline rationale on uniform validation.
            ValueError: If ``max_rows`` is less than 1, or if
                ``max_continuation_frames`` / ``max_total_rows`` is
                less than 1 (when not ``None``), or if
                ``text_errors`` is not a recognised UTF-8 decode
                error handler name.
            DecodeError: If ``unknown_role_policy`` is not one of
                ``"reject"``, ``"warn"``, ``"accept"``. Raised as
                ``DecodeError`` (not ``ValueError``) so callers
                using ``except DecodeError`` catch BOTH the
                construction-time and the deeper decode-time
                validators with a single arm — mirrors the
                ``ServersResponse.decode_body`` validator the
                policy is forwarded to.
        """
        # Validate ``version`` uniformly across both request- and
        # response-side construction. Originally gated on
        # ``not is_request`` because request decoders read the version
        # from the inbound handshake and don't traverse version-tagged
        # paths until then; but the ``decode_message`` helper writes
        # ``decoder._version = version`` unconditionally, and direct
        # ``MessageDecoder(is_request=True, version=...)`` callers can
        # likewise smuggle an unsupported value past validation. The
        # class invariant "no decoder accepts an unsupported version"
        # should hold uniformly so callers can catch a single
        # ``HandshakeError`` regardless of which side they're on.
        if version not in _SUPPORTED_VERSIONS:
            raise HandshakeError(
                f"Unsupported protocol version: {version:#x}. "
                f"Supported: {', '.join(f'{v:#x}' for v in sorted(_SUPPORTED_VERSIONS))}"
            )
        if max_message_size > ReadBuffer.MAX_MESSAGE_SIZE_CEILING:
            # Belt-and-suspenders: the ReadBuffer constructor below
            # also enforces this ceiling; checking here lets the
            # decoder validate the full kwarg set before instantiating
            # any owned objects. See
            # ReadBuffer.MAX_MESSAGE_SIZE_CEILING for the C-server-
            # anchored rationale.
            raise ValueError(
                f"max_message_size must be <= {ReadBuffer.MAX_MESSAGE_SIZE_CEILING} "
                f"(UINT32_MAX bytes; the C server's per-frame ceiling at "
                f"dqlite-upstream/src/conn.c:169 rejects any single frame above this "
                f"bound), got {max_message_size}"
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
            # NOTE: DecodeError (not ValueError, not EncodeError) — mirrors
            # the deeper ``ServersResponse.decode_body`` validator (which
            # raises DecodeError for the same domain). The policy is
            # consumed by a decode path; DecodeError is the right layer.
            # Callers can use a single ``except DecodeError`` for both
            # the construction-time and decode-time validators.
            raise DecodeError(
                f"unknown_role_policy must be one of 'reject', 'warn', 'accept'; "
                f"got {unknown_role_policy!r}"
            )
        # Validate ``text_errors`` against the well-known stdlib error
        # handler names so a typo (``"surrogate_escape"`` for
        # ``"surrogateescape"``) fails fast at construction rather than
        # on first TEXT cell deep inside ``RowsResponse.decode_body``.
        # ValueError matches the validation discipline of
        # ``max_rows`` / ``max_continuation_frames`` above (caller-side
        # configuration error, not a wire-decode failure).
        if text_errors not in (
            "strict",
            "replace",
            "ignore",
            "backslashreplace",
            "surrogateescape",
            "surrogatepass",
            "xmlcharrefreplace",
            "namereplace",
        ):
            raise ValueError(
                f"text_errors must be a recognised UTF-8 decode error handler "
                f"(strict, replace, ignore, backslashreplace, surrogateescape, "
                f"surrogatepass, xmlcharrefreplace, namereplace); got {text_errors!r}"
            )
        self._unknown_role_policy = unknown_role_policy
        self._text_errors = text_errors
        self._buffer = ReadBuffer(max_message_size=max_message_size)
        # Promote ``max_message_size`` to a decoder-level attribute so the
        # stateless ``decode_bytes`` path can enforce the cap without
        # reaching into the buffer's private. The streaming path checks
        # via ``ReadBuffer.read_message``; both checks read the same
        # ceiling, kept in lockstep by storing it once at construction.
        self._max_message_size = max_message_size
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

        Called from every *termination* arm of ``decode_continuation`` —
        clean (FAILURE, EMPTY, wrong-type, has_more=False) and dirty
        (oversized read_message, broad-BaseException catch) — and from
        ``reset()``. Centralising the discipline keeps the asymmetry
        between clean-exit and poison-exit arms from drifting back into
        partial clears, and makes the pattern grep-able for any future
        sixth termination path.

        Not called from the *guard* arms in ``skip_message`` /
        ``decode`` that reject the requested operation mid-continuation
        with ``ContinuationError``: those raise to tell the caller
        "the continuation is still in progress, drain via
        ``decode_continuation`` or abandon via ``reset()``" — they do
        NOT terminate the continuation themselves, so clearing the
        ``_continuation_expected`` flag would silently lose the
        invariant the guard exists to enforce. The counters remain
        readable to a future telemetry accessor but only while the
        flag is still True, which truthfully reflects the stream
        state.

        Counter staleness is functionally inaccessible behind the
        buffer's poison flag (``_check_poisoned`` short-circuits every
        post-poison decoder call), but a future telemetry accessor or a
        partial reset would surface the stale data; the helper closes
        that hazard now for the termination arms.
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
            body = data[HEADER_SIZE : HEADER_SIZE + header.body_size]

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
                # Snapshot the per-stream counters BEFORE finalising so
                # the caller's "retry the query" recovery has access to
                # how many continuation frames and rows were observed
                # before the wrong-type frame arrived. The buffer is
                # NOT poisoned on this arm (precedent set by the prior
                # coherent-offset fix), so these counters are the only
                # surviving record of partial-stream progress.
                snap_frames = self._continuation_frame_count
                snap_rows = self._continuation_total_rows
                self._finalize_continuation_state()
                raise StreamError(
                    f"Expected ROWS continuation (type {ResponseType.ROWS}), "
                    f"got type {header.msg_type}",
                    frame_count=snap_frames,
                    total_rows=snap_rows,
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
                # (8 bytes from a conforming peer) is treated as a
                # clean terminator, mirroring go-dqlite's
                # ``Protocol.Interrupt`` drain loop, and the buffer is
                # NOT poisoned. The reserved uint64 inside the body is
                # permissively read-and-discarded to match Go's
                # ``response.getUint64()``; a non-zero reserved value
                # does NOT poison.
                #
                # Decode body FIRST and finalise AFTER so the order
                # mirrors the FAILURE arm above. A malformed body
                # (wrong length) raises DecodeError which is caught
                # below and re-raised WITHOUT poison — ``read_message``
                # already advanced past the offending frame so the
                # buffer offset is wire-coherent, matching the
                # ``StreamError`` precedent for coherent-offset
                # anomalies.
                empty_result = EmptyResponse.decode_body(body, schema=header.schema)
                self._finalize_continuation_state()
                return empty_result
            # ``msg_type`` is ROWS here (the early type-recognition
            # check above narrowed the universe to FAILURE/EMPTY/ROWS).

            result = RowsResponse.decode_body(
                body, schema=header.schema, max_rows=self._max_rows, text_errors=self._text_errors
            )
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
        except StreamError:
            # Wrong ``msg_type`` mid-continuation. ``read_message``
            # already advanced past the offending frame; the buffer
            # offset is correctly aligned with the next frame
            # boundary and the continuation flag was cleared at the
            # raise site. The connection is wire-coherent — a hostile
            # or buggy peer that sent one unexpected-type frame
            # should not kill the connection. Mirror the
            # ``ServerFailure`` precedent above: do NOT poison.
            # ``StreamError`` propagates so the caller can choose to
            # invalidate or resync (e.g. issue the next request) —
            # Go's ``Protocol.Recv`` has the same non-poisoning
            # behaviour for unexpected types.
            raise
        except DecodeError:
            # Coherent-offset frame whose body decode failed (e.g. a
            # malformed EMPTY body of the wrong length).
            # ``read_message`` already advanced past the offending
            # frame so the buffer offset is wire-coherent; mirror the
            # ``StreamError`` precedent and re-raise WITHOUT poison
            # after finalising continuation state. The caller can
            # resync without dropping the connection.
            self._finalize_continuation_state()
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
                # Check the cap BEFORE writing any per-stream fields.
                # Writing first and raising after would leave the five
                # ``_continuation_*`` fields populated on a poison exit
                # — the helper at ``_finalize_continuation_state``
                # exists to keep that drift contained, but the broad-
                # except below intentionally does NOT call it (handshake
                # state on request-side decoders must not be clobbered
                # on a mid-stream poison). Easier and leaner to not
                # populate in the first place. Pinned by
                # tests/test_decoder_continuation_counter_resets_on_poison.py::
                # test_decode_initial_frame_cap_exceeded_does_not_populate_continuation_state.
                initial_row_count = len(msg.rows)
                if self._max_total_rows is not None and initial_row_count > self._max_total_rows:
                    raise DecodeError(
                        f"ROWS initial frame already exceeded max_total_rows cap "
                        f"({self._max_total_rows}); first frame had "
                        f"{initial_row_count} rows"
                    )
                self._continuation_expected = True
                # Initialize per-stream cap counters with the first
                # frame's row count so the cumulative-row check in
                # decode_continuation can compare against the running
                # total.
                self._continuation_frame_count = 1
                self._continuation_total_rows = initial_row_count
                # Snapshot column count and names for cross-frame
                # consistency. A continuation frame whose names tuple
                # differs from this snapshot is rejected as a corrupt
                # stream — see decode_continuation.
                self._continuation_column_count = len(msg.column_names)
                self._continuation_column_names = tuple(msg.column_names)
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
        body_size = header.body_size
        # Enforce ``max_message_size`` on the stateless path so the
        # cap discipline matches the streaming path's ``ReadBuffer.
        # read_message`` check. Apply AFTER ``Header.decode`` succeeds
        # (so a torn header still produces the standard "too short"
        # diagnostic) and BEFORE the body-slice / per-class dispatch
        # (so the cap fires before any potentially-large allocation).
        # When ``decode()`` delegates here through the streaming path
        # the cap check is benign and redundant — the cost is constant
        # per message.
        total_size = HEADER_SIZE + body_size
        if total_size > self._max_message_size:
            raise DecodeError(
                f"Message size {total_size:#x} bytes exceeds maximum {self._max_message_size}"
            )
        if len(data) < HEADER_SIZE + body_size:
            raise DecodeError(
                f"Message body too short: header says {body_size} bytes, "
                f"got {len(data) - HEADER_SIZE}"
            )
        # Strict-decode parity with every per-message decoder
        # (FailureResponse, LeaderResponse, RowsResponse,
        # ServersResponse, FilesResponse, all request decoders), each
        # of which enforces ``if offset != len(data): raise
        # DecodeError("trailing bytes")`` for its own body. The
        # envelope-level strip previously sliced silently on any input
        # length > ``HEADER_SIZE + body_size``, so a caller passing
        # ``data = header + body + garbage`` got a successful decode
        # with no indication that the input was malformed.
        # The streaming path (``ReadBuffer.read_message``) returns
        # exactly ``HEADER_SIZE + body_size`` bytes so this guard is
        # unreachable through streaming — the cost is borne only by
        # direct callers (mock servers, golden-byte harnesses,
        # fuzzers, packet replay tools) that benefit most from the
        # strict diagnostic.
        if len(data) > total_size:
            raise DecodeError(
                f"Message has {len(data) - total_size} trailing bytes "
                f"after declared body (header says {body_size} bytes "
                f"+ {HEADER_SIZE}-byte header = {total_size}, "
                f"got {len(data)})"
            )
        body = data[HEADER_SIZE:total_size]

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
        # Gate on ``not self._is_request`` as the primary clause: the
        # legacy LEADER dispatch is response-side only. ``RequestType.
        # CLIENT == 1`` collides numerically with
        # ``ResponseType.LEADER == 1``; a request-side decoder feeding
        # a CLIENT-type frame would otherwise share the numeric arm.
        # The redundant ``msg_class is LeaderResponse`` check is kept
        # as belt-and-suspenders, but the load-bearing predicate is
        # the direction guard — a future alias/subclass registered in
        # ``RESPONSE_TYPES`` would not silently bypass the legacy
        # dispatch.
        if (
            not self._is_request
            and header.msg_type == ResponseType.LEADER
            and self._version == PROTOCOL_VERSION_LEGACY
            and msg_class is LeaderResponse
        ):
            return LeaderResponse.decode_body_legacy(body)

        # RowsResponse takes an extra ``max_rows`` cap; ServersResponse
        # takes ``unknown_role_policy`` for forward-compat tolerance of
        # role bytes outside {0,1,2}; other classes share the generic
        # (body, schema) signature.
        if msg_class is RowsResponse:
            return RowsResponse.decode_body(
                body, schema=header.schema, max_rows=self._max_rows, text_errors=self._text_errors
            )
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

    def _force_handshake_for_stateless(self, version: int) -> None:
        """Bypass the handshake state machine for the stateless helper.

        Used ONLY by :func:`decode_message`, the stateless single-frame
        convenience function that has no inbound handshake bytes.
        Production / streaming callers MUST go through
        :meth:`decode_handshake` so the peek/commit/consume discipline
        runs against real wire bytes.

        Centralising the bypass on the class gives any future
        observability / metrics / setter introduction on the handshake
        state a single anchor to update — the previous inline
        ``decoder._handshake_done = True; decoder._version = version``
        write in ``decode_message`` would silently skip a future
        ``version.setter`` or ``_handshake_done`` property and lose
        whatever side effects the setter encapsulated.
        """
        # Defense-in-depth re-validation: ``MessageDecoder.__init__``
        # already runs the same check against ``_SUPPORTED_VERSIONS``,
        # so this branch is unreachable today. It future-proofs the
        # helper against a refactor that moves the constructor's
        # validation elsewhere.
        if version not in _SUPPORTED_VERSIONS:
            raise HandshakeError(
                f"Unsupported protocol version: {version:#x}. "
                f"Supported: {', '.join(f'{v:#x}' for v in sorted(_SUPPORTED_VERSIONS))}"
            )
        self._version = version
        self._handshake_done = True


_UNSET: Final[object] = object()


def decode_message(
    data: bytes | bytearray | memoryview,
    is_request: bool = False,
    version: int = PROTOCOL_VERSION,
    unknown_role_policy: str = "reject",
    max_message_size: int | None = None,
    max_rows: int | None = None,
    max_continuation_frames: int | None | object = _UNSET,
    max_total_rows: int | None | object = _UNSET,
    text_errors: str = "strict",
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
        unknown_role_policy: Forwarded to ``MessageDecoder``; controls
                 ``ServersResponse`` behaviour when an unknown role
                 byte is decoded. One of ``"reject"`` (default,
                 raises), ``"warn"`` (substitute SPARE + log warning),
                 or ``"accept"`` (substitute SPARE silently).
        max_message_size: Forwarded to ``MessageDecoder``. ``None``
                 keeps the default (``ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE``,
                 64 MiB). Legitimate large frames (a real cluster's
                 ``FilesResponse`` dump, a wide ``RowsResponse``) need
                 the cap raised; without this kwarg callers had to drop
                 the helper and instantiate ``MessageDecoder`` by hand.
        max_rows: Forwarded to ``MessageDecoder``. ``None`` keeps the
                 default (``RowsResponse.DEFAULT_MAX_ROWS``, 1_000_000).
        max_continuation_frames: Forwarded to ``MessageDecoder``. Pass
                 ``None`` to disable the cap; omit to keep
                 ``DEFAULT_MAX_CONTINUATION_FRAMES``. The stateless
                 helper decodes a single frame so the cap is mostly
                 inert here, but the kwarg is wired through so a
                 caller building a streaming-equivalent decoder via
                 the helper sees the same cap surface.
        max_total_rows: Forwarded to ``MessageDecoder``. Same
                 semantics as ``max_continuation_frames``.
        text_errors: Forwarded to ``MessageDecoder``. UTF-8 decode
                 error handler for per-row TEXT / ISO8601 cells. The
                 default ``"strict"`` matches the dqlite wire-spec
                 contract; see :meth:`MessageDecoder.__init__` for
                 the full set of accepted values and the caveats of
                 the permissive modes.
    """
    kwargs: dict[str, object] = {
        "is_request": is_request,
        "version": version,
        "unknown_role_policy": unknown_role_policy,
        "text_errors": text_errors,
    }
    if max_message_size is not None:
        kwargs["max_message_size"] = max_message_size
    if max_rows is not None:
        kwargs["max_rows"] = max_rows
    # ``max_continuation_frames`` / ``max_total_rows`` accept ``None``
    # (disable cap) AND a positive int as legitimate values, so a
    # sentinel is needed to discriminate "caller omitted the kwarg" from
    # "caller passed None explicitly to disable the cap".
    if max_continuation_frames is not _UNSET:
        kwargs["max_continuation_frames"] = max_continuation_frames
    if max_total_rows is not _UNSET:
        kwargs["max_total_rows"] = max_total_rows
    decoder = MessageDecoder(**kwargs)  # type: ignore[arg-type]
    if is_request:
        # Request decoders start with ``_handshake_done=False`` because
        # production callers obtain the version from the inbound
        # handshake. The stateless helper has no wire, so route the
        # bypass through the named ``_force_handshake_for_stateless``
        # method — a single anchor for any future observability /
        # setter introduction on the handshake state machine.
        decoder._force_handshake_for_stateless(version)
    return decoder.decode_bytes(data)


def encode_message(message: Message, *, max_message_size: int | None = None) -> bytes:
    """Convenience function to encode a single message.

    Args:
        message: The message to encode.
        max_message_size: Forwarded to ``MessageEncoder``. ``None``
                 keeps the default (``ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE``,
                 64 MiB) — symmetric with ``decode_message``'s same
                 kwarg. Legitimate large frames (a real cluster's
                 ``FilesResponse`` dump, a wide ``RowsResponse``) need
                 the cap raised on the encoder AND the decoder for the
                 round trip to stay symmetric.
    """
    if max_message_size is None:
        return MessageEncoder().encode(message)
    return MessageEncoder(max_message_size=max_message_size).encode(message)
