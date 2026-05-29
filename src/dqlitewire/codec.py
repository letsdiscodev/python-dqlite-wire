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

# HEARTBEAT and CONNECT are intentionally absent: upstream C's gateway.c
# rejects both for client-gateway traffic, so they decode to the unknown-type
# error path here. The classes live on privately for test/golden-byte harnesses.
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


# Max supported schema version per message type (default 0). Keyed separately
# by direction because RequestType and ResponseType share numeric codes; a
# single dict would let a server slip a response past a request's schema ceiling.
_REQUEST_MAX_SCHEMA: Final[dict[int, int]] = {
    RequestType.PREPARE: 1,
    RequestType.EXEC: 1,
    RequestType.QUERY: 1,
    RequestType.EXEC_SQL: 1,
    RequestType.QUERY_SQL: 1,
}
_RESPONSE_MAX_SCHEMA: Final[dict[int, int]] = {
    # STMT bumped to schema=1 for the V1 stmt-id+tail-offset shape.
    ResponseType.STMT: 1,
    # All other responses are schema=0 only; the .get(..., 0) fallback at the
    # dispatch sites fails fast on any non-zero schema. To bump one, add it here
    # AND extend its decode_body to dispatch on schema.
}


_SUPPORTED_VERSIONS: Final[frozenset[int]] = frozenset({PROTOCOL_VERSION, PROTOCOL_VERSION_LEGACY})


class MessageEncoder:
    """Encodes messages to wire protocol format. NOT thread-safe (single-owner)."""

    def __reduce__(self) -> NoReturn:
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
        """version drives the handshake bytes only; body shape is always modern
        (call LeaderResponse.encode_body_legacy() directly for a legacy body).
        max_message_size must match the decoder's cap for round-trip symmetry.
        """
        if version not in _SUPPORTED_VERSIONS:
            raise HandshakeError(
                f"Unsupported protocol version: {version:#x}. "
                f"Supported: {', '.join(f'{v:#x}' for v in sorted(_SUPPORTED_VERSIONS))}"
            )
        if max_message_size < 1:
            raise ValueError(f"max_message_size must be >= 1, got {max_message_size}")
        if max_message_size > ReadBuffer.MAX_MESSAGE_SIZE_CEILING:
            raise ValueError(
                f"max_message_size must be <= {ReadBuffer.MAX_MESSAGE_SIZE_CEILING} "
                f"(UINT32_MAX bytes; the C server's per-frame ceiling at "
                f"dqlite-upstream/src/conn.c:169 rejects any single frame above this "
                f"bound), got {max_message_size}"
            )
        self._version = version
        self._max_message_size = max_message_size

    def encode(self, message: Message) -> bytes:
        """Encode a message to bytes; raise EncodeError if it exceeds max_message_size."""
        frame = message.encode()
        if len(frame) > self._max_message_size:
            # Diagnostic wording matches the decoder side's "exceeds maximum".
            raise EncodeError(
                f"Message size {len(frame):#x} bytes exceeds maximum {self._max_message_size}"
            )
        return frame

    def encode_handshake(self) -> bytes:
        """Encode the protocol version handshake; must be sent before any other message."""
        return self._version.to_bytes(8, "little")


class MessageDecoder:
    """Decodes messages from wire protocol format. NOT thread-safe (single-owner).

    Concurrent misuse causes silent data corruption (lost-update races on the
    ReadBuffer's _pos), NOT exceptions; the is_poisoned flag does not detect it.
    Wrap every call site in a lock at the layer owning the socket.
    """

    def __reduce__(self) -> NoReturn:
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
        """version is ignored for request decoders (read from the handshake).
        max_rows is exclusive: set max_rows = N + 1 to permit N rows per frame.
        max_continuation_frames counts the initial frame; None disables it.
        max_total_rows / max_continuation_frames cap CPU and memory against a
        slow-dripping server. text_errors permissive modes carry the round-trip
        and log-sanitiser caveats documented on dqlitewire.types.decode_text.
        """
        # Validate version uniformly across both directions: decode_message and
        # direct is_request=True callers can otherwise smuggle an unsupported
        # value past, so callers catch one HandshakeError regardless of side.
        if version not in _SUPPORTED_VERSIONS:
            raise HandshakeError(
                f"Unsupported protocol version: {version:#x}. "
                f"Supported: {', '.join(f'{v:#x}' for v in sorted(_SUPPORTED_VERSIONS))}"
            )
        if max_message_size > ReadBuffer.MAX_MESSAGE_SIZE_CEILING:
            # Checked here too so the full kwarg set validates before the
            # ReadBuffer (which also enforces this ceiling) is instantiated.
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
            # DecodeError (not ValueError) mirrors the deeper
            # ServersResponse.decode_body validator so one except arm catches both.
            raise DecodeError(
                f"unknown_role_policy must be one of 'reject', 'warn', 'accept'; "
                f"got {unknown_role_policy!r}"
            )
        # Validate against stdlib error-handler names so a typo fails fast at
        # construction rather than on the first TEXT cell deep in decode_body.
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
        # Promoted to a decoder attribute so the stateless decode_bytes path can
        # enforce the same cap the streaming path checks via read_message.
        self._max_message_size = max_message_size
        self._is_request = is_request
        self._type_map = REQUEST_TYPES if is_request else RESPONSE_TYPES
        self._max_schema = _REQUEST_MAX_SCHEMA if is_request else _RESPONSE_MAX_SCHEMA
        # Client-side: version from the constructor. Server-side: set by decode_handshake().
        self._version: int | None = version if not is_request else None
        # Request (server-side) decoders must receive the handshake first.
        self._handshake_done = not is_request
        self._continuation_expected = False
        self._max_rows = max_rows
        self._max_continuation_frames: int | None = max_continuation_frames
        self._max_total_rows: int | None = max_total_rows
        self._continuation_frame_count = 0
        self._continuation_total_rows = 0
        # Snapshotted on the initial frame; continuation frames must agree or the
        # decoder would silently truncate (count=0) or overrun (count>original).
        self._continuation_column_count: int | None = None
        # Same strict cross-frame check on the names tuple: the public accessor
        # reports the initial frame's names, so a rotated names list would
        # silently mis-attribute per-row data.
        self._continuation_column_names: tuple[str, ...] | None = None

    @property
    def version(self) -> int | None:
        """Protocol version from handshake, or None if not yet received."""
        return self._version

    @property
    def is_poisoned(self) -> bool:
        """True after an unrecoverable mid-stream error; every decode* then raises
        ProtocolError until reset(), preventing decode from an unknown offset."""
        return self._buffer.is_poisoned

    def reset(self) -> None:
        """Reset to a fresh, un-poisoned state (re-arms handshake). Use after a reconnect."""
        self._buffer.reset()
        self._finalize_continuation_state()

    def _finalize_continuation_state(self) -> None:
        """Reset all per-stream continuation counters and flags.

        Call ONLY from termination arms (every clean and dirty exit of
        decode_continuation, and reset()). NOT from the ContinuationError guards
        in skip_message/decode: those do not terminate the continuation, so
        clearing the flag would drop the invariant the guard enforces.
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
        self._buffer.feed(data)

    def has_message(self) -> bool:
        return self._buffer.has_message()

    def skip_message(self) -> bool:
        """Skip the current message; recovers after an oversized-message DecodeError.

        Returns False (check is_skipping) until an oversized message is fully
        discarded. Raises ContinuationError if a ROWS continuation is in progress.
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
        """Decode the next ROWS continuation frame (call after has_more=True).

        A mid-continuation EmptyResponse is also a conforming terminator: upstream
        emits it when the query was cancelled by INTERRUPT; returning it mirrors
        go-dqlite's drain loop. Returns None if incomplete; raises ContinuationError
        if no continuation is in progress, ServerFailure on a FailureResponse, and
        StreamError on any other type.
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
            # Oversized continuation frame: no skip_message() recovery in
            # continuation mode, so poison to force reset().
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

            # Type-recognition check before the schema cap so an unknown type
            # with schema=1 reports "unknown type", not "unsupported schema".
            if header.msg_type not in (
                ResponseType.FAILURE,
                ResponseType.EMPTY,
                ResponseType.ROWS,
            ):
                # Snapshot counters before finalising; the buffer is NOT poisoned
                # here (coherent offset), so these are the only record of progress.
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
                # Decode body before clearing the flag so a malformed body still
                # hits the broad except and poisons; a well-formed failure is a
                # clean signal that leaves the offset coherent.
                failure = FailureResponse.decode_body(body, schema=header.schema)
                self._finalize_continuation_state()
                raise ServerFailure(failure.code, failure.message)
            if header.msg_type == ResponseType.EMPTY:
                # Clean terminator for an INTERRUPT-cancelled query (mirrors
                # go-dqlite's drain loop); not poisoned. A malformed body raises
                # DecodeError caught below and re-raised without poison since the
                # offset is already coherent. Decode before finalise like FAILURE.
                empty_result = EmptyResponse.decode_body(body, schema=header.schema)
                self._finalize_continuation_state()
                return empty_result
            # msg_type is ROWS here.

            result = RowsResponse.decode_body(
                body, schema=header.schema, max_rows=self._max_rows, text_errors=self._text_errors
            )
            # Per-stream caps: frame count bounds CPU, cumulative rows bound
            # memory against a slow-dripping server.
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
            # Continuation frames must keep the initial frame's column count or
            # the stream silently truncates (drop to 0) or overruns (grows).
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
            # Same strict check on names: the public accessor reports only the
            # initial frame's names, so rotated names would mis-attribute row data.
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
            # Coherent offset, flag already cleared: do NOT poison.
            raise
        except StreamError:
            # Wrong type mid-continuation but coherent offset (read_message
            # advanced past it); one bad frame should not kill the connection,
            # so do NOT poison. Matches go-dqlite's Protocol.Recv.
            raise
        except DecodeError:
            # Body decode failed but offset is coherent (read_message advanced):
            # re-raise without poison so the caller can resync.
            self._finalize_continuation_state()
            raise
        except BaseException as e:
            # Finalise before poisoning to stay symmetric with the clean arms.
            self._finalize_continuation_state()
            self._buffer.poison(
                e
                if isinstance(e, Exception)
                else RuntimeError(f"decode_continuation interrupted: {type(e).__name__}")
            )
            raise

    def decode(self) -> Message | None:
        """Decode the next message; None if incomplete.

        Raises if poisoned, if a request decoder hasn't handshaked, or if a ROWS
        continuation is in progress (drain via decode_continuation() or reset()).
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
        # An oversized-header DecodeError here is recoverable via skip_message()
        # (bytes not yet consumed), so do NOT poison on it.
        data = self._buffer.read_message()
        if data is None:
            return None

        # Bytes consumed: any failure now desyncs the stream, so poison. Catch
        # BaseException so a signal-delivered KeyboardInterrupt also poisons.
        try:
            msg = self.decode_bytes(data)
            # The flag store MUST stay inside the try: an async exception between
            # a successful decode and the store would otherwise leave the stream
            # desynced without poisoning, mis-framing the next continuation frame.
            if isinstance(msg, RowsResponse) and msg.has_more:
                # Check the cap before writing any _continuation_* field so a
                # poison exit doesn't leave them populated (the broad except
                # below deliberately skips _finalize to preserve handshake state).
                initial_row_count = len(msg.rows)
                if self._max_total_rows is not None and initial_row_count > self._max_total_rows:
                    raise DecodeError(
                        f"ROWS initial frame already exceeded max_total_rows cap "
                        f"({self._max_total_rows}); first frame had "
                        f"{initial_row_count} rows"
                    )
                self._continuation_expected = True
                # Seed the counters and column snapshots from the initial frame
                # for the cross-frame checks in decode_continuation.
                self._continuation_frame_count = 1
                self._continuation_total_rows = initial_row_count
                self._continuation_column_count = len(msg.column_names)
                self._continuation_column_names = tuple(msg.column_names)
        except BaseException as e:
            # Wrap non-Exception BaseExceptions so the poison cause stays inspectable.
            self._buffer.poison(
                e
                if isinstance(e, Exception)
                else RuntimeError(f"decode interrupted: {type(e).__name__}")
            )
            raise

        return msg

    def decode_bytes(self, data: bytes | bytearray | memoryview) -> Message:
        """Decode a message from bytes-like input."""
        self._buffer._check_poisoned()
        if not self._handshake_done:
            raise HandshakeError(
                "Protocol handshake not yet received. "
                "Call decode_handshake() before decode_bytes()."
            )
        # Materialise to bytes once: the per-message decoders own a bytes contract.
        if not isinstance(data, bytes):
            data = bytes(data)
        if len(data) < HEADER_SIZE:
            raise DecodeError(f"Message too short: {len(data)} bytes")

        header = Header.decode(data[:HEADER_SIZE])
        body_size = header.body_size
        # Enforce max_message_size after Header.decode (so a torn header still
        # reports "too short") and before the body slice (so the cap fires
        # before any large allocation), matching the streaming path's check.
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
        # Reject trailing bytes for parity with the per-message decoders;
        # unreachable via streaming (read_message returns an exact slice), this
        # only catches malformed input from direct callers.
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

        # Legacy LeaderResponse has no node_id prefix. The selector is locked to
        # the constructor's self._version (no byte-sniff fallback), so a misaligned
        # decoder version silently misdecodes the body — callers must use the
        # version negotiated on the wire. Gate on not self._is_request first:
        # RequestType.CLIENT == ResponseType.LEADER == 1 would otherwise collide.
        if (
            not self._is_request
            and header.msg_type == ResponseType.LEADER
            and self._version == PROTOCOL_VERSION_LEGACY
            and msg_class is LeaderResponse
        ):
            return LeaderResponse.decode_body_legacy(body)

        # RowsResponse and ServersResponse take extra kwargs; others use (body, schema).
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
        """Decode the version handshake; return the version or None if short.

        Must precede decode() on request decoders. On an unsupported version the
        8 bytes are left untouched so a retry is deterministic. Signal-safety:
        commit state then consume, reverting on an async exception so the bytes
        stay in the buffer and a retry repeats the peek/validate/commit cycle.
        """
        if self._handshake_done:
            raise HandshakeError("Handshake already completed")
        # Peek first so we only commit (and consume) on a valid version.
        peek = self._buffer.peek_bytes(8)
        if peek is None:
            return None
        version = int.from_bytes(peek, "little")
        if version not in _SUPPORTED_VERSIONS:
            raise HandshakeError(f"Unsupported protocol version: {version:#x}")
        # Commit before consuming; revert on an async exception so the pair is atomic.
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
        """Bypass the handshake state machine for the stateless decode_message helper.

        Streaming callers MUST use decode_handshake() instead. Centralised on the
        class so a future handshake-state setter has one anchor to update.
        """
        # Re-validation is unreachable today (__init__ checks the same set); kept
        # in case the constructor's validation ever moves.
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
    """Decode a single message statelessly (no handshake enforcement).

    Use MessageDecoder for protocol-aware streaming. All kwargs forward to it.
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
    # Sentinel needed because None is a legitimate value (disable the cap),
    # distinct from "caller omitted the kwarg".
    if max_continuation_frames is not _UNSET:
        kwargs["max_continuation_frames"] = max_continuation_frames
    if max_total_rows is not _UNSET:
        kwargs["max_total_rows"] = max_total_rows
    decoder = MessageDecoder(**kwargs)  # type: ignore[arg-type]
    if is_request:
        # No wire here, so bypass the handshake the request decoder would
        # otherwise require.
        decoder._force_handshake_for_stateless(version)
    return decoder.decode_bytes(data)


def encode_message(message: Message, *, max_message_size: int | None = None) -> bytes:
    """Encode a single message. max_message_size must match the decoder's to round-trip."""
    if max_message_size is None:
        return MessageEncoder().encode(message)
    return MessageEncoder(max_message_size=max_message_size).encode(message)
