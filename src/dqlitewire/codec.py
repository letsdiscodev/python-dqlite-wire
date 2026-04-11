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
    """Encodes messages to wire protocol format."""

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
    """Decodes messages from wire protocol format."""

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
        """
        return self._buffer.skip_message()

    @property
    def is_skipping(self) -> bool:
        """True if still discarding bytes from an oversized message."""
        return self._buffer.is_skipping

    def decode_continuation(
        self,
        column_names: list[str],
        column_count: int,
        max_rows: int = RowsResponse.DEFAULT_MAX_ROWS,
    ) -> RowsResponse | None:
        """Decode a ROWS continuation message from the buffer.

        After receiving a RowsResponse with has_more=True, the server sends
        additional ROWS messages containing only row data (no column_count or
        column_names prefix). Use this method instead of decode() for those
        continuation messages.

        Args:
            column_names: Column names from the initial RowsResponse.
            column_count: Number of columns from the initial RowsResponse.
            max_rows: Maximum rows to decode per message.

        Returns None if no complete message is available.
        """
        self._buffer._check_poisoned()
        data = self._buffer.read_message()
        if data is None:
            return None

        # Bytes have been consumed — ANY parse failure here leaves the
        # buffer at an unknown offset and must poison the decoder. Catch
        # broadly: `decode_body` implementations can raise struct.error,
        # ValueError, UnicodeDecodeError, IndexError, etc., and all of
        # them mean the stream is desynchronized.
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

            return RowsResponse.decode_rows_continuation(
                body, column_names, column_count, max_rows=max_rows
            )
        except Exception as e:
            self._buffer.poison(e)
            raise

    def decode(self) -> Message | None:
        """Decode the next message from the buffer.

        Returns None if no complete message is available.
        Raises ProtocolError if called on a request decoder before decode_handshake().
        Raises ProtocolError if the decoder is poisoned.
        """
        self._buffer._check_poisoned()
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

        # Bytes have been consumed. ANY parse failure now leaves the
        # buffer at an unknown offset; poison so subsequent calls fail
        # fast. Catch broadly: `decode_body` implementations can raise
        # struct.error, ValueError, UnicodeDecodeError, IndexError, etc.,
        # and all of them mean the stream is desynchronized.
        try:
            return self.decode_bytes(data)
        except Exception as e:
            self._buffer.poison(e)
            raise

    def decode_bytes(self, data: bytes) -> Message:
        """Decode a message from bytes.

        Raises ProtocolError if called on a request decoder before decode_handshake().
        """
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
        """
        if self._handshake_done:
            raise ProtocolError("Handshake already completed")
        data = self._buffer.read_bytes(8)
        if data is None:
            return None
        version = int.from_bytes(data, "little")
        if version not in _SUPPORTED_VERSIONS:
            raise ProtocolError(f"Unsupported protocol version: {version:#x}")
        self._version = version
        self._handshake_done = True
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
