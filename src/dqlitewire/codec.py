"""Message encoder and decoder for dqlite wire protocol."""

from dqlitewire.buffer import ReadBuffer
from dqlitewire.constants import HEADER_SIZE, PROTOCOL_VERSION, RequestType, ResponseType
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

    def __init__(self, is_request: bool = False) -> None:
        """Initialize decoder.

        Args:
            is_request: If True, decode as request messages.
                       If False (default), decode as response messages.
        """
        self._buffer = ReadBuffer()
        self._is_request = is_request
        self._type_map = REQUEST_TYPES if is_request else RESPONSE_TYPES
        # Request decoders (server-side) must receive the protocol version
        # handshake before decoding any messages. Response decoders (client-side)
        # don't receive an inbound handshake, so they skip this check.
        self._handshake_done = not is_request

    def feed(self, data: bytes) -> None:
        """Feed data to the decoder."""
        self._buffer.feed(data)

    def has_message(self) -> bool:
        """Check if a complete message is available."""
        return self._buffer.has_message()

    def decode(self) -> Message | None:
        """Decode the next message from the buffer.

        Returns None if no complete message is available.
        Raises ProtocolError if called on a request decoder before decode_handshake().
        """
        if not self._handshake_done:
            raise ProtocolError(
                "Protocol handshake not yet received. Call decode_handshake() before decode()."
            )
        data = self._buffer.read_message()
        if data is None:
            return None

        return self.decode_bytes(data)

    def decode_bytes(self, data: bytes) -> Message:
        """Decode a message from bytes."""
        if len(data) < HEADER_SIZE:
            raise DecodeError(f"Message too short: {len(data)} bytes")

        header = Header.decode(data[:HEADER_SIZE])
        body = data[HEADER_SIZE:]

        msg_class = self._type_map.get(header.msg_type)
        if msg_class is None:
            raise DecodeError(f"Unknown message type: {header.msg_type}")

        max_schema = _MAX_SCHEMA.get(header.msg_type, 0)
        if header.schema > max_schema:
            raise DecodeError(
                f"Unsupported schema version {header.schema} for message type "
                f"{header.msg_type} (max supported: {max_schema})"
            )

        return msg_class.decode_body(body, schema=header.schema)

    def decode_handshake(self) -> int | None:
        """Decode protocol version handshake.

        Returns the protocol version or None if not enough data.
        Must be called before decode() on request decoders.
        """
        data = self._buffer.read_bytes(8)
        if data is None:
            return None
        self._handshake_done = True
        return int.from_bytes(data, "little")


def decode_message(data: bytes, is_request: bool = False) -> Message:
    """Convenience function to decode a single message."""
    decoder = MessageDecoder(is_request=is_request)
    return decoder.decode_bytes(data)


def encode_message(message: Message) -> bytes:
    """Convenience function to encode a single message."""
    return message.encode()
