"""Tests for message codec."""

import pytest

from dqlitewire.codec import (
    MessageDecoder,
    MessageEncoder,
    decode_message,
    encode_message,
)
from dqlitewire.constants import PROTOCOL_VERSION, PROTOCOL_VERSION_LEGACY
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages import (
    ClientRequest,
    DbResponse,
    EmptyResponse,
    FailureResponse,
    HeartbeatRequest,
    LeaderRequest,
    LeaderResponse,
    OpenRequest,
    PrepareRequest,
    ResultResponse,
    ServersResponse,
    StmtResponse,
    WelcomeResponse,
)


class TestMessageEncoder:
    def test_encode_handshake(self) -> None:
        encoder = MessageEncoder()
        handshake = encoder.encode_handshake()
        assert len(handshake) == 8
        assert int.from_bytes(handshake, "little") == PROTOCOL_VERSION

    def test_encode_handshake_legacy(self) -> None:
        """Encoder with legacy version should produce the legacy handshake word."""
        encoder = MessageEncoder(version=PROTOCOL_VERSION_LEGACY)
        handshake = encoder.encode_handshake()
        assert len(handshake) == 8
        assert int.from_bytes(handshake, "little") == PROTOCOL_VERSION_LEGACY

    def test_legacy_version_constant_matches_go(self) -> None:
        """PROTOCOL_VERSION_LEGACY must match Go's VersionLegacy = 0x86104dd760433fe5."""
        assert PROTOCOL_VERSION_LEGACY == 0x86104DD760433FE5

    def test_encoder_has_no_buffer_attribute(self) -> None:
        """MessageEncoder should not have unused _buffer attribute."""
        encoder = MessageEncoder()
        assert not hasattr(encoder, "_buffer")

    def test_encode_message(self) -> None:
        from dqlitewire.constants import RequestType
        from dqlitewire.messages.base import Header

        encoder = MessageEncoder()
        msg = LeaderRequest()
        encoded = encoder.encode(msg)
        assert len(encoded) == 16
        header = Header.decode(encoded[:8])
        assert header.msg_type == RequestType.LEADER
        assert header.schema == 0
        assert header.size_words == 1


class TestMessageDecoder:
    def test_decode_handshake(self) -> None:
        decoder = MessageDecoder(is_request=True)
        decoder.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        version = decoder.decode_handshake()
        assert version == PROTOCOL_VERSION

    def test_decode_handshake_partial(self) -> None:
        decoder = MessageDecoder(is_request=True)
        decoder.feed(b"\x01\x00\x00")
        version = decoder.decode_handshake()
        assert version is None

    def test_decode_response(self) -> None:
        msg = LeaderResponse(node_id=1, address="localhost:9001")
        encoded = msg.encode()

        decoder = MessageDecoder(is_request=False)
        decoder.feed(encoded)
        assert decoder.has_message()

        decoded = decoder.decode()
        assert isinstance(decoded, LeaderResponse)
        assert decoded.node_id == 1
        assert decoded.address == "localhost:9001"

    def test_decode_request(self) -> None:
        msg = LeaderRequest()
        encoded = msg.encode()

        decoder = MessageDecoder(is_request=True)
        handshake = PROTOCOL_VERSION.to_bytes(8, "little")
        decoder.feed(handshake + encoded)
        decoder.decode_handshake()
        assert decoder.has_message()

        decoded = decoder.decode()
        assert isinstance(decoded, LeaderRequest)

    def test_decode_partial_message(self) -> None:
        msg = LeaderResponse(node_id=1, address="localhost:9001")
        encoded = msg.encode()

        decoder = MessageDecoder()
        # Feed only part of the message
        decoder.feed(encoded[:5])
        assert not decoder.has_message()
        assert decoder.decode() is None

        # Feed the rest
        decoder.feed(encoded[5:])
        assert decoder.has_message()

        decoded = decoder.decode()
        assert isinstance(decoded, LeaderResponse)

    def test_decode_multiple_messages(self) -> None:
        msg1 = LeaderResponse(node_id=1, address="node1")
        msg2 = WelcomeResponse(heartbeat_timeout=10000)
        encoded = msg1.encode() + msg2.encode()

        decoder = MessageDecoder()
        decoder.feed(encoded)

        decoded1 = decoder.decode()
        assert isinstance(decoded1, LeaderResponse)
        assert decoded1.address == "node1"

        decoded2 = decoder.decode()
        assert isinstance(decoded2, WelcomeResponse)
        assert decoded2.heartbeat_timeout == 10000

    def test_decode_unknown_type(self) -> None:
        # Create a message with an invalid type
        invalid = b"\x00\x00\x00\x00\xff\x00\x00\x00"  # type 255

        decoder = MessageDecoder()
        with pytest.raises(DecodeError, match="Unknown message type"):
            decoder.decode_bytes(invalid)

    def test_header_decode_short_data_raises_decode_error(self) -> None:
        """Header.decode() must raise DecodeError, not ValueError."""
        from dqlitewire.messages.base import Header

        with pytest.raises(DecodeError):
            Header.decode(b"\x00" * 7)

    def test_header_decode_empty_raises_decode_error(self) -> None:
        """Header.decode() with empty bytes must raise DecodeError."""
        from dqlitewire.messages.base import Header

        with pytest.raises(DecodeError):
            Header.decode(b"")

    def test_header_roundtrip_minimal(self) -> None:
        """Minimal header with msg_type=0 and schema=0 roundtrips."""
        from dqlitewire.messages.base import Header

        header = Header(size_words=1, msg_type=0, schema=0)
        data = header.encode()
        decoded = Header.decode(data)
        assert decoded.size_words == 1
        assert decoded.msg_type == 0

    def test_header_encode_overflow_raises_encode_error(self) -> None:
        """Header with size_words exceeding uint32 must raise EncodeError."""
        from dqlitewire.exceptions import EncodeError
        from dqlitewire.messages.base import Header

        header = Header(size_words=2**32, msg_type=0, schema=0)
        with pytest.raises(EncodeError, match="header"):
            header.encode()

    def test_unknown_schema_version_raises(self) -> None:
        """Unknown schema versions should raise DecodeError, not silently default."""
        from dqlitewire.messages.base import Header

        # Build an ExecRequest with schema=5 (unknown)
        body = b"\x01\x00\x00\x00" + b"\x02\x00\x00\x00"  # db_id=1, stmt_id=2
        header = Header(size_words=1, msg_type=5, schema=5)  # type 5 = EXEC
        data = header.encode() + body

        with pytest.raises(DecodeError, match="[Uu]nsupported schema"):
            decode_message(data, is_request=True)

    def test_decode_bytes_too_short(self) -> None:
        """decode_bytes with data shorter than HEADER_SIZE should raise DecodeError."""
        decoder = MessageDecoder()
        with pytest.raises(DecodeError, match="too short"):
            decoder.decode_bytes(b"\x00" * 7)

    def test_decode_bytes_empty(self) -> None:
        """decode_bytes with empty data should raise DecodeError."""
        decoder = MessageDecoder()
        with pytest.raises(DecodeError, match="too short"):
            decoder.decode_bytes(b"")


class TestHandshakeStateEnforcement:
    """Verify that request decoders enforce handshake-before-decode ordering."""

    def test_decode_handshake_rejects_unknown_version(self) -> None:
        """decode_handshake() must reject unknown protocol versions."""
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        # Feed garbage version bytes
        decoder.feed(b"\x42\x42\x42\x42\x42\x42\x42\x42")
        with pytest.raises(ProtocolError, match="[Uu]nsupported protocol version"):
            decoder.decode_handshake()

    def test_decode_handshake_failure_does_not_consume_bytes(self) -> None:
        """On an unsupported version, decode_handshake() must NOT consume the
        handshake bytes.

        Previously the method read 8 bytes BEFORE validating the version, so
        a failed handshake left the buffer advanced by 8 with ``_handshake_done``
        still False. A retry consumed the next 8 bytes as a "version", which
        was almost always the header of a real message — silently desynchronizing
        the stream. Peek-before-consume means the bytes stay in the buffer and
        a retry is deterministic: same bytes, same error.
        """
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        bogus = b"\x42" * 8
        decoder.feed(bogus)

        with pytest.raises(ProtocolError, match="[Uu]nsupported protocol version"):
            decoder.decode_handshake()

        # The 8 handshake bytes must still be in the buffer.
        assert decoder._buffer.available() == 8
        assert not decoder._handshake_done

        # A retry on the same bytes gets the same error, deterministically.
        with pytest.raises(ProtocolError, match="[Uu]nsupported protocol version"):
            decoder.decode_handshake()
        assert decoder._buffer.available() == 8

    def test_decode_handshake_partial_data_leaves_bytes_intact(self) -> None:
        """With fewer than 8 bytes buffered, decode_handshake() returns None
        and leaves the partial data untouched so a subsequent feed() can
        complete the handshake.
        """
        decoder = MessageDecoder(is_request=True)
        version_bytes = PROTOCOL_VERSION_LEGACY.to_bytes(8, "little")
        decoder.feed(version_bytes[:4])
        assert decoder.decode_handshake() is None
        assert decoder._buffer.available() == 4

        decoder.feed(version_bytes[4:])
        assert decoder.decode_handshake() == PROTOCOL_VERSION_LEGACY

    def test_decode_handshake_failure_preserves_following_bytes(self) -> None:
        """Direct reproducer of the original bug: a buffer containing a bogus
        version followed by a real valid version. The pre-fix code consumed
        both 8-byte chunks across two retries; the fix must consume neither
        on the first failure.
        """
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        bogus = b"\x42" * 8
        valid = PROTOCOL_VERSION_LEGACY.to_bytes(8, "little")
        decoder.feed(bogus + valid)

        with pytest.raises(ProtocolError, match="[Uu]nsupported protocol version"):
            decoder.decode_handshake()
        # All 16 bytes still in the buffer — neither chunk has been consumed.
        assert decoder._buffer.available() == 16

        # Even after a retry, the valid bytes behind are untouched.
        with pytest.raises(ProtocolError, match="[Uu]nsupported protocol version"):
            decoder.decode_handshake()
        assert decoder._buffer.available() == 16

    def test_decode_handshake_recoverable_via_reset(self) -> None:
        """After a handshake failure, reset() clears the buffer and the
        decoder accepts a fresh handshake on a reconnect.
        """
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        decoder.feed(b"\x42" * 8)
        with pytest.raises(ProtocolError, match="[Uu]nsupported protocol version"):
            decoder.decode_handshake()

        decoder.reset()
        assert decoder._buffer.available() == 0
        assert not decoder._handshake_done

        # Fresh valid handshake works.
        decoder.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        assert decoder.decode_handshake() == PROTOCOL_VERSION
        assert decoder._handshake_done

    def test_peek_bytes_does_not_advance_position(self) -> None:
        """Sanity: ReadBuffer.peek_bytes() must return the requested bytes
        without advancing _pos. This is the primitive decode_handshake()
        depends on.
        """
        from dqlitewire.buffer import ReadBuffer

        buf = ReadBuffer()
        buf.feed(b"abcdefghij")
        assert buf.peek_bytes(4) == b"abcd"
        assert buf.available() == 10
        # Repeated peeks return the same bytes.
        assert buf.peek_bytes(4) == b"abcd"
        assert buf.available() == 10
        # Asking for more than available returns None.
        assert buf.peek_bytes(20) is None
        assert buf.available() == 10

    def test_decode_handshake_rejects_zero_version(self) -> None:
        """Version 0 is not a valid protocol version."""
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        decoder.feed(b"\x00" * 8)
        with pytest.raises(ProtocolError, match="[Uu]nsupported protocol version"):
            decoder.decode_handshake()

    def test_decode_handshake_accepts_legacy_version(self) -> None:
        """Legacy version (0x86104dd760433fe5) must be accepted."""
        decoder = MessageDecoder(is_request=True)
        decoder.feed(PROTOCOL_VERSION_LEGACY.to_bytes(8, "little"))
        version = decoder.decode_handshake()
        assert version == PROTOCOL_VERSION_LEGACY

    def test_request_decoder_rejects_decode_before_handshake(self) -> None:
        """A request decoder must not allow decode() before decode_handshake().

        The dqlite wire protocol requires the client to send an 8-byte protocol
        version before any messages. If a request decoder (server-side) skips
        the handshake, it would misinterpret the version bytes as a message header.
        """
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        msg = LeaderRequest()
        decoder.feed(msg.encode())

        with pytest.raises(ProtocolError, match="[Hh]andshake"):
            decoder.decode()

    def test_request_decoder_allows_decode_after_handshake(self) -> None:
        """After handshake, decode() should work normally."""
        decoder = MessageDecoder(is_request=True)

        # Feed handshake + message
        handshake = PROTOCOL_VERSION.to_bytes(8, "little")
        msg = LeaderRequest()
        decoder.feed(handshake + msg.encode())

        version = decoder.decode_handshake()
        assert version == PROTOCOL_VERSION

        decoded = decoder.decode()
        assert isinstance(decoded, LeaderRequest)

    def test_request_decoder_decode_bytes_rejects_before_handshake(self) -> None:
        """decode_bytes() must also enforce handshake on request decoders."""
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        msg = LeaderRequest()
        with pytest.raises(ProtocolError, match="[Hh]andshake"):
            decoder.decode_bytes(msg.encode())

    def test_double_decode_handshake_raises(self) -> None:
        """Calling decode_handshake() twice must raise ProtocolError.

        A second call would consume 8 bytes of actual message data from the
        buffer and interpret them as a version number, silently corrupting
        the stream.
        """
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        handshake = PROTOCOL_VERSION.to_bytes(8, "little")
        msg = LeaderRequest()
        decoder.feed(handshake + msg.encode())

        # First handshake succeeds
        version = decoder.decode_handshake()
        assert version == PROTOCOL_VERSION

        # Second handshake must raise, not consume message bytes
        with pytest.raises(ProtocolError, match="[Hh]andshake already completed"):
            decoder.decode_handshake()

        # The message should still be decodable (not consumed by second handshake)
        decoded = decoder.decode()
        assert isinstance(decoded, LeaderRequest)

    def test_decode_handshake_on_client_decoder_raises(self) -> None:
        """Client-side decoder should reject decode_handshake() since _handshake_done=True."""
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        decoder.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        with pytest.raises(ProtocolError, match="[Hh]andshake already completed"):
            decoder.decode_handshake()

    def test_response_decoder_decode_bytes_works_without_handshake(self) -> None:
        """decode_bytes() on response decoders should work without handshake."""
        decoder = MessageDecoder(is_request=False)
        msg = LeaderResponse(node_id=1, address="localhost:9001")
        decoded = decoder.decode_bytes(msg.encode())
        assert isinstance(decoded, LeaderResponse)

    def test_response_decoder_allows_decode_without_handshake(self) -> None:
        """Response decoders (client-side) don't require inbound handshake."""
        decoder = MessageDecoder(is_request=False)
        msg = LeaderResponse(node_id=1, address="localhost:9001")
        decoder.feed(msg.encode())

        # Should work without calling decode_handshake first
        decoded = decoder.decode()
        assert isinstance(decoded, LeaderResponse)

    def test_legacy_handshake_decodes_leader_response_as_legacy(self) -> None:
        """Legacy version should decode LeaderResponse in legacy format."""
        from dqlitewire.codec import decode_message
        from dqlitewire.constants import ResponseType
        from dqlitewire.messages.base import Header
        from dqlitewire.types import encode_text

        address = "192.168.1.1:9001"
        body = encode_text(address)
        header = Header(
            size_words=len(body) // 8,
            msg_type=ResponseType.LEADER,
            schema=0,
        )
        data = header.encode() + body

        decoded = decode_message(data, is_request=False, version=PROTOCOL_VERSION_LEGACY)
        assert isinstance(decoded, LeaderResponse)
        assert decoded.node_id == 0
        assert decoded.address == address

    def test_modern_handshake_decodes_leader_response_as_modern(self) -> None:
        """Modern version should decode LeaderResponse in modern format."""
        from dqlitewire.codec import decode_message

        msg = LeaderResponse(node_id=42, address="node1:9001")

        decoded = decode_message(msg.encode(), is_request=False, version=PROTOCOL_VERSION)
        assert isinstance(decoded, LeaderResponse)
        assert decoded.node_id == 42
        assert decoded.address == "node1:9001"

    def test_decoder_version_property_request(self) -> None:
        """Request decoder should expose version=None before handshake, version after."""
        decoder = MessageDecoder(is_request=True)
        assert decoder.version is None
        decoder.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        decoder.decode_handshake()
        assert decoder.version == PROTOCOL_VERSION

    def test_client_decoder_with_version_parameter(self) -> None:
        """Client-side decoder should accept version parameter for legacy support."""
        from dqlitewire.constants import ResponseType
        from dqlitewire.messages.base import Header
        from dqlitewire.types import encode_text

        address = "192.168.1.1:9001"
        body = encode_text(address)
        header = Header(
            size_words=len(body) // 8,
            msg_type=ResponseType.LEADER,
            schema=0,
        )
        data = header.encode() + body

        # With legacy version, should decode using legacy format
        decoder = MessageDecoder(is_request=False, version=PROTOCOL_VERSION_LEGACY)
        assert decoder.version == PROTOCOL_VERSION_LEGACY
        decoder.feed(data)
        decoded = decoder.decode()
        assert isinstance(decoded, LeaderResponse)
        assert decoded.node_id == 0
        assert decoded.address == address

    def test_client_decoder_default_version_is_modern(self) -> None:
        """Client-side decoder should default to modern protocol version."""
        decoder = MessageDecoder(is_request=False)
        assert decoder.version == PROTOCOL_VERSION

    def test_client_decoder_modern_version_decodes_modern_leader(self) -> None:
        """Client-side decoder with modern version should use modern format."""
        decoder = MessageDecoder(is_request=False, version=PROTOCOL_VERSION)
        msg = LeaderResponse(node_id=42, address="node1:9001")
        decoder.feed(msg.encode())
        decoded = decoder.decode()
        assert isinstance(decoded, LeaderResponse)
        assert decoded.node_id == 42
        assert decoded.address == "node1:9001"


class TestConvenienceFunctions:
    def test_encode_message(self) -> None:
        msg = ClientRequest(client_id=42)
        encoded = encode_message(msg)
        # Verify full roundtrip, not just length
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, ClientRequest)
        assert decoded.client_id == 42

    def test_decode_message_response(self) -> None:
        msg = FailureResponse(code=1, message="test")
        encoded = msg.encode()
        decoded = decode_message(encoded, is_request=False)
        assert isinstance(decoded, FailureResponse)
        assert decoded.code == 1

    def test_decode_message_request(self) -> None:
        msg = OpenRequest(name="test.db")
        encoded = msg.encode()
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, OpenRequest)
        assert decoded.name == "test.db"

    def test_decode_message_with_legacy_version(self) -> None:
        """decode_message with version=PROTOCOL_VERSION_LEGACY should use legacy format."""
        from dqlitewire.messages.base import Header
        from dqlitewire.types import encode_text

        address = "10.0.0.1:9001"
        body = encode_text(address)
        from dqlitewire.constants import ResponseType

        header = Header(size_words=len(body) // 8, msg_type=ResponseType.LEADER)
        message_bytes = header.encode() + body

        decoded = decode_message(message_bytes, version=PROTOCOL_VERSION_LEGACY)
        assert isinstance(decoded, LeaderResponse)
        assert decoded.node_id == 0
        assert decoded.address == address


class TestRoundTrip:
    """End-to-end encode/decode tests."""

    def test_leader_request(self) -> None:
        original = LeaderRequest()
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, LeaderRequest)

    def test_client_request(self) -> None:
        original = ClientRequest(client_id=123456789)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, ClientRequest)
        assert decoded.client_id == 123456789

    def test_open_request(self) -> None:
        original = OpenRequest(name="my_database.db", flags=6, vfs="unix-excl")
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, OpenRequest)
        assert decoded.name == "my_database.db"
        assert decoded.flags == 6
        assert decoded.vfs == "unix-excl"

    def test_db_response(self) -> None:
        original = DbResponse(db_id=7)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=False)
        assert isinstance(decoded, DbResponse)
        assert decoded.db_id == 7

    def test_result_response(self) -> None:
        original = ResultResponse(last_insert_id=100, rows_affected=5)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=False)
        assert isinstance(decoded, ResultResponse)
        assert decoded.last_insert_id == 100
        assert decoded.rows_affected == 5

    def test_failure_response_unicode(self) -> None:
        original = FailureResponse(code=42, message="Error: \u00e9\u00e8\u00e0")
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=False)
        assert isinstance(decoded, FailureResponse)
        assert decoded.message == "Error: \u00e9\u00e8\u00e0"

    def test_prepare_request_schema_survives_roundtrip(self) -> None:
        """PrepareRequest with schema=1 must preserve schema through codec round-trip."""
        original = PrepareRequest(db_id=1, sql="SELECT 1", schema=1)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, PrepareRequest)
        assert decoded.schema == 1

    def test_decode_bytes_slices_body_to_header_size(self) -> None:
        """decode_bytes() must not pass trailing bytes to the message decoder.

        AssignRequest uses len(data) to distinguish Promote (1 word) from
        Assign (2 words). If trailing bytes leak through, a Promote message
        would be misinterpreted as Assign with a garbage role.
        """
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.requests import AssignRequest
        from dqlitewire.types import encode_uint64

        # Build a Promote message (1 word body: just node_id)
        body = encode_uint64(42)  # node_id=42, no role
        header = Header(size_words=1, msg_type=13, schema=0)  # type 13 = ASSIGN

        # Append garbage trailing bytes that look like a role field
        trailing = encode_uint64(99)
        data = header.encode() + body + trailing

        decoded = decode_message(data, is_request=True)
        assert isinstance(decoded, AssignRequest)
        assert decoded.node_id == 42
        # Must be None (Promote), not 99 (which would happen if trailing leaked)
        assert decoded.role is None

    def test_decode_bytes_rejects_short_body(self) -> None:
        """decode_bytes() must raise DecodeError if body is shorter than header claims."""
        from dqlitewire.messages.base import Header

        # Header claims 2 words (16 bytes) but only 8 bytes follow
        header = Header(size_words=2, msg_type=0, schema=0)
        data = header.encode() + b"\x00" * 8  # Only 8 bytes, not 16
        with pytest.raises(DecodeError, match="[Bb]ody.*short"):
            decode_message(data, is_request=True)

    def test_heartbeat_request(self) -> None:
        original = HeartbeatRequest(timestamp=1710000000)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, HeartbeatRequest)
        assert decoded.timestamp == 1710000000

    def test_finalize_request(self) -> None:
        from dqlitewire.messages.requests import FinalizeRequest

        original = FinalizeRequest(db_id=1, stmt_id=42)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, FinalizeRequest)
        assert decoded.db_id == 1
        assert decoded.stmt_id == 42

    def test_interrupt_request(self) -> None:
        from dqlitewire.messages.requests import InterruptRequest

        original = InterruptRequest(db_id=7)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, InterruptRequest)
        assert decoded.db_id == 7

    def test_connect_request(self) -> None:
        from dqlitewire.messages.requests import ConnectRequest

        original = ConnectRequest(node_id=5, address="10.0.0.1:9001")
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, ConnectRequest)
        assert decoded.node_id == 5
        assert decoded.address == "10.0.0.1:9001"

    def test_add_request(self) -> None:
        from dqlitewire.messages.requests import AddRequest

        original = AddRequest(node_id=3, address="node3:9001")
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, AddRequest)
        assert decoded.node_id == 3

    def test_assign_request(self) -> None:
        from dqlitewire.messages.requests import AssignRequest

        original = AssignRequest(node_id=1, role=2)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, AssignRequest)
        assert decoded.node_id == 1
        assert decoded.role == 2

    def test_remove_request(self) -> None:
        from dqlitewire.messages.requests import RemoveRequest

        original = RemoveRequest(node_id=99)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, RemoveRequest)
        assert decoded.node_id == 99

    def test_dump_request(self) -> None:
        from dqlitewire.messages.requests import DumpRequest

        original = DumpRequest(name="mydb")
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, DumpRequest)
        assert decoded.name == "mydb"

    def test_cluster_request(self) -> None:
        from dqlitewire.messages.requests import ClusterRequest

        original = ClusterRequest(format=1)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, ClusterRequest)
        assert decoded.format == 1

    def test_transfer_request(self) -> None:
        from dqlitewire.messages.requests import TransferRequest

        original = TransferRequest(target_node_id=42)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, TransferRequest)
        assert decoded.target_node_id == 42

    def test_describe_request(self) -> None:
        from dqlitewire.messages.requests import DescribeRequest

        original = DescribeRequest(format=0)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, DescribeRequest)
        assert decoded.format == 0

    def test_weight_request(self) -> None:
        from dqlitewire.messages.requests import WeightRequest

        original = WeightRequest(weight=100)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, WeightRequest)
        assert decoded.weight == 100

    def test_welcome_response(self) -> None:
        original = WelcomeResponse(heartbeat_timeout=15000)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=False)
        assert isinstance(decoded, WelcomeResponse)
        assert decoded.heartbeat_timeout == 15000

    def test_stmt_response(self) -> None:
        original = StmtResponse(db_id=1, stmt_id=2, num_params=3)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=False)
        assert isinstance(decoded, StmtResponse)
        assert decoded.db_id == 1
        assert decoded.stmt_id == 2
        assert decoded.num_params == 3
        assert decoded.tail_offset is None

    def test_empty_response(self) -> None:
        original = EmptyResponse()
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=False)
        assert isinstance(decoded, EmptyResponse)

    def test_servers_response(self) -> None:
        from dqlitewire.messages.responses import NodeInfo

        original = ServersResponse(nodes=[NodeInfo(1, "n1:9001", 0), NodeInfo(2, "n2:9001", 1)])
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=False)
        assert isinstance(decoded, ServersResponse)
        assert len(decoded.nodes) == 2
        assert decoded.nodes[0].node_id == 1
        assert decoded.nodes[1].address == "n2:9001"

    def test_metadata_response(self) -> None:
        from dqlitewire.messages.responses import MetadataResponse

        original = MetadataResponse(failure_domain=1, weight=50)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=False)
        assert isinstance(decoded, MetadataResponse)
        assert decoded.failure_domain == 1
        assert decoded.weight == 50

    def test_files_response(self) -> None:
        from dqlitewire.messages.responses import FilesResponse

        original = FilesResponse(files={"db.sqlite": b"\x00" * 512})
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=False)
        assert isinstance(decoded, FilesResponse)
        assert decoded.files["db.sqlite"] == b"\x00" * 512

    def test_255_params_uses_v0_schema(self) -> None:
        """Exactly 255 params should use V0 (schema=0), not V1."""
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.requests import ExecRequest

        msg = ExecRequest(db_id=0, stmt_id=0, params=list(range(255)))
        assert msg._get_schema() == 0

        encoded = encode_message(msg)
        header = Header.decode(encoded[:8])
        assert header.schema == 0

        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, ExecRequest)
        assert len(decoded.params) == 255
        assert decoded.params == list(range(255))

    def test_max_uint64_roundtrip(self) -> None:
        """Max uint64 values should survive full codec roundtrip."""
        original = ClientRequest(client_id=2**64 - 1)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, ClientRequest)
        assert decoded.client_id == 2**64 - 1

    def test_stmt_response_v0_with_trailing_data_not_detected_as_v1(self) -> None:
        """StmtResponse V0 with trailing bytes must not be misdetected as V1."""
        from dqlitewire.messages.base import Header
        from dqlitewire.types import encode_uint32, encode_uint64

        # Build a V0 body (16 bytes) + 8 extra trailing bytes (total 24)
        # Without schema awareness, len >= 24 triggers V1 decode reading garbage tail_offset
        body = encode_uint32(1) + encode_uint32(2) + encode_uint64(3) + b"\x00" * 8
        header = Header(size_words=3, msg_type=5, schema=0)
        data = header.encode() + body
        decoded = decode_message(data, is_request=False)
        assert isinstance(decoded, StmtResponse)
        # With schema=0 in header, tail_offset should NOT be decoded
        assert decoded.tail_offset is None


class TestDecoderContinuation:
    """Test decode_continuation() for multi-part ROWS responses."""

    def test_decode_continuation_exists(self) -> None:
        """MessageDecoder should have a decode_continuation method."""
        decoder = MessageDecoder(is_request=False)
        assert hasattr(decoder, "decode_continuation")

    def test_decode_continuation_roundtrip(self) -> None:
        """Multi-part ROWS: initial decode() + continuation decode_continuation()."""
        from dqlitewire.constants import ROW_DONE_MARKER, ROW_PART_MARKER, ValueType
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import RowsResponse
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        types = [ValueType.INTEGER, ValueType.TEXT]

        # Build initial ROWS message with PART marker
        body1 = encode_uint64(2)  # column_count
        body1 += encode_text("id") + encode_text("name")
        body1 += encode_row_header(types)
        body1 += encode_row_values([1, "alice"], types)
        body1 += encode_uint64(ROW_PART_MARKER)
        header1 = Header(size_words=len(body1) // 8, msg_type=7, schema=0)
        msg1_bytes = header1.encode() + body1

        # Build continuation ROWS message with DONE marker
        # (C server always includes column_count + column_names)
        body2 = encode_uint64(2)
        body2 += encode_text("id") + encode_text("name")
        body2 += encode_row_header(types)
        body2 += encode_row_values([2, "bob"], types)
        body2 += encode_uint64(ROW_DONE_MARKER)
        header2 = Header(size_words=len(body2) // 8, msg_type=7, schema=0)
        msg2_bytes = header2.encode() + body2

        # Feed both messages
        decoder = MessageDecoder(is_request=False)
        decoder.feed(msg1_bytes + msg2_bytes)

        # Decode initial response
        initial = decoder.decode()
        assert isinstance(initial, RowsResponse)
        assert initial.has_more is True
        assert len(initial.rows) == 1
        assert initial.rows[0] == [1, "alice"]

        # Decode continuation
        continuation = decoder.decode_continuation()
        assert isinstance(continuation, RowsResponse)
        assert continuation.has_more is False
        assert len(continuation.rows) == 1
        assert continuation.rows[0] == [2, "bob"]

    def test_decode_continuation_with_column_header(self) -> None:
        """Continuation frames from the C server include column_count +
        column_names (same layout as the initial frame). Verify that
        decode_continuation handles this correctly.
        """
        from dqlitewire.constants import ROW_DONE_MARKER, ROW_PART_MARKER, ValueType
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import RowsResponse
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        types = [ValueType.INTEGER, ValueType.TEXT]

        # Initial ROWS message (PART marker)
        body1 = encode_uint64(2)
        body1 += encode_text("id") + encode_text("name")
        body1 += encode_row_header(types)
        body1 += encode_row_values([1, "alice"], types)
        body1 += encode_uint64(ROW_PART_MARKER)
        h1 = Header(size_words=len(body1) // 8, msg_type=7, schema=0)

        # Continuation WITH column header (matching C server output)
        body2 = encode_uint64(2)
        body2 += encode_text("id") + encode_text("name")
        body2 += encode_row_header(types)
        body2 += encode_row_values([2, "bob"], types)
        body2 += encode_uint64(ROW_DONE_MARKER)
        h2 = Header(size_words=len(body2) // 8, msg_type=7, schema=0)

        decoder = MessageDecoder(is_request=False)
        decoder.feed(h1.encode() + body1 + h2.encode() + body2)

        initial = decoder.decode()
        assert isinstance(initial, RowsResponse)
        assert initial.has_more is True
        assert initial.rows[0] == [1, "alice"]

        cont = decoder.decode_continuation()
        assert isinstance(cont, RowsResponse)
        assert cont.has_more is False
        assert cont.rows[0] == [2, "bob"]

    def test_decode_continuation_returns_none_when_no_data(self) -> None:
        """decode_continuation should return None when no message is available."""
        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        result = decoder.decode_continuation()
        assert result is None

    def test_decode_continuation_raises_on_failure_response(self) -> None:
        """decode_continuation should surface server error, not a confusing DecodeError."""
        from dqlitewire.exceptions import ProtocolError
        from dqlitewire.messages.responses import FailureResponse

        failure = FailureResponse(code=266, message="disk I/O error")
        failure_bytes = failure.encode()

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(failure_bytes)

        with pytest.raises(ProtocolError, match="disk I/O error") as exc_info:
            decoder.decode_continuation()
        # Must be a ProtocolError, NOT a DecodeError (which would mean the
        # failure body was misinterpreted as row data).
        assert type(exc_info.value) is ProtocolError

    def test_decode_continuation_raises_on_unexpected_type(self) -> None:
        """decode_continuation should raise ProtocolError for non-ROWS, non-FAILURE type."""
        from dqlitewire.exceptions import ProtocolError

        result = ResultResponse(last_insert_id=0, rows_affected=0)
        result_bytes = result.encode()

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(result_bytes)

        with pytest.raises(ProtocolError, match="Expected ROWS continuation"):
            decoder.decode_continuation()


class TestDecoderContinuationExpected:
    """Regression tests for issue 058.

    When ``decode()`` returns a ``RowsResponse`` with ``has_more=True``,
    the decoder enters a "continuation expected" state. Calling ``decode()``
    again before draining all continuations via ``decode_continuation()``
    is a protocol error — the next frame in the buffer is a continuation
    (no column header prefix), so ``decode()`` would misparse it. The
    ``_continuation_expected`` flag makes this misuse fail loudly with
    ``ProtocolError`` instead of producing silent stream desynchronization.
    """

    def test_decode_raises_when_continuation_expected(self) -> None:
        """decode() must refuse while a ROWS continuation is in progress."""
        from dqlitewire.constants import ROW_PART_MARKER, ValueType
        from dqlitewire.exceptions import ProtocolError
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import RowsResponse
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        types = [ValueType.INTEGER]

        # Build a RowsResponse with has_more=True
        body = encode_uint64(1)  # column_count
        body += encode_text("id")
        body += encode_row_header(types)
        body += encode_row_values([1], types)
        body += encode_uint64(ROW_PART_MARKER)
        header = Header(size_words=len(body) // 8, msg_type=7, schema=0)
        msg_bytes = header.encode() + body

        # Also feed a second (standalone) message that decode() would try to read
        second = ResultResponse(last_insert_id=0, rows_affected=0).encode()

        decoder = MessageDecoder(is_request=False)
        decoder.feed(msg_bytes + second)

        # First decode() returns the initial RowsResponse with has_more=True
        result = decoder.decode()
        assert isinstance(result, RowsResponse)
        assert result.has_more is True

        # Second decode() must raise — we're in "continuation expected" state
        with pytest.raises(ProtocolError, match="continuation"):
            decoder.decode()

    def test_decode_continuation_clears_flag(self) -> None:
        """After draining all continuations (has_more=False), decode() works again."""
        from dqlitewire.constants import ROW_DONE_MARKER, ROW_PART_MARKER, ValueType
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import RowsResponse
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        types = [ValueType.INTEGER]

        # Initial frame (has_more=True)
        body1 = encode_uint64(1)
        body1 += encode_text("id")
        body1 += encode_row_header(types)
        body1 += encode_row_values([1], types)
        body1 += encode_uint64(ROW_PART_MARKER)
        h1 = Header(size_words=len(body1) // 8, msg_type=7, schema=0)

        # Continuation frame (has_more=False) — includes column header
        body2 = encode_uint64(1)
        body2 += encode_text("id")
        body2 += encode_row_header(types)
        body2 += encode_row_values([2], types)
        body2 += encode_uint64(ROW_DONE_MARKER)
        h2 = Header(size_words=len(body2) // 8, msg_type=7, schema=0)

        # Normal message after the ROWS sequence
        normal = ResultResponse(last_insert_id=5, rows_affected=3).encode()

        decoder = MessageDecoder(is_request=False)
        decoder.feed(h1.encode() + body1 + h2.encode() + body2 + normal)

        # decode initial
        initial = decoder.decode()
        assert isinstance(initial, RowsResponse) and initial.has_more

        # decode continuation
        cont = decoder.decode_continuation()
        assert isinstance(cont, RowsResponse) and not cont.has_more

        # Now decode() should work again
        result = decoder.decode()
        assert isinstance(result, ResultResponse)
        assert result.last_insert_id == 5

    def test_reset_clears_continuation_expected(self) -> None:
        """reset() must clear the continuation-expected flag."""
        from dqlitewire.constants import ROW_PART_MARKER, ValueType
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import RowsResponse
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        types = [ValueType.INTEGER]
        body = encode_uint64(1)
        body += encode_text("id")
        body += encode_row_header(types)
        body += encode_row_values([1], types)
        body += encode_uint64(ROW_PART_MARKER)
        header = Header(size_words=len(body) // 8, msg_type=7, schema=0)

        decoder = MessageDecoder(is_request=False)
        decoder.feed(header.encode() + body)
        result = decoder.decode()
        assert isinstance(result, RowsResponse) and result.has_more

        # Reset should clear the flag
        decoder.reset()
        normal = ResultResponse(last_insert_id=0, rows_affected=0).encode()
        decoder.feed(normal)
        msg = decoder.decode()
        assert isinstance(msg, ResultResponse)

    def test_has_more_false_does_not_set_flag(self) -> None:
        """A RowsResponse with has_more=False should NOT set the flag."""
        from dqlitewire.messages.responses import RowsResponse

        decoder = MessageDecoder(is_request=False)
        msg = RowsResponse(
            column_names=["x"],
            column_types=[1],
            rows=[[1]],
            has_more=False,
        )
        normal = ResultResponse(last_insert_id=0, rows_affected=0).encode()
        decoder.feed(msg.encode() + normal)

        result = decoder.decode()
        assert isinstance(result, RowsResponse) and not result.has_more

        # decode() should work fine — no continuation expected
        result2 = decoder.decode()
        assert isinstance(result2, ResultResponse)


class TestContinuationFlagCompleteness:
    """Regression tests for issues 063 and 064.

    Issue 058 added the ``_continuation_expected`` flag and made
    ``decode()`` check it. Issues 063 and 064 complete the state
    machine: ``decode_continuation()`` must refuse when the flag is
    False (no continuation in progress), and ``skip_message()`` must
    refuse when the flag is True (continuation in progress).
    """

    def test_decode_continuation_raises_when_not_expected(self) -> None:
        """decode_continuation() must refuse when no continuation is
        in progress — calling it without a prior has_more=True would
        silently consume and misparse the next message.
        """
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        result = ResultResponse(last_insert_id=0, rows_affected=0)
        decoder.feed(result.encode())

        with pytest.raises(ProtocolError, match="no ROWS continuation"):
            decoder.decode_continuation()

        # The message must NOT have been consumed
        assert decoder.has_message(), (
            "decode_continuation must not consume the message when the guard fires"
        )
        # Caller can still decode it correctly
        msg = decoder.decode()
        assert isinstance(msg, ResultResponse)

    def test_skip_message_raises_during_continuation(self) -> None:
        """skip_message() must refuse while a continuation is in progress."""
        from dqlitewire.constants import ROW_PART_MARKER, ValueType
        from dqlitewire.exceptions import ProtocolError
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import RowsResponse
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        types = [ValueType.INTEGER]
        body = encode_uint64(1)
        body += encode_text("id")
        body += encode_row_header(types)
        body += encode_row_values([1], types)
        body += encode_uint64(ROW_PART_MARKER)
        header = Header(size_words=len(body) // 8, msg_type=7, schema=0)

        decoder = MessageDecoder(is_request=False)
        decoder.feed(header.encode() + body)
        result = decoder.decode()
        assert isinstance(result, RowsResponse) and result.has_more

        with pytest.raises(ProtocolError, match="continuation"):
            decoder.skip_message()


class TestDecoderSkipMessage:
    """Test skip_message() and is_skipping on MessageDecoder."""

    def test_skip_message_exists_on_decoder(self) -> None:
        """MessageDecoder should expose skip_message() for oversized recovery."""
        decoder = MessageDecoder(is_request=False)
        assert hasattr(decoder, "skip_message")
        assert hasattr(decoder, "is_skipping")

    def test_skip_oversized_and_recover(self) -> None:
        """After skipping an oversized message, normal messages should decode."""
        import struct

        from dqlitewire.exceptions import DecodeError

        # Create a decoder with a small max_message_size
        decoder = MessageDecoder(is_request=False)
        decoder._buffer._max_message_size = 128

        # Build an oversized message header (size_words=100 → 808 bytes total)
        oversized_header = struct.pack("<IBBH", 100, 0, 0, 0)
        # Build a normal message after it
        normal_msg = FailureResponse(code=42, message="test").encode()

        # Feed just the oversized header
        decoder.feed(oversized_header)

        # has_message() is total; the raise surfaces from decode() / read_message().
        assert decoder.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            decoder.decode()

        # skip_message() should handle the oversized message
        result = decoder.skip_message()
        assert result is False  # haven't received full oversized body yet
        assert decoder.is_skipping is True

        # Feed enough bytes to complete the capped skip + normal message.
        # _skip_remaining is capped to max_message_size (128), header was 8,
        # so _skip_remaining = 128 - 8 = 120 bytes.
        decoder.feed(b"\x00" * decoder._buffer._skip_remaining + normal_msg)

        # Now should be able to decode the normal message
        assert decoder.is_skipping is False
        decoded = decoder.decode()
        assert isinstance(decoded, FailureResponse)
        assert decoded.code == 42


class TestDecoderPoisonedState:
    """Once the decoder has produced a mid-stream error from already-consumed
    bytes, subsequent operations must fail fast with a poisoned-state error —
    the buffer's _pos is in an unknown position and silently continuing would
    corrupt downstream reads. A reset() method returns the decoder to a clean
    state after a reconnect.
    """

    def test_decode_poisons_on_unknown_message_type(self) -> None:
        """An unknown message type surfaces via decode(), which must poison
        the decoder so retries fail loudly.
        """
        import struct

        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        # Valid framing, but an unregistered message type (0xFE).
        header = struct.pack("<IBBH", 0, 0xFE, 0, 0)
        decoder.feed(header)

        assert decoder.is_poisoned is False
        with pytest.raises(DecodeError):
            decoder.decode()
        assert decoder.is_poisoned is True

        # Subsequent operations fail fast with a ProtocolError from poisoning.
        # Per issue 038, every mutating/consuming entry point on the buffer
        # and decoder — including feed() — refuses to run on a poisoned
        # buffer. Recovery requires reset().
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.feed(struct.pack("<IBBH", 0, 0xFE, 0, 0))
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()

    def test_reset_unpoisons_decoder(self) -> None:
        """reset() clears buffer state and the poison flag so the decoder
        can be reused after a reconnect.
        """
        import struct

        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        header = struct.pack("<IBBH", 0, 0xFE, 0, 0)
        decoder.feed(header)
        with pytest.raises(DecodeError):
            decoder.decode()
        assert decoder.is_poisoned is True

        decoder.reset()
        assert decoder.is_poisoned is False

        # Fresh valid message should decode cleanly.
        leader = LeaderResponse(node_id=1, address="host:1234").encode()
        decoder.feed(leader)
        result = decoder.decode()
        assert isinstance(result, LeaderResponse)
        assert result.node_id == 1

        # Sanity: a fresh poison error references a ProtocolError ancestor too.
        decoder.feed(struct.pack("<IBBH", 0, 0xFE, 0, 0))
        with pytest.raises(ProtocolError):
            decoder.decode()

    def test_oversized_header_does_not_poison(self) -> None:
        """An oversized-header error from the buffer framing layer is
        recoverable via skip_message(); it must NOT poison because the
        bytes have not yet been consumed from the buffer.
        """
        import struct

        decoder = MessageDecoder(is_request=False)
        decoder._buffer._max_message_size = 128
        oversized = struct.pack("<IBBH", 100, 0, 0, 0)  # 808-byte body > 128
        decoder.feed(oversized)

        with pytest.raises(DecodeError, match="exceeds maximum"):
            decoder.decode()
        # NOT poisoned — skip_message() is the documented recovery path.
        assert decoder.is_poisoned is False

        # Recovery via skip_message should still work.
        decoder.skip_message()
        decoder.feed(b"\x00" * decoder._buffer._skip_remaining)
        assert decoder.is_poisoned is False

        # And a fresh message decodes afterwards.
        normal = FailureResponse(code=1, message="ok").encode()
        decoder.feed(normal)
        assert isinstance(decoder.decode(), FailureResponse)

    def test_poison_catches_non_protocol_exceptions(self) -> None:
        """Any exception from decode_bytes — not just DecodeError/ProtocolError —
        must poison the decoder. A `decode_body` implementation could raise
        struct.error, ValueError, UnicodeDecodeError, IndexError, etc., and
        all of them mean the bytes have been consumed and the offset is now
        unknown. Catching only the protocol exception types would let other
        failures leak through with the buffer in a desynchronized state.
        """
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        # Build a real message and feed it.
        msg = LeaderResponse(node_id=1, address="x:1").encode()
        decoder.feed(msg)

        # Monkey-patch decode_bytes to raise a non-protocol exception, simulating
        # a bug or unexpected input in a real decode_body implementation.
        sentinel = ValueError("simulated body parse failure")

        def boom(_data: bytes) -> object:
            raise sentinel

        decoder.decode_bytes = boom  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="simulated body parse failure"):
            decoder.decode()
        assert decoder.is_poisoned is True

        # Subsequent decode() must raise poisoned, even though decode_bytes
        # is monkey-patched. Restore the real decode_bytes first to make
        # sure the poison check fires before any decode_bytes call.
        # Per issue 038, feed() on a poisoned buffer also raises, so we
        # cannot push more bytes through without a reset — the poison
        # check fires directly on decode().
        del decoder.decode_bytes  # type: ignore[attr-defined]
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.feed(msg)
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()

    def test_decode_continuation_poisons_on_wrong_type(self) -> None:
        """decode_continuation() must poison the decoder when the next
        message is not a ROWS continuation. The bytes have been consumed
        from the buffer; subsequent operations cannot trust the offset.
        """
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        # Feed a non-ROWS message where decode_continuation expects ROWS.
        # An EmptyResponse is type 127, body is empty.
        empty = EmptyResponse().encode()
        decoder.feed(empty)

        with pytest.raises(ProtocolError, match="Expected ROWS continuation"):
            decoder.decode_continuation()
        assert decoder.is_poisoned is True

        # Subsequent decode() also fails poisoned.
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()

    def test_decode_bytes_honors_poison(self) -> None:
        """Regression for issue 039.

        ``MessageDecoder.decode_bytes`` is a public method that parses a
        single caller-supplied bytes object. ``decode()`` goes through a
        poison gate before calling ``decode_bytes``, but a user who
        catches ``ProtocolError`` from ``decode()`` and tries to
        "resume" by feeding the raw bytes to ``decode_bytes`` directly
        bypasses the poison contract entirely. Any public entry point
        that consumes bytes and runs parser code must refuse to run on
        a poisoned decoder; ``decode_bytes`` was the last gap.
        """
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        decoder._buffer.poison(DecodeError("original cause"))

        # Build a well-formed message that would otherwise decode fine.
        msg = LeaderResponse(node_id=1, address="x:1").encode()
        with pytest.raises(ProtocolError, match="poisoned") as ei:
            decoder.decode_bytes(msg)
        assert isinstance(ei.value.__cause__, DecodeError)

        # decode_message() helper creates a fresh decoder on every call,
        # so it must still work even though the dispatch name overlaps.
        from dqlitewire.codec import decode_message

        fresh = decode_message(msg, is_request=False)
        assert isinstance(fresh, LeaderResponse)
        assert fresh.node_id == 1

    def test_poison_is_first_error_wins(self) -> None:
        """ReadBuffer.poison() must NOT overwrite an existing poison error.
        First error wins so the original __cause__ remains visible.
        """
        from dqlitewire.buffer import ReadBuffer

        buf = ReadBuffer()
        first = DecodeError("first failure")
        second = DecodeError("second failure")
        buf.poison(first)
        buf.poison(second)
        assert buf._poisoned is first

    def test_request_decoder_reset_clears_handshake_state(self) -> None:
        """For a server-side (request) decoder, reset() must also un-do the
        handshake so a reconnect requires a fresh handshake. Otherwise the
        decoder would silently accept message bytes as a continuation of
        the dead session.
        """
        decoder = MessageDecoder(is_request=True)
        # Complete a handshake.
        decoder.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        decoder.decode_handshake()
        assert decoder._handshake_done is True
        assert decoder.version == PROTOCOL_VERSION

        decoder.reset()
        assert decoder._handshake_done is False
        assert decoder.version is None
        # And the decoder must refuse decode() until a fresh handshake.
        from dqlitewire.exceptions import ProtocolError

        with pytest.raises(ProtocolError, match="[Hh]andshake"):
            decoder.decode()


class TestDecoderPoisonOnMalformedBody:
    """Verify decode() poisons on a malformed message body — not just
    on unknown types or oversized headers.
    """

    def test_decode_poisons_on_truncated_rows_body(self) -> None:
        """A ROWS message with a valid header but a body too short for
        the claimed column_count must poison the decoder.
        """
        import struct

        from dqlitewire.constants import ResponseType
        from dqlitewire.exceptions import ProtocolError

        # Valid ROWS header (type 7), body = 8 bytes (column_count=1
        # but no column name follows — decode_body will fail)
        body = struct.pack("<Q", 1)
        header = struct.pack("<IBBH", len(body) // 8, ResponseType.ROWS, 0, 0)

        decoder = MessageDecoder(is_request=False)
        decoder.feed(header + body)

        with pytest.raises((DecodeError, ProtocolError)):
            decoder.decode()

        assert decoder.is_poisoned, (
            "decode() must poison the buffer when decode_body raises on a malformed message body"
        )

        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()


class TestStreamingContinuation:
    """End-to-end streaming decode tests for the continuation protocol.
    These verify the full feed→decode→decode_continuation path with
    realistic wire data (including column headers in continuations,
    matching the C server format).
    """

    def test_full_streaming_continuation_roundtrip(self) -> None:
        """082: encode multi-part, feed, decode initial + continuation,
        verify all row data and column names.
        """
        from dqlitewire.constants import ValueType
        from dqlitewire.messages.responses import RowsResponse

        initial = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[[1, "alice"]],
            has_more=True,
        )
        continuation = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[[2, "bob"]],
            has_more=False,
        )

        decoder = MessageDecoder(is_request=False)
        decoder.feed(initial.encode() + continuation.encode())

        msg1 = decoder.decode()
        assert isinstance(msg1, RowsResponse)
        assert msg1.has_more is True
        assert msg1.rows == [[1, "alice"]]
        assert msg1.column_names == ["id", "name"]

        msg2 = decoder.decode_continuation()
        assert isinstance(msg2, RowsResponse)
        assert msg2.has_more is False
        assert msg2.rows == [[2, "bob"]]

        # Decoder ready for next message
        normal = ResultResponse(last_insert_id=0, rows_affected=0)
        decoder.feed(normal.encode())
        msg3 = decoder.decode()
        assert isinstance(msg3, ResultResponse)

    def test_failure_during_continuation_poisons(self) -> None:
        """083: server sends FailureResponse instead of continuation."""
        from dqlitewire.constants import ValueType
        from dqlitewire.exceptions import ProtocolError
        from dqlitewire.messages.responses import FailureResponse, RowsResponse

        initial = RowsResponse(
            column_names=["id"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            has_more=True,
        )
        failure = FailureResponse(code=5, message="disk I/O error")

        decoder = MessageDecoder(is_request=False)
        decoder.feed(initial.encode() + failure.encode())

        msg = decoder.decode()
        assert isinstance(msg, RowsResponse) and msg.has_more

        with pytest.raises(ProtocolError, match="disk I/O error"):
            decoder.decode_continuation()

        assert decoder.is_poisoned

    def test_chained_continuations_part_part_done(self) -> None:
        """084: three frames — initial(PART) + cont(PART) + cont(DONE)."""
        from dqlitewire.constants import ValueType
        from dqlitewire.messages.responses import RowsResponse

        frame1 = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            has_more=True,
        )
        frame2 = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[2]],
            has_more=True,
        )
        frame3 = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[3]],
            has_more=False,
        )

        decoder = MessageDecoder(is_request=False)
        decoder.feed(frame1.encode() + frame2.encode() + frame3.encode())

        msg1 = decoder.decode()
        assert isinstance(msg1, RowsResponse) and msg1.has_more

        msg2 = decoder.decode_continuation()
        assert isinstance(msg2, RowsResponse)
        assert msg2.has_more is True
        assert msg2.rows == [[2]]

        msg3 = decoder.decode_continuation()
        assert isinstance(msg3, RowsResponse)
        assert msg3.has_more is False
        assert msg3.rows == [[3]]

        # Decoder should now accept decode() again
        normal = ResultResponse(last_insert_id=0, rows_affected=0)
        decoder.feed(normal.encode())
        msg4 = decoder.decode()
        assert isinstance(msg4, ResultResponse)
