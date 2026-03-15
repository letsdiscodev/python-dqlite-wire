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
    FailureResponse,
    LeaderRequest,
    LeaderResponse,
    OpenRequest,
    PrepareRequest,
    ResultResponse,
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
        decoder = MessageDecoder()
        decoder.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        version = decoder.decode_handshake()
        assert version == PROTOCOL_VERSION

    def test_decode_handshake_partial(self) -> None:
        decoder = MessageDecoder()
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

        decoder = MessageDecoder(is_request=True)
        with pytest.raises(DecodeError, match="[Uu]nsupported schema"):
            decoder.decode_bytes(data)

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

    def test_response_decoder_allows_decode_without_handshake(self) -> None:
        """Response decoders (client-side) don't require inbound handshake."""
        decoder = MessageDecoder(is_request=False)
        msg = LeaderResponse(node_id=1, address="localhost:9001")
        decoder.feed(msg.encode())

        # Should work without calling decode_handshake first
        decoded = decoder.decode()
        assert isinstance(decoded, LeaderResponse)


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
