"""Tests for message codec."""

import pytest

from dqlitewire.codec import (
    MessageDecoder,
    MessageEncoder,
    decode_message,
    encode_message,
)
from dqlitewire.constants import PROTOCOL_VERSION, PROTOCOL_VERSION_LEGACY
from dqlitewire.exceptions import DecodeError, HandshakeError, ProtocolError
from dqlitewire.messages import (
    ClientRequest,
    DbResponse,
    EmptyResponse,
    FailureResponse,
    LeaderRequest,
    LeaderResponse,
    OpenRequest,
    PrepareRequest,
    ResultResponse,
    ServersResponse,
    StmtResponse,
    WelcomeResponse,
)
from dqlitewire.messages.requests import _HeartbeatRequest


class TestMessageEncoder:
    def test_encode_handshake(self) -> None:
        encoder = MessageEncoder()
        handshake = encoder.encode_handshake()
        assert len(handshake) == 8
        assert int.from_bytes(handshake, "little") == PROTOCOL_VERSION

    def test_encode_handshake_legacy(self) -> None:
        encoder = MessageEncoder(version=PROTOCOL_VERSION_LEGACY)
        handshake = encoder.encode_handshake()
        assert len(handshake) == 8
        assert int.from_bytes(handshake, "little") == PROTOCOL_VERSION_LEGACY

    def test_legacy_version_constant_matches_go(self) -> None:
        """PROTOCOL_VERSION_LEGACY must match Go's VersionLegacy = 0x86104dd760433fe5."""
        assert PROTOCOL_VERSION_LEGACY == 0x86104DD760433FE5

    def test_encoder_has_no_buffer_attribute(self) -> None:
        encoder = MessageEncoder()
        assert not hasattr(encoder, "_buffer")

    def test_encoder_rejects_invalid_version(self) -> None:
        with pytest.raises(ProtocolError, match="Unsupported protocol version"):
            MessageEncoder(version=0xDEADBEEF)

    def test_encoder_accepts_legacy_version(self) -> None:
        encoder = MessageEncoder(version=PROTOCOL_VERSION_LEGACY)
        assert encoder.encode_handshake() is not None

    def test_encoder_accepts_default_version(self) -> None:
        encoder = MessageEncoder(version=PROTOCOL_VERSION)
        assert encoder.encode_handshake() is not None

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
    def test_decoder_rejects_invalid_version(self) -> None:
        with pytest.raises(ProtocolError, match="Unsupported protocol version"):
            MessageDecoder(version=0xDEADBEEF)

    def test_decoder_accepts_legacy_version(self) -> None:
        decoder = MessageDecoder(version=PROTOCOL_VERSION_LEGACY)
        assert decoder.version == PROTOCOL_VERSION_LEGACY

    def test_decoder_request_validates_version(self) -> None:
        """Request decoders must reject unsupported versions at construction too:
        __init__ once skipped the check when is_request=True, leaking past validation."""
        with pytest.raises(HandshakeError, match="Unsupported protocol version"):
            MessageDecoder(is_request=True, version=0xBADBEEF)

    def test_decoder_request_accepts_supported_version_at_construction(self) -> None:
        """Request decoders accept supported versions; ``version`` stays None until
        ``decode_handshake()`` populates it from the wire."""
        decoder = MessageDecoder(is_request=True, version=PROTOCOL_VERSION_LEGACY)
        assert decoder.version is None  # not set until handshake

    def test_decode_message_request_rejects_unsupported_version(self) -> None:
        """decode_message(is_request=True) validates version via the same __init__ invariant."""
        with pytest.raises(HandshakeError, match="Unsupported protocol version"):
            decode_message(b"", is_request=True, version=0xBADBEEF)

    def test_decoder_custom_max_message_size(self) -> None:
        import struct

        from dqlitewire.exceptions import DecodeError

        decoder = MessageDecoder(max_message_size=64)
        oversized = struct.pack("<IBBH", 100, 0, 0, 0)  # 808-byte body
        decoder.feed(oversized)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            decoder.decode()

    def test_decoder_default_max_message_size(self) -> None:
        from dqlitewire.buffer import ReadBuffer

        decoder = MessageDecoder()
        assert decoder._buffer._max_message_size == ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE

    def test_decoder_custom_max_rows(self) -> None:
        """max_rows must reach RowsResponse.decode_body: MessageDecoder once omitted it
        at the call site, pinning the 1M default regardless of caller config."""
        from dqlitewire.constants import ValueType
        from dqlitewire.exceptions import DecodeError
        from dqlitewire.messages.responses import RowsResponse

        msg = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]] * 3,
            rows=[[1], [2], [3]],
        )
        encoded = msg.encode()

        decoder = MessageDecoder(max_rows=2)
        decoder.feed(encoded)
        with pytest.raises(DecodeError, match="reached limit 2"):
            decoder.decode()

    def test_decoder_default_max_rows(self) -> None:
        from dqlitewire.messages.responses import RowsResponse

        decoder = MessageDecoder()
        assert decoder._max_rows == RowsResponse.DEFAULT_MAX_ROWS

    def test_decoder_rejects_zero_max_rows(self) -> None:
        with pytest.raises(ValueError, match="max_rows must be >= 1"):
            MessageDecoder(max_rows=0)

    def test_decoder_rejects_zero_max_message_size(self) -> None:
        """max_message_size < 1 rejected at construction; else feed() raises a
        confusing 'projected ... > 0' error on the first byte."""
        with pytest.raises(ValueError, match="max_message_size must be >= 1"):
            MessageDecoder(max_message_size=0)

    def test_decoder_rejects_negative_max_message_size(self) -> None:
        with pytest.raises(ValueError, match="max_message_size must be >= 1"):
            MessageDecoder(max_message_size=-1)

    def test_decoder_rejects_zero_max_continuation_frames(self) -> None:
        """max_continuation_frames < 1 rejected so a cap defaulting to 0 can't
        silently let continuation frames through."""
        with pytest.raises(ValueError, match="max_continuation_frames must be >= 1"):
            MessageDecoder(max_continuation_frames=0)

    def test_decoder_rejects_negative_max_continuation_frames(self) -> None:
        with pytest.raises(ValueError, match="max_continuation_frames must be >= 1"):
            MessageDecoder(max_continuation_frames=-5)

    def test_decoder_rejects_zero_max_total_rows(self) -> None:
        with pytest.raises(ValueError, match="max_total_rows must be >= 1"):
            MessageDecoder(max_total_rows=0)

    def test_decoder_rejects_negative_max_total_rows(self) -> None:
        with pytest.raises(ValueError, match="max_total_rows must be >= 1"):
            MessageDecoder(max_total_rows=-1)

    def test_decoder_continuation_honors_max_rows(self) -> None:
        """decode_continuation honors max_rows per-frame (the cap applies to each
        continuation response, even though a stream may exceed it across frames)."""
        from dqlitewire.constants import ValueType
        from dqlitewire.exceptions import DecodeError
        from dqlitewire.messages.responses import RowsResponse

        cont = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]] * 3,
            rows=[[1], [2], [3]],
            has_more=False,
        )
        cont_bytes = cont.encode()

        decoder = MessageDecoder(max_rows=2)
        # Mark decoder as mid-continuation so decode_continuation() runs.
        decoder._continuation_expected = True
        decoder.feed(cont_bytes)
        with pytest.raises(DecodeError, match="reached limit 2"):
            decoder.decode_continuation()

    def test_decoder_continuation_rejects_total_rows_overflow(self) -> None:
        """Cumulative max_total_rows cap fires across continuation frames — a refactor
        dropping the running total would reopen an unbounded-stream DoS."""
        from dqlitewire.constants import ValueType
        from dqlitewire.exceptions import DecodeError
        from dqlitewire.messages.responses import RowsResponse

        first = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[1]],
            has_more=True,
        )
        second = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[2]],
            has_more=False,
        )
        decoder = MessageDecoder(max_total_rows=1)
        decoder.feed(first.encode())
        # First frame is exactly at the cap; allowed.
        msg = decoder.decode()
        assert isinstance(msg, RowsResponse)
        # Second frame pushes cumulative count over the cap.
        decoder.feed(second.encode())
        with pytest.raises(DecodeError, match="max_total_rows"):
            decoder.decode_continuation()

    def test_decoder_continuation_rejects_too_many_frames(self) -> None:
        """max_continuation_frames cap fires after one frame too many (slow-drip DoS defence)."""
        from dqlitewire.constants import ValueType
        from dqlitewire.exceptions import DecodeError
        from dqlitewire.messages.responses import RowsResponse

        first = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[1]],
            has_more=True,
        )
        next_frame = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[2]],
            has_more=True,
        )
        last_frame = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[3]],
            has_more=False,
        )
        # The initial frame counts as frame 1, so max=2 admits it plus one
        # continuation; the second continuation (third overall) trips the cap.
        decoder = MessageDecoder(max_continuation_frames=2, max_total_rows=100)
        decoder.feed(first.encode())
        decoder.decode()
        decoder.feed(next_frame.encode())
        decoder.decode_continuation()
        decoder.feed(last_frame.encode())
        with pytest.raises(DecodeError, match="max_continuation_frames"):
            decoder.decode_continuation()

    def test_decoder_initial_frame_already_exceeded_max_total_rows(self) -> None:
        """A first ROWS frame already over max_total_rows must fail at decode(),
        not be delayed until decode_continuation()."""
        from dqlitewire.constants import ValueType
        from dqlitewire.exceptions import DecodeError
        from dqlitewire.messages.responses import RowsResponse

        first = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER], [ValueType.INTEGER]],
            rows=[[1], [2]],
            has_more=True,
        )
        decoder = MessageDecoder(max_total_rows=1)
        decoder.feed(first.encode())
        with pytest.raises(DecodeError, match="max_total_rows"):
            decoder.decode()

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
        from dqlitewire.messages.base import Header

        with pytest.raises(DecodeError):
            Header.decode(b"")

    def test_header_roundtrip_minimal(self) -> None:
        from dqlitewire.messages.base import Header

        header = Header(size_words=1, msg_type=0, schema=0)
        data = header.encode()
        decoded = Header.decode(data)
        assert decoded.size_words == 1
        assert decoded.msg_type == 0

    def test_header_encode_overflow_raises_encode_error(self) -> None:
        """size_words over uint32 raises EncodeError at construction (not encode()),
        with a field-name + observed-value diagnostic."""
        from dqlitewire.exceptions import EncodeError
        from dqlitewire.messages.base import Header

        with pytest.raises(EncodeError, match="size_words.*out of range"):
            Header(size_words=2**32, msg_type=0, schema=0)

    def test_header_encode_msg_type_overflow_raises_encode_error(self) -> None:
        from dqlitewire.exceptions import EncodeError
        from dqlitewire.messages.base import Header

        with pytest.raises(EncodeError, match="msg_type.*out of range"):
            Header(size_words=1, msg_type=256, schema=0)

    def test_header_encode_schema_overflow_raises_encode_error(self) -> None:
        from dqlitewire.exceptions import EncodeError
        from dqlitewire.messages.base import Header

        with pytest.raises(EncodeError, match="schema.*out of range"):
            Header(size_words=1, msg_type=0, schema=256)

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
        decoder = MessageDecoder()
        with pytest.raises(DecodeError, match="too short"):
            decoder.decode_bytes(b"\x00" * 7)

    def test_decode_bytes_empty(self) -> None:
        decoder = MessageDecoder()
        with pytest.raises(DecodeError, match="too short"):
            decoder.decode_bytes(b"")


class TestHandshakeStateEnforcement:
    """Verify that request decoders enforce handshake-before-decode ordering."""

    def test_decode_handshake_rejects_unknown_version(self) -> None:
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        decoder.feed(b"\x42\x42\x42\x42\x42\x42\x42\x42")
        with pytest.raises(ProtocolError, match="[Uu]nsupported protocol version"):
            decoder.decode_handshake()

    def test_decode_handshake_failure_does_not_consume_bytes(self) -> None:
        """An unsupported version must NOT consume the handshake bytes: peek-before-
        consume keeps them in the buffer so a retry is deterministic (same bytes,
        same error) instead of misreading the next message as a version."""
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
        """Fewer than 8 bytes buffered: decode_handshake() returns None and leaves
        the partial data so a later feed() can complete the handshake."""
        decoder = MessageDecoder(is_request=True)
        version_bytes = PROTOCOL_VERSION_LEGACY.to_bytes(8, "little")
        decoder.feed(version_bytes[:4])
        assert decoder.decode_handshake() is None
        assert decoder._buffer.available() == 4

        decoder.feed(version_bytes[4:])
        assert decoder.decode_handshake() == PROTOCOL_VERSION_LEGACY

    def test_decode_handshake_failure_preserves_following_bytes(self) -> None:
        """Bug reproducer: bogus version followed by a valid one. Pre-fix code consumed
        both chunks across two retries; the fix consumes neither on the first failure."""
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
        """After a handshake failure, reset() clears the buffer and accepts a fresh handshake."""
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        decoder.feed(b"\x42" * 8)
        with pytest.raises(ProtocolError, match="[Uu]nsupported protocol version"):
            decoder.decode_handshake()

        decoder.reset()
        assert decoder._buffer.available() == 0
        assert not decoder._handshake_done

        decoder.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        assert decoder.decode_handshake() == PROTOCOL_VERSION
        assert decoder._handshake_done

    def test_peek_bytes_does_not_advance_position(self) -> None:
        """ReadBuffer.peek_bytes() returns bytes without advancing _pos (the primitive
        decode_handshake() depends on)."""
        from dqlitewire.buffer import ReadBuffer

        buf = ReadBuffer()
        buf.feed(b"abcdefghij")
        assert buf.peek_bytes(4) == b"abcd"
        assert buf.available() == 10
        assert buf.peek_bytes(4) == b"abcd"
        assert buf.available() == 10
        # More than available returns None.
        assert buf.peek_bytes(20) is None
        assert buf.available() == 10

    def test_decode_handshake_rejects_zero_version(self) -> None:
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
        """A request decoder must reject decode() before decode_handshake(); else the
        8-byte version prefix would be misread as a message header."""
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        msg = LeaderRequest()
        decoder.feed(msg.encode())

        with pytest.raises(ProtocolError, match="[Hh]andshake"):
            decoder.decode()

    def test_request_decoder_allows_decode_after_handshake(self) -> None:
        decoder = MessageDecoder(is_request=True)
        handshake = PROTOCOL_VERSION.to_bytes(8, "little")
        msg = LeaderRequest()
        decoder.feed(handshake + msg.encode())

        version = decoder.decode_handshake()
        assert version == PROTOCOL_VERSION

        decoded = decoder.decode()
        assert isinstance(decoded, LeaderRequest)

    def test_request_decoder_decode_bytes_rejects_before_handshake(self) -> None:
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        msg = LeaderRequest()
        with pytest.raises(ProtocolError, match="[Hh]andshake"):
            decoder.decode_bytes(msg.encode())

    def test_double_decode_handshake_raises(self) -> None:
        """A second decode_handshake() must raise, not consume 8 message bytes as a
        version number (silent stream corruption)."""
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=True)
        handshake = PROTOCOL_VERSION.to_bytes(8, "little")
        msg = LeaderRequest()
        decoder.feed(handshake + msg.encode())

        version = decoder.decode_handshake()
        assert version == PROTOCOL_VERSION

        with pytest.raises(ProtocolError, match="[Hh]andshake already completed"):
            decoder.decode_handshake()

        # Message still decodable — the rejected second handshake consumed nothing.
        decoded = decoder.decode()
        assert isinstance(decoded, LeaderRequest)

    def test_decode_handshake_on_client_decoder_raises(self) -> None:
        """Client-side decoder rejects decode_handshake() (_handshake_done starts True)."""
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        decoder.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        with pytest.raises(ProtocolError, match="[Hh]andshake already completed"):
            decoder.decode_handshake()

    def test_response_decoder_decode_bytes_works_without_handshake(self) -> None:
        decoder = MessageDecoder(is_request=False)
        msg = LeaderResponse(node_id=1, address="localhost:9001")
        decoded = decoder.decode_bytes(msg.encode())
        assert isinstance(decoded, LeaderResponse)

    def test_response_decoder_allows_decode_without_handshake(self) -> None:
        """Response decoders (client-side) don't require an inbound handshake."""
        decoder = MessageDecoder(is_request=False)
        msg = LeaderResponse(node_id=1, address="localhost:9001")
        decoder.feed(msg.encode())

        decoded = decoder.decode()
        assert isinstance(decoded, LeaderResponse)

    def test_legacy_handshake_decodes_leader_response_as_legacy(self) -> None:
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
        from dqlitewire.codec import decode_message

        msg = LeaderResponse(node_id=42, address="node1:9001")

        decoded = decode_message(msg.encode(), is_request=False, version=PROTOCOL_VERSION)
        assert isinstance(decoded, LeaderResponse)
        assert decoded.node_id == 42
        assert decoded.address == "node1:9001"

    def test_decoder_version_property_request(self) -> None:
        decoder = MessageDecoder(is_request=True)
        assert decoder.version is None
        decoder.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        decoder.decode_handshake()
        assert decoder.version == PROTOCOL_VERSION

    def test_client_decoder_with_version_parameter(self) -> None:
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

        decoder = MessageDecoder(is_request=False, version=PROTOCOL_VERSION_LEGACY)
        assert decoder.version == PROTOCOL_VERSION_LEGACY
        decoder.feed(data)
        decoded = decoder.decode()
        assert isinstance(decoded, LeaderResponse)
        assert decoded.node_id == 0
        assert decoded.address == address

    def test_client_decoder_default_version_is_modern(self) -> None:
        decoder = MessageDecoder(is_request=False)
        assert decoder.version == PROTOCOL_VERSION

    def test_client_decoder_modern_version_decodes_modern_leader(self) -> None:
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

    def test_rows_response(self) -> None:
        from dqlitewire.constants import ValueType
        from dqlitewire.messages.responses import RowsResponse

        original = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            row_types=[
                [ValueType.INTEGER, ValueType.TEXT],
                [ValueType.INTEGER, ValueType.TEXT],
            ],
            rows=[[1, "alice"], [2, "bob"]],
        )
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=False)
        assert isinstance(decoded, RowsResponse)
        assert decoded.column_names == ["id", "name"]
        assert decoded.rows == [[1, "alice"], [2, "bob"]]
        assert decoded.has_more is False

    def test_prepare_request_schema_survives_roundtrip(self) -> None:
        original = PrepareRequest(db_id=1, sql="SELECT 1", schema=1)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, PrepareRequest)
        assert decoded.schema == 1

    def test_stmt_response_v1_through_codec(self) -> None:
        msg = StmtResponse(db_id=1, stmt_id=2, num_params=3, tail_offset=42)
        encoded = msg.encode()

        decoder = MessageDecoder(is_request=False)
        decoder.feed(encoded)
        decoded = decoder.decode()

        assert isinstance(decoded, StmtResponse)
        assert decoded.db_id == 1
        assert decoded.stmt_id == 2
        assert decoded.num_params == 3
        assert decoded.tail_offset == 42

    def test_stmt_response_v1_through_decode_message(self) -> None:
        msg = StmtResponse(db_id=1, stmt_id=2, num_params=3, tail_offset=42)
        encoded = msg.encode()
        decoded = decode_message(encoded)
        assert isinstance(decoded, StmtResponse)
        assert decoded.tail_offset == 42

    def test_decode_bytes_rejects_envelope_trailing_bytes(self) -> None:
        """decode_bytes() rejects envelope trailing bytes beyond the declared body size:
        the envelope strip once silently sliced ``header + body + garbage``, masking
        malformed input (strict-decode parity with the per-message checks)."""
        from dqlitewire.messages.base import Header
        from dqlitewire.types import encode_uint64

        # Build a Promote message (1 word body: just node_id)
        body = encode_uint64(42)  # node_id=42, no role
        header = Header(size_words=1, msg_type=13, schema=0)  # type 13 = ASSIGN

        # Append garbage trailing bytes that look like a role field
        trailing = encode_uint64(99)
        data = header.encode() + body + trailing

        with pytest.raises(DecodeError, match="trailing"):
            decode_message(data, is_request=True)

    def test_decode_bytes_rejects_short_body(self) -> None:
        from dqlitewire.messages.base import Header

        # Header claims 2 words (16 bytes) but only 8 follow.
        header = Header(size_words=2, msg_type=0, schema=0)
        data = header.encode() + b"\x00" * 8
        with pytest.raises(DecodeError, match="[Bb]ody.*short"):
            decode_message(data, is_request=True)

    def test_heartbeat_request_rejected_by_decode_message(self) -> None:
        """Type-2 (HEARTBEAT) is absent from REQUEST_TYPES — upstream C falls through to
        DQLITE_PARSE for it — so a synthesised frame must hit the unknown-type error."""
        from dqlitewire.codec import REQUEST_TYPES
        from dqlitewire.constants import RequestType

        assert RequestType.HEARTBEAT not in REQUEST_TYPES

        encoded = encode_message(_HeartbeatRequest(timestamp=1710000000))
        with pytest.raises(DecodeError, match="Unknown message type"):
            decode_message(encoded, is_request=True)

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

    def test_connect_request_rejected_by_public_dispatch(self) -> None:
        """CONNECT (type 11) is a Raft-transport frame the public dispatcher rejects as
        unknown (gateway.c DQLITE_PARSE fallthrough); _ConnectRequest stays private."""
        from dqlitewire.messages.requests import _ConnectRequest

        original = _ConnectRequest(node_id=5, address="10.0.0.1:9001")
        encoded = encode_message(original)
        with pytest.raises(DecodeError, match="Unknown message type"):
            decode_message(encoded, is_request=True)

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
        from dqlitewire.constants import NodeRole
        from dqlitewire.messages.responses import NodeInfo

        original = ServersResponse(
            nodes=[
                NodeInfo(1, "n1:9001", NodeRole.VOTER),
                NodeInfo(2, "n2:9001", NodeRole.STANDBY),
            ]
        )
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
        original = ClientRequest(client_id=2**64 - 1)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, ClientRequest)
        assert decoded.client_id == 2**64 - 1

    def test_stmt_response_v0_with_trailing_data_rejected(self) -> None:
        """StmtResponse V0 with trailing bytes is rejected outright: a 24-byte schema=0
        body is invalid since Go/C never pad fixed-body responses (strict-length)."""
        from dqlitewire.exceptions import DecodeError
        from dqlitewire.messages.base import Header
        from dqlitewire.types import encode_uint32, encode_uint64

        body = encode_uint32(1) + encode_uint32(2) + encode_uint64(3) + b"\x00" * 8
        header = Header(size_words=3, msg_type=5, schema=0)
        data = header.encode() + body
        with pytest.raises(DecodeError, match=r"schema=0 body must be exactly 16 bytes"):
            decode_message(data, is_request=False)


class TestDecoderContinuation:
    """Test decode_continuation() for multi-part ROWS responses."""

    def test_decode_continuation_exists(self) -> None:
        decoder = MessageDecoder(is_request=False)
        assert hasattr(decoder, "decode_continuation")

    def test_decode_continuation_roundtrip(self) -> None:
        from dqlitewire.constants import ROW_DONE_MARKER, ROW_PART_MARKER, ValueType
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import RowsResponse
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        types = [ValueType.INTEGER, ValueType.TEXT]

        # Initial ROWS message with PART marker.
        body1 = encode_uint64(2)  # column_count
        body1 += encode_text("id") + encode_text("name")
        body1 += encode_row_header(types)
        body1 += encode_row_values([1, "alice"], types)
        body1 += encode_uint64(ROW_PART_MARKER)
        header1 = Header(size_words=len(body1) // 8, msg_type=7, schema=0)
        msg1_bytes = header1.encode() + body1

        # Continuation with DONE marker; C server always re-sends column_count + names.
        body2 = encode_uint64(2)
        body2 += encode_text("id") + encode_text("name")
        body2 += encode_row_header(types)
        body2 += encode_row_values([2, "bob"], types)
        body2 += encode_uint64(ROW_DONE_MARKER)
        header2 = Header(size_words=len(body2) // 8, msg_type=7, schema=0)
        msg2_bytes = header2.encode() + body2

        decoder = MessageDecoder(is_request=False)
        decoder.feed(msg1_bytes + msg2_bytes)

        initial = decoder.decode()
        assert isinstance(initial, RowsResponse)
        assert initial.has_more is True
        assert len(initial.rows) == 1
        assert initial.rows[0] == [1, "alice"]

        continuation = decoder.decode_continuation()
        assert isinstance(continuation, RowsResponse)
        assert continuation.has_more is False
        assert len(continuation.rows) == 1
        assert continuation.rows[0] == [2, "bob"]

    def test_decode_continuation_with_column_header(self) -> None:
        """decode_continuation handles continuation frames that carry column_count +
        column_names (the C server uses the same layout as the initial frame)."""
        from dqlitewire.constants import ROW_DONE_MARKER, ROW_PART_MARKER, ValueType
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import RowsResponse
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        types = [ValueType.INTEGER, ValueType.TEXT]

        body1 = encode_uint64(2)
        body1 += encode_text("id") + encode_text("name")
        body1 += encode_row_header(types)
        body1 += encode_row_values([1, "alice"], types)
        body1 += encode_uint64(ROW_PART_MARKER)
        h1 = Header(size_words=len(body1) // 8, msg_type=7, schema=0)

        # Continuation with column header, matching C server output.
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
        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        result = decoder.decode_continuation()
        assert result is None

    def test_decode_continuation_raises_on_failure_response(self) -> None:
        """A mid-continuation server error surfaces as ServerFailure with structured
        code/message — not DecodeError (misread body) or bare ProtocolError — so
        callers can tell a recoverable server error from fatal stream desync."""
        from dqlitewire.exceptions import ServerFailure
        from dqlitewire.messages.responses import FailureResponse

        failure = FailureResponse(code=266, message="disk I/O error")
        failure_bytes = failure.encode()

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(failure_bytes)

        with pytest.raises(ServerFailure, match="disk I/O error") as exc_info:
            decoder.decode_continuation()
        assert exc_info.value.code == 266
        assert exc_info.value.message == "disk I/O error"

    def test_decode_continuation_raises_on_unexpected_type(self) -> None:
        from dqlitewire.exceptions import ProtocolError

        result = ResultResponse(last_insert_id=0, rows_affected=0)
        result_bytes = result.encode()

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(result_bytes)

        with pytest.raises(ProtocolError, match="Expected ROWS continuation"):
            decoder.decode_continuation()

    def test_decode_continuation_accepts_empty_response_as_terminator(self) -> None:
        """A mid-stream EmptyResponse is a clean terminator (must not poison): upstream
        emits it instead of a final ROWS frame when an INTERRUPT cancels the query."""
        from dqlitewire.messages.responses import EmptyResponse

        empty_bytes = EmptyResponse().encode()

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(empty_bytes)

        result = decoder.decode_continuation()
        assert isinstance(result, EmptyResponse)
        assert decoder._continuation_expected is False
        assert not decoder._buffer.is_poisoned

    def test_decode_continuation_empty_after_partial_rows(self) -> None:
        """Interrupt-mid-stream: ROWS-PART then an EmptyResponse instead of ROWS-DONE
        clears the continuation flag without poisoning."""
        from dqlitewire.constants import ROW_PART_MARKER, ValueType
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import EmptyResponse, RowsResponse
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        types = [ValueType.INTEGER]
        body = encode_uint64(1)  # column_count
        body += encode_text("id")
        body += encode_row_header(types)
        body += encode_row_values([1], types)
        body += encode_uint64(ROW_PART_MARKER)
        rows_header = Header(size_words=len(body) // 8, msg_type=7, schema=0)
        rows_bytes = rows_header.encode() + body

        empty_bytes = EmptyResponse().encode()

        decoder = MessageDecoder(is_request=False)
        decoder.feed(rows_bytes + empty_bytes)

        first = decoder.decode()
        assert isinstance(first, RowsResponse)
        assert first.has_more is True

        terminator = decoder.decode_continuation()
        assert isinstance(terminator, EmptyResponse)
        assert decoder._continuation_expected is False
        assert not decoder._buffer.is_poisoned

    def test_decode_continuation_clean_failure_does_not_poison(self) -> None:
        """A well-formed mid-continuation FailureResponse surfaces as ServerFailure
        without poisoning: the buffer stays wire-coherent, so later requests decode
        without reset(). Upstream query_work_done can emit FAILURE after ROWS_PART."""
        from dqlitewire.exceptions import ServerFailure
        from dqlitewire.messages.responses import FailureResponse

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True

        failure_bytes = FailureResponse(code=19, message="CHECK constraint failed").encode()
        decoder.feed(failure_bytes)

        with pytest.raises(ServerFailure) as exc_info:
            decoder.decode_continuation()
        assert exc_info.value.code == 19
        assert "CHECK constraint" in exc_info.value.message
        assert decoder.is_poisoned is False
        assert decoder._continuation_expected is False

        # A subsequent message decodes cleanly through the same buffer.
        from dqlitewire.messages.responses import LeaderResponse

        leader_bytes = LeaderResponse(node_id=1, address="127.0.0.1:9001").encode()
        decoder.feed(leader_bytes)
        msg = decoder.decode()
        assert isinstance(msg, LeaderResponse)
        assert msg.node_id == 1
        assert msg.address == "127.0.0.1:9001"

    def test_decode_continuation_malformed_failure_body_does_not_poison(self) -> None:
        """A malformed FAILURE body raises DecodeError without poisoning: read_message
        already advanced past the frame, so the offset stays wire-coherent (same
        non-poisoning discipline as the clean-failure ServerFailure path)."""
        import struct as _struct

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.messages.base import Header

        # 8-byte code, then a bareword with no NUL — decode_text raises on the missing NUL.
        body = _struct.pack("<Q", 1) + b"abcdefgh"
        header = Header(size_words=len(body) // 8, msg_type=0, schema=0)
        bad_failure = header.encode() + body

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(bad_failure)

        with pytest.raises(DecodeError):
            decoder.decode_continuation()
        assert decoder.is_poisoned is False

    def test_decode_continuation_empty_with_non_zero_reserved_does_not_poison(self) -> None:
        """EmptyResponse.decode_body permissively accepts a non-zero reserved word
        (Go parity); only a length mismatch poisons, not the reserved value."""
        import struct

        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import EmptyResponse

        # Well-framed EmptyResponse with a non-zero reserved word.
        body = struct.pack("<Q", 0xDEADBEEFDEADBEEF)
        header = Header(size_words=1, msg_type=8, schema=0)
        empty_with_garbage = header.encode() + body

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(empty_with_garbage)

        msg = decoder.decode_continuation()
        assert isinstance(msg, EmptyResponse)
        assert decoder.is_poisoned is False

    def test_decode_continuation_malformed_empty_size_does_not_poison(self) -> None:
        """A wrong-length EmptyResponse (16 bytes vs 8) raises DecodeError without
        poisoning: read_message already advanced past the frame (coherent offset).
        The reserved field is permissive on decode, so a length mismatch is used."""
        import struct

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.messages.base import Header

        # EmptyResponse body must be exactly 8 bytes; provide 16 to trip
        # the strict-length check.
        body = struct.pack("<QQ", 0, 0)
        header = Header(size_words=2, msg_type=8, schema=0)  # ResponseType.EMPTY = 8
        empty_bad = header.encode() + body

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(empty_bad)

        with pytest.raises(DecodeError, match="EmptyResponse body must be exactly 8 bytes"):
            decoder.decode_continuation()
        assert decoder.is_poisoned is False


class TestDecoderContinuationExpected:
    """After decode() returns a has_more=True RowsResponse, the _continuation_expected
    flag makes a second decode() (instead of decode_continuation) fail loudly rather
    than misparse the headerless continuation frame as a new message."""

    def test_decode_raises_when_continuation_expected(self) -> None:
        """decode() must refuse while a ROWS continuation is in progress."""
        from dqlitewire.constants import ROW_PART_MARKER, ValueType
        from dqlitewire.exceptions import ProtocolError
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import RowsResponse
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        types = [ValueType.INTEGER]

        body = encode_uint64(1)  # column_count
        body += encode_text("id")
        body += encode_row_header(types)
        body += encode_row_values([1], types)
        body += encode_uint64(ROW_PART_MARKER)
        header = Header(size_words=len(body) // 8, msg_type=7, schema=0)
        msg_bytes = header.encode() + body

        # A standalone second message decode() would wrongly read mid-continuation.
        second = ResultResponse(last_insert_id=0, rows_affected=0).encode()

        decoder = MessageDecoder(is_request=False)
        decoder.feed(msg_bytes + second)

        result = decoder.decode()
        assert isinstance(result, RowsResponse)
        assert result.has_more is True

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

        body1 = encode_uint64(1)
        body1 += encode_text("id")
        body1 += encode_row_header(types)
        body1 += encode_row_values([1], types)
        body1 += encode_uint64(ROW_PART_MARKER)
        h1 = Header(size_words=len(body1) // 8, msg_type=7, schema=0)

        body2 = encode_uint64(1)
        body2 += encode_text("id")
        body2 += encode_row_header(types)
        body2 += encode_row_values([2], types)
        body2 += encode_uint64(ROW_DONE_MARKER)
        h2 = Header(size_words=len(body2) // 8, msg_type=7, schema=0)

        normal = ResultResponse(last_insert_id=5, rows_affected=3).encode()

        decoder = MessageDecoder(is_request=False)
        decoder.feed(h1.encode() + body1 + h2.encode() + body2 + normal)

        initial = decoder.decode()
        assert isinstance(initial, RowsResponse) and initial.has_more

        cont = decoder.decode_continuation()
        assert isinstance(cont, RowsResponse) and not cont.has_more

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

        decoder.reset()
        normal = ResultResponse(last_insert_id=0, rows_affected=0).encode()
        decoder.feed(normal)
        msg = decoder.decode()
        assert isinstance(msg, ResultResponse)

    def test_has_more_false_does_not_set_flag(self) -> None:
        from dqlitewire.constants import ValueType
        from dqlitewire.messages.responses import RowsResponse

        decoder = MessageDecoder(is_request=False)
        msg = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            has_more=False,
        )
        normal = ResultResponse(last_insert_id=0, rows_affected=0).encode()
        decoder.feed(msg.encode() + normal)

        result = decoder.decode()
        assert isinstance(result, RowsResponse) and not result.has_more

        result2 = decoder.decode()
        assert isinstance(result2, ResultResponse)


class TestContinuationFlagCompleteness:
    """State-machine completeness: decode_continuation() must refuse when the flag is
    False, and skip_message() must refuse when it is True."""

    def test_decode_continuation_raises_when_not_expected(self) -> None:
        """decode_continuation() must refuse with no continuation in progress; else it
        would silently consume and misparse the next message."""
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        result = ResultResponse(last_insert_id=0, rows_affected=0)
        decoder.feed(result.encode())

        with pytest.raises(ProtocolError, match="no ROWS continuation"):
            decoder.decode_continuation()

        assert decoder.has_message(), (
            "decode_continuation must not consume the message when the guard fires"
        )
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

    def test_skip_message_during_continuation_reset_recovery(self) -> None:
        """reset() is the documented recovery after skip_message() is rejected
        mid-continuation; the decoder then accepts a fresh top-level message."""
        from dqlitewire.constants import ROW_PART_MARKER, ValueType
        from dqlitewire.exceptions import ProtocolError
        from dqlitewire.messages.base import Header
        from dqlitewire.messages.responses import ResultResponse, RowsResponse
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
        first = decoder.decode()
        assert isinstance(first, RowsResponse) and first.has_more

        with pytest.raises(ProtocolError, match="continuation"):
            decoder.skip_message()

        decoder.reset()

        recovered = ResultResponse(last_insert_id=0, rows_affected=0)
        decoder.feed(recovered.encode())
        msg = decoder.decode()
        assert isinstance(msg, ResultResponse)
        assert msg.last_insert_id == 0
        assert msg.rows_affected == 0


class TestDecoderSkipMessage:
    """Test skip_message() and is_skipping on MessageDecoder."""

    def test_skip_message_exists_on_decoder(self) -> None:
        decoder = MessageDecoder(is_request=False)
        assert hasattr(decoder, "skip_message")
        assert hasattr(decoder, "is_skipping")

    def test_skip_oversized_poisons_after_capped_skip(self) -> None:
        """After a capped oversized skip, buffer is poisoned."""
        import struct

        from dqlitewire.exceptions import DecodeError, ProtocolError

        decoder = MessageDecoder(is_request=False)
        decoder._buffer._max_message_size = 128

        # size_words=100 → 808 bytes total, over the 128 cap.
        oversized_header = struct.pack("<IBBH", 100, 0, 0, 0)
        decoder.feed(oversized_header)

        assert decoder.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            decoder.decode()

        result = decoder.skip_message()
        assert result is False
        assert decoder.is_skipping is True

        decoder.feed(b"\x00" * decoder._buffer._skip_remaining)

        # Capped skip completed — stream is desynchronized, so the buffer is poisoned.
        assert decoder.is_skipping is False
        assert decoder.is_poisoned is True

        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()


class TestDecoderPoisonedState:
    """After a mid-stream error from already-consumed bytes, _pos is unknown, so all
    operations must fail fast with a poisoned-state error until reset()."""

    def test_decode_poisons_on_unknown_message_type(self) -> None:
        """An unknown message type must poison the decoder so retries fail loudly."""
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

        # Poison-gate contract: every consuming entry point, including feed(),
        # refuses on a poisoned buffer; recovery requires reset().
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.feed(struct.pack("<IBBH", 0, 0xFE, 0, 0))
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()

    def test_reset_unpoisons_decoder(self) -> None:
        """reset() clears buffer state and the poison flag for reuse after a reconnect."""
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

    def test_oversized_header_does_not_poison_before_skip(self) -> None:
        """An oversized-header error must NOT poison before skip_message() — the bytes
        aren't consumed yet. After a capped skip completes, the buffer IS poisoned."""
        import struct

        decoder = MessageDecoder(is_request=False)
        decoder._buffer._max_message_size = 128
        oversized = struct.pack("<IBBH", 100, 0, 0, 0)  # 808-byte body > 128
        decoder.feed(oversized)

        with pytest.raises(DecodeError, match="exceeds maximum"):
            decoder.decode()
        assert decoder.is_poisoned is False

        decoder.skip_message()
        decoder.feed(b"\x00" * decoder._buffer._skip_remaining)
        assert decoder.is_poisoned is True

    def test_poison_catches_non_protocol_exceptions(self) -> None:
        """Any exception from decode_bytes (struct.error, ValueError, ...) must poison:
        the bytes are consumed and the offset is unknown, so catching only protocol
        types would leave the buffer desynchronized."""
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        msg = LeaderResponse(node_id=1, address="x:1").encode()
        decoder.feed(msg)

        # Simulate a non-protocol exception from a real decode_body.
        sentinel = ValueError("simulated body parse failure")

        def boom(_data: bytes) -> object:
            raise sentinel

        decoder.decode_bytes = boom  # type: ignore[assignment]

        with pytest.raises(ValueError, match="simulated body parse failure"):
            decoder.decode()
        assert decoder.is_poisoned is True

        # Restore real decode_bytes; the poison check fires before it runs, and
        # feed() on a poisoned buffer also raises (no reset done).
        del decoder.decode_bytes
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.feed(msg)
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()

    def test_decode_continuation_raises_streamerror_on_wrong_type_without_poison(
        self,
    ) -> None:
        """An unexpected-type continuation frame (not ROWS/FAILURE/EMPTY) raises
        StreamError without poisoning: read_message left a coherent offset, so one
        bad frame from a peer must not kill the decoder (Go Protocol.Recv parity)."""
        from dqlitewire.exceptions import StreamError

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        result = ResultResponse(last_insert_id=0, rows_affected=0).encode()
        decoder.feed(result)

        with pytest.raises(StreamError, match="Expected ROWS continuation"):
            decoder.decode_continuation()
        # Load-bearing: not poisoned, so the next legitimate frame still decodes.
        assert decoder.is_poisoned is False
        assert decoder._continuation_expected is False

        # Now-empty buffer returns None, not poisoned.
        assert decoder.decode() is None

        msg = LeaderResponse(node_id=1, address="x:1").encode()
        decoder.feed(msg)
        decoded = decoder.decode()
        assert isinstance(decoded, LeaderResponse)
        assert decoded.node_id == 1

    def test_decode_bytes_honors_poison(self) -> None:
        """decode_bytes must honor the poison flag too: it's a public parse entry point,
        and a user "resuming" by feeding raw bytes to it would bypass the poison gate."""
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        decoder._buffer.poison(DecodeError("original cause"))

        msg = LeaderResponse(node_id=1, address="x:1").encode()
        with pytest.raises(ProtocolError, match="poisoned") as ei:
            decoder.decode_bytes(msg)
        assert isinstance(ei.value.__cause__, DecodeError)

        # decode_message() builds a fresh decoder per call, so it still works.
        from dqlitewire.codec import decode_message

        fresh = decode_message(msg, is_request=False)
        assert isinstance(fresh, LeaderResponse)
        assert fresh.node_id == 1

    def test_poison_is_first_error_wins(self) -> None:
        """ReadBuffer.poison() keeps the first error so the original __cause__ stays visible."""
        from dqlitewire.buffer import ReadBuffer

        buf = ReadBuffer()
        first = DecodeError("first failure")
        second = DecodeError("second failure")
        buf.poison(first)
        buf.poison(second)
        assert buf._poisoned is first

    def test_has_message_true_then_decode_raises_and_poisons_on_reserved(self) -> None:
        """has_message() inspects only the size_words prefix, so a non-zero reserved
        field returns True; decode() then surfaces the DecodeError and poisons."""
        import struct

        from dqlitewire.constants import ResponseType
        from dqlitewire.exceptions import PoisonedError, ProtocolError

        decoder = MessageDecoder(is_request=False)
        # size_words=1, type=FAILURE, schema=0, reserved=0xBEEF (non-zero).
        header = struct.pack("<IBBH", 1, ResponseType.FAILURE, 0, 0xBEEF)
        body = b"\x00" * 8
        decoder.feed(header + body)

        assert decoder.has_message() is True
        assert decoder.is_poisoned is False
        with pytest.raises(DecodeError, match="reserved field must be 0"):
            decoder.decode()
        assert decoder.is_poisoned is True

        # Poisoned buffer raises the poison error, not a re-run of the header parse.
        with pytest.raises((PoisonedError, ProtocolError), match="poisoned"):
            decoder.decode()

    def test_request_decoder_reset_clears_handshake_state(self) -> None:
        """A request decoder's reset() must also undo the handshake; else a reconnect
        would silently accept message bytes as a continuation of the dead session."""
        decoder = MessageDecoder(is_request=True)
        decoder.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        decoder.decode_handshake()
        assert decoder._handshake_done is True
        assert decoder.version == PROTOCOL_VERSION

        decoder.reset()
        assert decoder._handshake_done is False
        assert decoder.version is None
        # Must refuse decode() until a fresh handshake.
        from dqlitewire.exceptions import ProtocolError

        with pytest.raises(ProtocolError, match="[Hh]andshake"):
            decoder.decode()


class TestDecoderPoisonOnMalformedBody:
    """decode() poisons on a malformed body, not just on unknown types or oversized headers."""

    def test_decode_poisons_on_truncated_rows_body(self) -> None:
        """A ROWS message with a valid header but a body too short for column_count poisons."""
        import struct

        from dqlitewire.constants import ResponseType
        from dqlitewire.exceptions import ProtocolError

        # column_count=1 but no column name follows — decode_body fails.
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

        normal = ResultResponse(last_insert_id=0, rows_affected=0)
        decoder.feed(normal.encode())
        msg3 = decoder.decode()
        assert isinstance(msg3, ResultResponse)

    def test_failure_during_continuation_does_not_poison(self) -> None:
        """A FailureResponse instead of a continuation is a clean signal: not poisoned,
        connection stays usable. Upstream query_work_done hits this on mid-stream
        I/O/constraint failures after PART rows — users see OperationalError, not a reconnect."""
        from dqlitewire.constants import ValueType
        from dqlitewire.exceptions import ServerFailure
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

        with pytest.raises(ServerFailure, match="disk I/O error"):
            decoder.decode_continuation()

        assert not decoder.is_poisoned

    def test_failure_during_continuation_clears_flag(self) -> None:
        """_continuation_expected must be False after a FailureResponse."""
        from dqlitewire.constants import ValueType
        from dqlitewire.exceptions import ServerFailure
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

        with pytest.raises(ServerFailure, match="disk I/O error"):
            decoder.decode_continuation()

        assert not decoder._continuation_expected

    def test_chained_continuations_part_part_done(self) -> None:
        """Three frames: initial(PART) + cont(PART) + cont(DONE)."""
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

        normal = ResultResponse(last_insert_id=0, rows_affected=0)
        decoder.feed(normal.encode())
        msg4 = decoder.decode()
        assert isinstance(msg4, ResultResponse)

    def test_decode_continuation_rejects_unsupported_schema(self) -> None:
        """decode_continuation must validate schema like decode_bytes."""
        import struct

        from dqlitewire.constants import ValueType
        from dqlitewire.messages.responses import RowsResponse

        initial = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            has_more=True,
        )
        # Continuation with schema=1, unsupported for ROWS.
        cont_body = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[2]],
            has_more=False,
        ).encode_body()
        from dqlitewire.constants import ResponseType

        size_words = len(cont_body) // 8
        bad_header = struct.pack("<IBBH", size_words, ResponseType.ROWS, 1, 0)
        bad_frame = bad_header + cont_body

        decoder = MessageDecoder(is_request=False)
        decoder.feed(initial.encode() + bad_frame)

        msg1 = decoder.decode()
        assert isinstance(msg1, RowsResponse) and msg1.has_more

        with pytest.raises(DecodeError, match="Unsupported schema version"):
            decoder.decode_continuation()

    def test_oversized_continuation_frame_clears_flag_and_poisons(self) -> None:
        import struct

        from dqlitewire.constants import ValueType
        from dqlitewire.exceptions import ProtocolError
        from dqlitewire.messages.responses import RowsResponse

        initial = RowsResponse(
            column_names=["id"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            has_more=True,
        )

        decoder = MessageDecoder(is_request=False)
        decoder._buffer._max_message_size = 128

        # Oversized continuation header: 200 words = 1600 bytes > 128
        oversized_header = struct.pack("<IBBH", 200, 7, 0, 0)  # type 7 = ROWS

        decoder.feed(initial.encode() + oversized_header)

        msg = decoder.decode()
        assert isinstance(msg, RowsResponse) and msg.has_more

        with pytest.raises(DecodeError, match="exceeds maximum"):
            decoder.decode_continuation()

        assert not decoder._continuation_expected
        assert decoder.is_poisoned

        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()


class TestEndToEndPipeline:
    """End-to-end tests exercising the full MessageEncoder + MessageDecoder pipeline."""

    def test_handshake_then_request_response_roundtrip(self) -> None:
        """Full stateful pipeline: handshake → request → response."""
        encoder = MessageEncoder()
        client_decoder = MessageDecoder(is_request=False)
        server_decoder = MessageDecoder(is_request=True)

        handshake_bytes = encoder.encode_handshake()

        server_decoder.feed(handshake_bytes)
        version = server_decoder.decode_handshake()
        assert version == PROTOCOL_VERSION

        request = LeaderRequest()
        request_bytes = encoder.encode(request)

        server_decoder.feed(request_bytes)
        assert server_decoder.has_message()
        decoded_request = server_decoder.decode()
        assert isinstance(decoded_request, LeaderRequest)

        response = LeaderResponse(node_id=1, address="127.0.0.1:9001")
        response_bytes = response.encode()

        client_decoder.feed(response_bytes)
        assert client_decoder.has_message()
        decoded_response = client_decoder.decode()
        assert isinstance(decoded_response, LeaderResponse)
        assert decoded_response.node_id == 1
        assert decoded_response.address == "127.0.0.1:9001"

    def test_multiple_messages_single_feed(self) -> None:
        decoder = MessageDecoder(is_request=False)

        msg1 = WelcomeResponse(heartbeat_timeout=15000000000)
        msg2 = DbResponse(db_id=0)
        msg3 = StmtResponse(db_id=0, stmt_id=1, num_params=2)

        all_bytes = msg1.encode() + msg2.encode() + msg3.encode()
        decoder.feed(all_bytes)

        decoded1 = decoder.decode()
        assert isinstance(decoded1, WelcomeResponse)
        assert decoded1.heartbeat_timeout == 15000000000

        decoded2 = decoder.decode()
        assert isinstance(decoded2, DbResponse)
        assert decoded2.db_id == 0

        decoded3 = decoder.decode()
        assert isinstance(decoded3, StmtResponse)
        assert decoded3.db_id == 0
        assert decoded3.stmt_id == 1
        assert decoded3.num_params == 2

        assert not decoder.has_message()

    def test_partial_feed_chunked(self) -> None:
        decoder = MessageDecoder(is_request=False)
        msg = ResultResponse(last_insert_id=42, rows_affected=7)
        encoded = msg.encode()

        for i in range(len(encoded) - 1):
            decoder.feed(encoded[i : i + 1])
            assert not decoder.has_message(), f"should not be complete at byte {i}"

        decoder.feed(encoded[-1:])
        assert decoder.has_message()

        decoded = decoder.decode()
        assert isinstance(decoded, ResultResponse)
        assert decoded.last_insert_id == 42
        assert decoded.rows_affected == 7


class TestReadmeExample:
    """Verify the README usage example actually works."""

    def test_readme_example_runs_without_error(self) -> None:
        """The README shows encode_message / decode_message with a LeaderRequest."""
        from dqlitewire import decode_message, encode_message
        from dqlitewire.messages import LeaderRequest

        data = encode_message(LeaderRequest())
        message = decode_message(data, is_request=True)
        assert isinstance(message, LeaderRequest)


class TestPicklePrevention:
    """Core classes must not be picklable."""

    def test_message_decoder_pickle_raises(self) -> None:
        import pickle

        with pytest.raises(TypeError, match="cannot pickle"):
            pickle.dumps(MessageDecoder())

    def test_message_encoder_pickle_raises(self) -> None:
        import pickle

        with pytest.raises(TypeError, match="cannot pickle"):
            pickle.dumps(MessageEncoder())

    def test_message_decoder_pickle_message_does_not_claim_connection_binding(
        self,
    ) -> None:
        """The rejection message must cite the ReadBuffer stream state, not invent a
        "single connection" binding the class does not hold."""
        import pickle

        try:
            pickle.dumps(MessageDecoder())
        except TypeError as e:
            msg = str(e)
        else:
            pytest.fail("expected TypeError")
        assert "single connection" not in msg, (
            f"rejection message must not claim a connection binding "
            f"the class does not hold; got: {msg!r}"
        )
        assert "MessageDecoder" in msg

    def test_message_encoder_pickle_message_aligns_with_stateless_docstring(
        self,
    ) -> None:
        """The rejection message must not claim a "single connection" binding, contradicting
        the class's "effectively stateless" docstring (vars is just {'_version': int})."""
        import pickle

        try:
            pickle.dumps(MessageEncoder())
        except TypeError as e:
            msg = str(e)
        else:
            pytest.fail("expected TypeError")
        assert "single connection" not in msg, (
            f"rejection message must not contradict the class's "
            f"'effectively stateless' docstring; got: {msg!r}"
        )
        assert "MessageEncoder" in msg


class TestMaxSchemaDirection:
    """The schema ceiling must be direction-specific: Request/Response share numeric
    codes (e.g. QUERY_SQL=9 vs FILES=9), so a shared lookup would let a server emit a
    schema>0 response on a type that doesn't support it."""

    def test_response_decoder_rejects_schema_1_on_db(self) -> None:
        """DB (type 4) must not accept schema=1 even though PREPARE
        (also type 4) does on the request side."""
        import struct

        from dqlitewire.constants import ResponseType

        decoder = MessageDecoder(is_request=False)
        # DbResponse body is a single uint64 db_id.
        body = b"\x00" * 8
        size_words = len(body) // 8
        header = struct.pack("<IBBH", size_words, ResponseType.DB, 1, 0)
        decoder.feed(header + body)
        with pytest.raises(DecodeError, match="Unsupported schema version"):
            decoder.decode()

    def test_response_decoder_rejects_schema_1_on_files(self) -> None:
        """FILES (type 9) must not accept schema=1 even though QUERY_SQL
        (also type 9) does on the request side."""
        import struct

        from dqlitewire.constants import ResponseType

        decoder = MessageDecoder(is_request=False)
        # Build a FILES response body with count=0 so decode_body would
        # otherwise succeed. Header carries schema=1.
        body = b"\x00" * 8  # uint64 count = 0
        size_words = len(body) // 8
        header = struct.pack("<IBBH", size_words, ResponseType.FILES, 1, 0)
        decoder.feed(header + body)
        with pytest.raises(DecodeError, match="Unsupported schema version"):
            decoder.decode()

    def test_response_decoder_rejects_schema_1_on_result(self) -> None:
        """RESULT (type 6) must not accept schema=1 even though QUERY
        (also type 6) does on the request side."""
        import struct

        from dqlitewire.constants import ResponseType

        decoder = MessageDecoder(is_request=False)
        # ResultResponse body is two uint64s (last_insert_id, rows_affected).
        body = b"\x00" * 16
        size_words = len(body) // 8
        header = struct.pack("<IBBH", size_words, ResponseType.RESULT, 1, 0)
        decoder.feed(header + body)
        with pytest.raises(DecodeError, match="Unsupported schema version"):
            decoder.decode()

    def test_response_decoder_rejects_schema_1_on_empty(self) -> None:
        """EMPTY (type 8) must not accept schema=1 even though EXEC_SQL
        (also type 8) does on the request side."""
        import struct

        from dqlitewire.constants import ResponseType

        decoder = MessageDecoder(is_request=False)
        body = b"\x00" * 8  # EMPTY body is a single placeholder uint64.
        size_words = len(body) // 8
        header = struct.pack("<IBBH", size_words, ResponseType.EMPTY, 1, 0)
        decoder.feed(header + body)
        with pytest.raises(DecodeError, match="Unsupported schema version"):
            decoder.decode()

    def test_response_decoder_accepts_schema_1_on_stmt(self) -> None:
        """STMT is the only response type that genuinely supports schema=1.
        Regression guard: the direction split must not break this path."""
        decoder = MessageDecoder(is_request=False)
        stmt = StmtResponse(db_id=1, stmt_id=2, num_params=3, tail_offset=0)
        decoder.feed(stmt.encode())
        decoded = decoder.decode()
        assert isinstance(decoded, StmtResponse)
        assert decoded.tail_offset == 0

    def test_request_decoder_still_accepts_schema_1_on_query_sql(self) -> None:
        """QUERY_SQL (request, type 9) keeps schema=1 support even though
        FILES (response, type 9) loses it. Forcing schema=1 on the request
        side by supplying >255 params (the V1 count-size cutoff)."""
        from dqlitewire.messages import QuerySqlRequest

        decoder = MessageDecoder(is_request=True)
        decoder._handshake_done = True
        decoder._version = PROTOCOL_VERSION
        params = [1] * 256  # >255 → encoder picks schema=1
        req = QuerySqlRequest(db_id=1, sql="SELECT 1", params=params)
        decoder.feed(req.encode())
        decoded = decoder.decode()
        assert isinstance(decoded, QuerySqlRequest)
        assert len(decoded.params) == 256
