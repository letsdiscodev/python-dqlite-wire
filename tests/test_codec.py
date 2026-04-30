"""Tests for message codec."""

import pytest

from dqlitewire.codec import (
    MessageDecoder,
    MessageEncoder,
    decode_message,
    encode_message,
)
from dqlitewire.constants import PROTOCOL_VERSION, PROTOCOL_VERSION_LEGACY
from dqlitewire.exceptions import DecodeError, ProtocolError
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

    def test_encoder_rejects_invalid_version(self) -> None:
        """102: invalid handshake version should raise at construction time."""
        with pytest.raises(ProtocolError, match="Unsupported protocol version"):
            MessageEncoder(version=0xDEADBEEF)

    def test_encoder_accepts_legacy_version(self) -> None:
        """102: PROTOCOL_VERSION_LEGACY must be accepted."""
        encoder = MessageEncoder(version=PROTOCOL_VERSION_LEGACY)
        assert encoder.encode_handshake() is not None

    def test_encoder_accepts_default_version(self) -> None:
        """102: default PROTOCOL_VERSION must be accepted."""
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
        """122: response decoder must reject invalid version at construction."""
        with pytest.raises(ProtocolError, match="Unsupported protocol version"):
            MessageDecoder(version=0xDEADBEEF)

    def test_decoder_accepts_legacy_version(self) -> None:
        """122: PROTOCOL_VERSION_LEGACY must be accepted."""
        decoder = MessageDecoder(version=PROTOCOL_VERSION_LEGACY)
        assert decoder.version == PROTOCOL_VERSION_LEGACY

    def test_decoder_request_ignores_version(self) -> None:
        """122: request decoders ignore version param (comes from handshake)."""
        decoder = MessageDecoder(is_request=True, version=0xDEADBEEF)
        assert decoder.version is None  # not set until handshake

    def test_decoder_custom_max_message_size(self) -> None:
        """138: max_message_size should be configurable via constructor."""
        import struct

        from dqlitewire.exceptions import DecodeError

        decoder = MessageDecoder(max_message_size=64)
        # Feed a message larger than 64 bytes
        oversized = struct.pack("<IBBH", 100, 0, 0, 0)  # 808-byte body
        decoder.feed(oversized)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            decoder.decode()

    def test_decoder_default_max_message_size(self) -> None:
        """138: default max_message_size should match ReadBuffer default."""
        from dqlitewire.buffer import ReadBuffer

        decoder = MessageDecoder()
        assert decoder._buffer._max_message_size == ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE

    def test_decoder_custom_max_rows(self) -> None:
        """231: max_rows should be forwarded to RowsResponse.decode_body.

        The knob exists on RowsResponse.decode_body, but MessageDecoder
        previously omitted it at the call site, leaving the 1M default
        always in force regardless of caller configuration.
        """
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
        with pytest.raises(DecodeError, match="reached maximum 2"):
            decoder.decode()

    def test_decoder_default_max_rows(self) -> None:
        """231: default max_rows should match RowsResponse.DEFAULT_MAX_ROWS."""
        from dqlitewire.messages.responses import RowsResponse

        decoder = MessageDecoder()
        assert decoder._max_rows == RowsResponse.DEFAULT_MAX_ROWS

    def test_decoder_rejects_zero_max_rows(self) -> None:
        """231: max_rows < 1 should be rejected at construction time."""
        with pytest.raises(ValueError, match="max_rows must be >= 1"):
            MessageDecoder(max_rows=0)

    def test_decoder_rejects_zero_max_message_size(self) -> None:
        """max_message_size < 1 should be rejected at construction time,
        symmetric with the max_rows validation. Otherwise feed() raises a
        confusing 'projected ... > 0' error on first byte."""
        with pytest.raises(ValueError, match="max_message_size must be >= 1"):
            MessageDecoder(max_message_size=0)

    def test_decoder_rejects_negative_max_message_size(self) -> None:
        """Negative max_message_size is also invalid."""
        with pytest.raises(ValueError, match="max_message_size must be >= 1"):
            MessageDecoder(max_message_size=-1)

    def test_decoder_rejects_zero_max_continuation_frames(self) -> None:
        """``max_continuation_frames < 1`` rejected at construction so a
        bug that defaults the cap to 0 cannot silently let any
        continuation frame through."""
        with pytest.raises(ValueError, match="max_continuation_frames must be >= 1"):
            MessageDecoder(max_continuation_frames=0)

    def test_decoder_rejects_negative_max_continuation_frames(self) -> None:
        with pytest.raises(ValueError, match="max_continuation_frames must be >= 1"):
            MessageDecoder(max_continuation_frames=-5)

    def test_decoder_rejects_zero_max_total_rows(self) -> None:
        """``max_total_rows < 1`` rejected at construction symmetric
        with the other two caps."""
        with pytest.raises(ValueError, match="max_total_rows must be >= 1"):
            MessageDecoder(max_total_rows=0)

    def test_decoder_rejects_negative_max_total_rows(self) -> None:
        with pytest.raises(ValueError, match="max_total_rows must be >= 1"):
            MessageDecoder(max_total_rows=-1)

    def test_decoder_continuation_honors_max_rows(self) -> None:
        """231: decode_continuation should also honor max_rows.

        A multi-frame ROWS stream can exceed max_rows across frames; the
        per-frame cap still applies to each continuation response.
        """
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
        with pytest.raises(DecodeError, match="reached maximum 2"):
            decoder.decode_continuation()

    def test_decoder_continuation_rejects_total_rows_overflow(self) -> None:
        """Cumulative ``max_total_rows`` cap fires across frames in a
        ROWS continuation. Pin the runtime check so a future refactor
        that drops the running total cannot silently let an unbounded
        stream through (DoS surface)."""
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
        """``max_continuation_frames`` cap fires after the runtime
        decoder sees one frame too many. Pin the slow-drip DoS
        defence."""
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
        # The decoder counts the initial frame as frame 1, so
        # max_continuation_frames=2 admits the initial frame plus
        # exactly one continuation frame; the second continuation
        # frame (third overall) trips the cap.
        decoder = MessageDecoder(max_continuation_frames=2, max_total_rows=100)
        decoder.feed(first.encode())
        decoder.decode()
        decoder.feed(next_frame.encode())
        decoder.decode_continuation()
        decoder.feed(last_frame.encode())
        with pytest.raises(DecodeError, match="max_continuation_frames"):
            decoder.decode_continuation()

    def test_decoder_initial_frame_already_exceeded_max_total_rows(self) -> None:
        """When the first ROWS frame already overflows
        ``max_total_rows`` (server-side aggregation bug, attacker
        flood, etc.), the decoder must fail at ``decode()`` time on
        the initial frame, not delay until ``decode_continuation``."""
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

    def test_header_encode_msg_type_overflow_raises_encode_error(self) -> None:
        """Header with msg_type exceeding uint8 must raise EncodeError."""
        from dqlitewire.exceptions import EncodeError
        from dqlitewire.messages.base import Header

        header = Header(size_words=1, msg_type=256, schema=0)
        with pytest.raises(EncodeError, match="header"):
            header.encode()

    def test_header_encode_schema_overflow_raises_encode_error(self) -> None:
        """Header with schema exceeding uint8 must raise EncodeError."""
        from dqlitewire.exceptions import EncodeError
        from dqlitewire.messages.base import Header

        header = Header(size_words=1, msg_type=0, schema=256)
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

    def test_rows_response(self) -> None:
        """131: RowsResponse through full codec round-trip."""
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
        """PrepareRequest with schema=1 must preserve schema through codec round-trip."""
        original = PrepareRequest(db_id=1, sql="SELECT 1", schema=1)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, PrepareRequest)
        assert decoded.schema == 1

    def test_stmt_response_v1_through_codec(self) -> None:
        """StmtResponse V1 must pass through the codec schema-version gate."""
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
        """112: StmtResponse V1 via decode_message convenience function."""
        msg = StmtResponse(db_id=1, stmt_id=2, num_params=3, tail_offset=42)
        encoded = msg.encode()
        decoded = decode_message(encoded)
        assert isinstance(decoded, StmtResponse)
        assert decoded.tail_offset == 42

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
        # Legacy PROMOTE decodes to VOTER (not 99 from trailing
        # bytes; the body slice is bounded by the header word count).
        from dqlitewire.constants import NodeRole

        assert decoded.role == NodeRole.VOTER

    def test_decode_bytes_rejects_short_body(self) -> None:
        """decode_bytes() must raise DecodeError if body is shorter than header claims."""
        from dqlitewire.messages.base import Header

        # Header claims 2 words (16 bytes) but only 8 bytes follow
        header = Header(size_words=2, msg_type=0, schema=0)
        data = header.encode() + b"\x00" * 8  # Only 8 bytes, not 16
        with pytest.raises(DecodeError, match="[Bb]ody.*short"):
            decode_message(data, is_request=True)

    def test_heartbeat_request_rejected_by_decode_message(self) -> None:
        """Type-2 frames are not decodable via the public registry.

        ``_HeartbeatRequest`` stays private and is intentionally absent
        from ``REQUEST_TYPES`` (upstream C's dispatcher falls through to
        ``DQLITE_PARSE`` for type 2, so no real server accepts one). A
        synthesised type-2 frame must surface as the unknown-type error
        rather than silently decode.
        """
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
        """CONNECT (type 11) is a Raft-transport frame; the public
        request dispatcher rejects it with an ``unknown type`` error
        mirroring upstream ``gateway.c``'s ``DQLITE_PARSE`` fallthrough.
        The private ``_ConnectRequest`` class is still available for
        test-mock / golden-byte harnesses.
        """
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
        """Max uint64 values should survive full codec roundtrip."""
        original = ClientRequest(client_id=2**64 - 1)
        encoded = encode_message(original)
        decoded = decode_message(encoded, is_request=True)
        assert isinstance(decoded, ClientRequest)
        assert decoded.client_id == 2**64 - 1

    def test_stmt_response_v0_with_trailing_data_rejected(self) -> None:
        """StmtResponse V0 with trailing bytes must be rejected outright.

        Previously the schema-aware decode only prevented mis-detection
        as V1 (tail_offset was forced None). Under strict-length semantics
        a 24-byte body under schema=0 is itself invalid — the Go/C
        servers never emit trailing padding on fixed-body responses, and
        accepting it silently is the same "two states collapsed into one"
        bug that schema-awareness was meant to fix.
        """
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
        """decode_continuation should surface server error as ServerFailure.

        The exception must be a ServerFailure (subclass of ProtocolError)
        carrying structured ``code`` and ``message`` attributes — NOT a
        DecodeError (which would mean the failure body was misinterpreted
        as row data) and NOT a bare ProtocolError — callers need to
        distinguish recoverable server errors from fatal stream
        desync.
        """
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
        """decode_continuation should raise ProtocolError for non-ROWS, non-FAILURE type."""
        from dqlitewire.exceptions import ProtocolError

        result = ResultResponse(last_insert_id=0, rows_affected=0)
        result_bytes = result.encode()

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(result_bytes)

        with pytest.raises(ProtocolError, match="Expected ROWS continuation"):
            decoder.decode_continuation()

    def test_decode_continuation_accepts_empty_response_as_terminator(self) -> None:
        """decode_continuation must accept an EmptyResponse mid-stream as
        a clean terminator.

        The upstream C server emits ``EmptyResponse`` instead of a final
        ROWS frame when the in-flight query is cancelled by an INTERRUPT
        request (gateway.c::handle_query_done_cb). The reference Go
        client (Protocol.Interrupt) loops reading until it sees
        ``ResponseEmpty`` and treats that frame as the normal
        acknowledgement that the in-flight query was interrupted. The
        Python codec must not poison its buffer on that frame.
        """
        from dqlitewire.messages.responses import EmptyResponse

        empty_bytes = EmptyResponse().encode()

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(empty_bytes)

        result = decoder.decode_continuation()
        assert isinstance(result, EmptyResponse)
        # Flag cleared so the decoder can resume normal traffic.
        assert decoder._continuation_expected is False
        # Buffer is NOT poisoned.
        assert not decoder._buffer.is_poisoned

    def test_decode_continuation_empty_after_partial_rows(self) -> None:
        """Realistic interrupt-mid-stream sequence: a ROWS-PART frame
        with rows is delivered, then the caller drains via
        ``decode_continuation`` and receives an EmptyResponse instead of
        a final ROWS-DONE frame. Pin that this transition clears the
        continuation flag without poisoning."""
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
        """A well-formed FailureResponse mid-continuation surfaces as
        ``ServerFailure`` without poisoning the buffer. Upstream's
        ``query_work_done`` path can emit FAILURE after ROWS_PART chunks
        have already been written (e.g. mid-stream constraint or
        SQLITE_BUSY). The connection is still wire-coherent at the
        offset following the failure body, so subsequent requests must
        decode normally without ``reset()``."""
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

    def test_decode_continuation_malformed_failure_body_still_poisons(self) -> None:
        """The clean-failure path (above) is opt-in to well-formed
        FailureResponse bodies only. A malformed body (e.g., missing
        null terminator on the message text) routes through the broad
        ``except`` and poisons the buffer — same discipline as the
        malformed-empty case."""
        import struct as _struct

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.messages.base import Header

        # Body: 8 bytes of code, then 8 bytes of a bareword without a NUL
        # terminator. ``decode_text`` raises DecodeError on the missing NUL.
        body = _struct.pack("<Q", 1) + b"abcdefgh"
        header = Header(size_words=len(body) // 8, msg_type=0, schema=0)
        bad_failure = header.encode() + body

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(bad_failure)

        with pytest.raises(DecodeError):
            decoder.decode_continuation()
        assert decoder.is_poisoned is True

    def test_decode_continuation_malformed_empty_still_poisons(self) -> None:
        """A malformed EmptyResponse (non-zero reserved field) must
        surface the underlying DecodeError and poison the buffer — the
        clean-terminator path is opt-in to well-formed frames only."""
        import struct

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.messages.base import Header

        # EMPTY body is a single uint64 reserved field, must be 0.
        body = struct.pack("<Q", 1)
        header = Header(size_words=1, msg_type=8, schema=0)  # ResponseType.EMPTY = 8
        empty_bad = header.encode() + body

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        decoder.feed(empty_bad)

        with pytest.raises(DecodeError, match="reserved field must be 0"):
            decoder.decode_continuation()
        assert decoder.is_poisoned is True


class TestDecoderContinuationExpected:
    """Regression tests for the continuation-expected state machine.

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

        # decode() should work fine — no continuation expected
        result2 = decoder.decode()
        assert isinstance(result2, ResultResponse)


class TestContinuationFlagCompleteness:
    """Regression tests for continuation-flag completeness.

    The ``_continuation_expected`` flag was added so ``decode()``
    could refuse to run on a frame that is actually a continuation.
    The state machine is complete only when
    ``decode_continuation()`` also refuses when the flag is False
    (no continuation in progress), and ``skip_message()`` refuses
    when the flag is True (continuation in progress).
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

    def test_skip_oversized_poisons_after_capped_skip(self) -> None:
        """After a capped oversized skip, buffer is poisoned."""
        import struct

        from dqlitewire.exceptions import DecodeError, ProtocolError

        # Create a decoder with a small max_message_size
        decoder = MessageDecoder(is_request=False)
        decoder._buffer._max_message_size = 128

        # Build an oversized message header (size_words=100 → 808 bytes total)
        oversized_header = struct.pack("<IBBH", 100, 0, 0, 0)

        # Feed just the oversized header
        decoder.feed(oversized_header)

        assert decoder.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            decoder.decode()

        # skip_message() should handle the oversized message
        result = decoder.skip_message()
        assert result is False
        assert decoder.is_skipping is True

        # Feed enough bytes to complete the capped skip
        decoder.feed(b"\x00" * decoder._buffer._skip_remaining)

        # Buffer is now poisoned — stream is desynchronized
        assert decoder.is_skipping is False
        assert decoder.is_poisoned is True

        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()


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
        # Per the poison-gate contract, every mutating/consuming entry point on the buffer
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

    def test_oversized_header_does_not_poison_before_skip(self) -> None:
        """An oversized-header error from the buffer framing layer is
        recoverable via skip_message(); it must NOT poison because the
        bytes have not yet been consumed from the buffer.

        After a capped skip completes, the buffer IS poisoned because the
        stream is desynchronized.
        """
        import struct

        decoder = MessageDecoder(is_request=False)
        decoder._buffer._max_message_size = 128
        oversized = struct.pack("<IBBH", 100, 0, 0, 0)  # 808-byte body > 128
        decoder.feed(oversized)

        with pytest.raises(DecodeError, match="exceeds maximum"):
            decoder.decode()
        # NOT poisoned before skip — skip_message() is the recovery path.
        assert decoder.is_poisoned is False

        # After capped skip completes, buffer IS poisoned (desync).
        decoder.skip_message()
        decoder.feed(b"\x00" * decoder._buffer._skip_remaining)
        assert decoder.is_poisoned is True

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

        decoder.decode_bytes = boom  # type: ignore[assignment]

        with pytest.raises(ValueError, match="simulated body parse failure"):
            decoder.decode()
        assert decoder.is_poisoned is True

        # Subsequent decode() must raise poisoned, even though decode_bytes
        # is monkey-patched. Restore the real decode_bytes first to make
        # sure the poison check fires before any decode_bytes call.
        # Per the poison-gate contract, feed() on a poisoned buffer also raises, so we
        # cannot push more bytes through without a reset — the poison
        # check fires directly on decode().
        del decoder.decode_bytes
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.feed(msg)
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()

    def test_decode_continuation_poisons_on_wrong_type(self) -> None:
        """decode_continuation() must poison the decoder when the next
        message is not a ROWS continuation, FAILURE, or EMPTY. The bytes
        have been consumed from the buffer; subsequent operations cannot
        trust the offset.

        ``EmptyResponse`` is now treated as a clean interrupt-ack
        terminator and does NOT poison — that case is covered in
        :class:`TestDecoderContinuation` separately. Use a
        ``ResultResponse`` here to exercise the genuine-desync path.
        """
        from dqlitewire.exceptions import ProtocolError

        decoder = MessageDecoder(is_request=False)
        decoder._continuation_expected = True
        result = ResultResponse(last_insert_id=0, rows_affected=0).encode()
        decoder.feed(result)

        with pytest.raises(ProtocolError, match="Expected ROWS continuation"):
            decoder.decode_continuation()
        assert decoder.is_poisoned is True

        # Subsequent decode() also fails poisoned.
        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()

    def test_decode_bytes_honors_poison(self) -> None:
        """Regression: decode_bytes must honor the poison flag.

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

    def test_has_message_true_then_decode_raises_and_poisons_on_reserved(self) -> None:
        """``has_message()`` inspects only the 4-byte ``size_words`` prefix —
        a well-framed header with a non-zero 16-bit reserved field returns
        True, but ``decode()`` then reads the bytes, surfaces
        ``DecodeError("reserved field must be 0")`` from ``Header.decode``,
        and poisons the buffer. Pin the compound contract so a regression
        that moves reserved validation out of the decode path (or narrows
        the ``except BaseException`` poison handler) fails loudly.
        """
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

        # Subsequent decode on the poisoned buffer must raise the
        # poison-specific error (a ProtocolError subclass), not re-run
        # the header parse.
        with pytest.raises((PoisonedError, ProtocolError), match="poisoned"):
            decoder.decode()

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

    def test_failure_during_continuation_does_not_poison(self) -> None:
        """083: server sends FailureResponse instead of continuation.

        The well-formed failure decode is a clean operational signal —
        the buffer is NOT poisoned and the connection remains usable
        for subsequent requests. Upstream's ``query_work_done`` reaches
        this path on mid-stream I/O / constraint failures after PART
        rows have already been written; users should see a normal
        OperationalError, not a forced reconnect."""
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
        """103: _continuation_expected must be False after FailureResponse."""
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

        # The continuation flag must be cleared on the clean-failure path.
        assert not decoder._continuation_expected

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

    def test_decode_continuation_rejects_unsupported_schema(self) -> None:
        """099: decode_continuation must validate schema like decode_bytes."""
        import struct

        from dqlitewire.constants import ValueType
        from dqlitewire.messages.responses import RowsResponse

        # Build an initial ROWS frame with has_more=True
        initial = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            has_more=True,
        )
        # Build a continuation frame with schema=1 (unsupported for ROWS)
        cont_body = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[2]],
            has_more=False,
        ).encode_body()
        # Manually build header with schema=1
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
        """123: oversized continuation frame must clear flag and poison."""
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

        # Flag must be cleared and buffer poisoned
        assert not decoder._continuation_expected
        assert decoder.is_poisoned

        with pytest.raises(ProtocolError, match="poisoned"):
            decoder.decode()


class TestEndToEndPipeline:
    """End-to-end tests exercising the full MessageEncoder + MessageDecoder pipeline."""

    def test_handshake_then_request_response_roundtrip(self) -> None:
        """Full stateful pipeline: handshake → request → response."""
        # Client side
        encoder = MessageEncoder()
        client_decoder = MessageDecoder(is_request=False)

        # Server side
        server_decoder = MessageDecoder(is_request=True)

        # Step 1: Client sends handshake
        handshake_bytes = encoder.encode_handshake()

        # Step 2: Server receives and decodes handshake
        server_decoder.feed(handshake_bytes)
        version = server_decoder.decode_handshake()
        assert version == PROTOCOL_VERSION

        # Step 3: Client sends a LeaderRequest
        request = LeaderRequest()
        request_bytes = encoder.encode(request)

        # Step 4: Server receives and decodes request
        server_decoder.feed(request_bytes)
        assert server_decoder.has_message()
        decoded_request = server_decoder.decode()
        assert isinstance(decoded_request, LeaderRequest)

        # Step 5: Server sends a LeaderResponse
        response = LeaderResponse(node_id=1, address="127.0.0.1:9001")
        response_bytes = response.encode()

        # Step 6: Client receives and decodes response
        client_decoder.feed(response_bytes)
        assert client_decoder.has_message()
        decoded_response = client_decoder.decode()
        assert isinstance(decoded_response, LeaderResponse)
        assert decoded_response.node_id == 1
        assert decoded_response.address == "127.0.0.1:9001"

    def test_multiple_messages_single_feed(self) -> None:
        """Concatenate multiple encoded messages, feed all at once, decode sequentially."""
        decoder = MessageDecoder(is_request=False)

        msg1 = WelcomeResponse(heartbeat_timeout=15000000000)
        msg2 = DbResponse(db_id=0)
        msg3 = StmtResponse(db_id=0, stmt_id=1, num_params=2)

        # Concatenate all bytes and feed in one call
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

        # No more messages
        assert not decoder.has_message()

    def test_partial_feed_chunked(self) -> None:
        """Split encoded message at arbitrary points, feed in chunks."""
        decoder = MessageDecoder(is_request=False)
        msg = ResultResponse(last_insert_id=42, rows_affected=7)
        encoded = msg.encode()

        # Feed one byte at a time
        for i in range(len(encoded) - 1):
            decoder.feed(encoded[i : i + 1])
            assert not decoder.has_message(), f"should not be complete at byte {i}"

        # Feed the last byte — now the message should be complete
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

        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(MessageDecoder())

    def test_message_encoder_pickle_raises(self) -> None:
        import pickle

        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(MessageEncoder())


class TestMaxSchemaDirection:
    """Schema-version ceiling must not bleed across request/response directions.

    ``RequestType`` and ``ResponseType`` share numeric codes (e.g.
    ``RequestType.QUERY_SQL=9`` collides with ``ResponseType.FILES=9``).
    A direction-agnostic ceiling lookup would let a hostile or buggy
    server emit a response whose type does not actually support schema>0
    but whose numeric code matches a request type that does.
    """

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
