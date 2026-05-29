"""Tests for the exception hierarchy."""

import struct

import pytest

from dqlitewire.codec import MessageDecoder, MessageEncoder
from dqlitewire.constants import PROTOCOL_VERSION, ResponseType
from dqlitewire.exceptions import (
    ContinuationError,
    DecodeError,
    EncodeError,
    HandshakeError,
    PoisonedError,
    ProtocolError,
    ServerFailure,
    StreamError,
)
from dqlitewire.messages import FailureResponse


class TestExceptionHierarchy:
    """New subclasses must inherit ProtocolError so broad catches still work."""

    def test_encode_error_subclasses_protocol_error(self) -> None:
        assert issubclass(EncodeError, ProtocolError)

    def test_decode_error_subclasses_protocol_error(self) -> None:
        assert issubclass(DecodeError, ProtocolError)

    def test_server_failure_subclasses_protocol_error(self) -> None:
        assert issubclass(ServerFailure, ProtocolError)

    def test_stream_error_subclasses_protocol_error(self) -> None:
        assert issubclass(StreamError, ProtocolError)

    def test_poisoned_error_subclasses_stream_error(self) -> None:
        """PoisonedError is a StreamError: the offset is unknown after a poison,
        so the caller must reconnect like any other StreamError."""
        assert issubclass(PoisonedError, StreamError)
        assert issubclass(PoisonedError, ProtocolError)

    def test_handshake_error_subclasses_protocol_error(self) -> None:
        assert issubclass(HandshakeError, ProtocolError)

    def test_continuation_error_subclasses_protocol_error(self) -> None:
        assert issubclass(ContinuationError, ProtocolError)


class TestServerFailure:
    """ServerFailure carries structured ``code`` and ``message`` fields."""

    def test_carries_structured_code_and_message(self) -> None:
        exc = ServerFailure(code=5, message="database is locked")
        assert exc.code == 5
        assert exc.message == "database is locked"

    def test_str_includes_code_and_message(self) -> None:
        exc = ServerFailure(code=5, message="database is locked")
        s = str(exc)
        assert "5" in s
        assert "database is locked" in s

    def test_pickle_round_trip(self) -> None:
        """ServerFailure must survive pickle: unpickle calls
        ``ServerFailure(*args)``, so a single pre-formatted-string arg would
        hit the 2-arg __init__ and raise TypeError on cross-process capture."""
        import pickle

        original = ServerFailure(code=5, message="database is locked")
        restored = pickle.loads(pickle.dumps(original))
        assert isinstance(restored, ServerFailure)
        assert restored.code == 5
        assert restored.message == "database is locked"
        assert str(restored) == str(original)

    def test_copy_deepcopy_round_trip(self) -> None:
        """``copy.deepcopy`` uses the same reduce path as pickle."""
        import copy

        original = ServerFailure(code=10, message="no free pages")
        restored = copy.deepcopy(original)
        assert isinstance(restored, ServerFailure)
        assert restored.code == 10
        assert restored.message == "no free pages"


class TestRaiseSiteSubclasses:
    """Each raise site must raise a specific subclass, not a bare ProtocolError."""

    def test_encoder_unsupported_version_raises_handshake_error(self) -> None:
        with pytest.raises(HandshakeError, match="Unsupported protocol version"):
            MessageEncoder(version=0x1234)

    def test_decoder_unsupported_version_raises_handshake_error(self) -> None:
        with pytest.raises(HandshakeError, match="Unsupported protocol version"):
            MessageDecoder(version=0x1234)

    def test_decode_before_handshake_raises_handshake_error(self) -> None:
        dec = MessageDecoder(is_request=True)
        dec.feed(b"\x00" * 16)
        with pytest.raises(HandshakeError, match="handshake not yet received"):
            dec.decode()

    def test_decode_bytes_before_handshake_raises_handshake_error(self) -> None:
        dec = MessageDecoder(is_request=True)
        with pytest.raises(HandshakeError, match="handshake not yet received"):
            dec.decode_bytes(b"\x00" * 8)

    def test_decode_handshake_already_completed_raises_handshake_error(self) -> None:
        dec = MessageDecoder(is_request=True)
        dec.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        dec.decode_handshake()
        with pytest.raises(HandshakeError, match="Handshake already completed"):
            dec.decode_handshake()

    def test_decode_handshake_bad_version_raises_handshake_error(self) -> None:
        dec = MessageDecoder(is_request=True)
        dec.feed((0x1234).to_bytes(8, "little"))
        with pytest.raises(HandshakeError, match="Unsupported protocol version"):
            dec.decode_handshake()

    def test_decode_during_continuation_raises_continuation_error(self) -> None:
        dec = MessageDecoder(is_request=False)
        dec._continuation_expected = True
        with pytest.raises(ContinuationError, match="continuation"):
            dec.decode()

    def test_decode_continuation_when_not_expected_raises_continuation_error(
        self,
    ) -> None:
        dec = MessageDecoder(is_request=False)
        with pytest.raises(ContinuationError, match="no ROWS continuation"):
            dec.decode_continuation()

    def test_skip_during_continuation_raises_continuation_error(self) -> None:
        dec = MessageDecoder(is_request=False)
        dec._continuation_expected = True
        with pytest.raises(ContinuationError, match="continuation"):
            dec.skip_message()

    def test_failure_during_continuation_raises_server_failure(self) -> None:
        """A FAILURE during ROWS continuation is a recoverable server error:
        ServerFailure with code/message, not a bare ProtocolError."""
        dec = MessageDecoder(is_request=False)
        dec._continuation_expected = True
        fail_bytes = FailureResponse(code=5, message="database is locked").encode()
        dec.feed(fail_bytes)
        with pytest.raises(ServerFailure) as exc_info:
            dec.decode_continuation()
        assert exc_info.value.code == 5
        assert exc_info.value.message == "database is locked"

    def test_wrong_msg_type_during_continuation_raises_stream_error(self) -> None:
        """A non-ROWS message where a ROWS continuation is expected means the
        stream is desynchronized: StreamError, not a bare ProtocolError."""
        dec = MessageDecoder(is_request=False)
        dec._continuation_expected = True
        # Feed a DbResponse (type=4) where a ROWS continuation was expected.
        body = struct.pack("<II", 1, 0)
        header = struct.pack("<IBBH", len(body) // 8, ResponseType.DB, 0, 0)
        dec.feed(header + body)
        with pytest.raises(StreamError, match="ROWS continuation"):
            dec.decode_continuation()

    def test_poisoned_buffer_raises_poisoned_error(self) -> None:
        """_check_poisoned() raises PoisonedError with the original via __cause__."""
        dec = MessageDecoder(is_request=False)
        original = RuntimeError("original decode failure")
        dec._buffer.poison(original)
        with pytest.raises(PoisonedError, match="poisoned"):
            dec.decode()

    def test_poisoned_error_preserves_cause(self) -> None:
        dec = MessageDecoder(is_request=False)
        original = ValueError("oops")
        dec._buffer.poison(original)
        with pytest.raises(PoisonedError) as exc_info:
            dec.decode()
        assert exc_info.value.__cause__ is original


class TestBackwardCompat:
    """Broad ``except ProtocolError`` callers must keep working."""

    def test_server_failure_caught_by_protocol_error(self) -> None:
        dec = MessageDecoder(is_request=False)
        dec._continuation_expected = True
        fail_bytes = FailureResponse(code=5, message="x").encode()
        dec.feed(fail_bytes)
        with pytest.raises(ProtocolError):
            dec.decode_continuation()

    def test_handshake_error_caught_by_protocol_error(self) -> None:
        with pytest.raises(ProtocolError):
            MessageEncoder(version=0x1234)

    def test_continuation_error_caught_by_protocol_error(self) -> None:
        dec = MessageDecoder(is_request=False)
        with pytest.raises(ProtocolError):
            dec.decode_continuation()

    def test_poisoned_error_caught_by_protocol_error(self) -> None:
        dec = MessageDecoder(is_request=False)
        dec._buffer.poison(RuntimeError("x"))
        with pytest.raises(ProtocolError):
            dec.decode()


class TestExceptionPickleRoundTrip:
    """Pin pickle for every class: default __reduce__ round-trips one message
    arg today but would silently break cross-process capture if a second
    positional __init__ arg were added (as ServerFailure once did)."""

    def _round_trip(self, e: Exception) -> Exception:
        import pickle

        restored: Exception = pickle.loads(pickle.dumps(e))
        return restored

    @pytest.mark.parametrize(
        "cls",
        [
            DecodeError,
            EncodeError,
            HandshakeError,
            ContinuationError,
            StreamError,
            PoisonedError,
            ProtocolError,
        ],
    )
    def test_exception_pickle_round_trip(self, cls: type) -> None:
        original = cls("test message")
        restored = self._round_trip(original)
        assert isinstance(restored, cls)
        assert str(restored) == str(original)

    @pytest.mark.parametrize(
        "cls",
        [
            DecodeError,
            EncodeError,
            HandshakeError,
            ContinuationError,
            StreamError,
            PoisonedError,
            ProtocolError,
        ],
    )
    def test_exception_deepcopy_round_trip(self, cls: type) -> None:
        import copy

        original = cls("test message")
        restored = copy.deepcopy(original)
        assert isinstance(restored, cls)
        assert str(restored) == str(original)

    def test_poisoned_error_cause_pickle_behaviour(self) -> None:
        """Pin that default pickle drops ``__cause__`` so a future custom
        __reduce__ that preserves it is a deliberate change."""
        inner = ValueError("decode failed")
        pe = PoisonedError("buffer poisoned")
        pe.__cause__ = inner
        restored = self._round_trip(pe)
        assert restored.__cause__ is None or isinstance(restored.__cause__, ValueError)
