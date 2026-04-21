"""Tests for the exception hierarchy.

``ProtocolError`` was previously raised for six qualitatively different
failure conditions, forcing callers to string-match to distinguish
recoverable server errors from fatal stream desync. Subclasses give
callers first-class discrimination.
"""

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
    """The new subclasses must all inherit from ProtocolError so existing
    ``except ProtocolError`` catches continue to work.
    """

    def test_encode_error_subclasses_protocol_error(self) -> None:
        assert issubclass(EncodeError, ProtocolError)

    def test_decode_error_subclasses_protocol_error(self) -> None:
        assert issubclass(DecodeError, ProtocolError)

    def test_server_failure_subclasses_protocol_error(self) -> None:
        assert issubclass(ServerFailure, ProtocolError)

    def test_stream_error_subclasses_protocol_error(self) -> None:
        assert issubclass(StreamError, ProtocolError)

    def test_poisoned_error_subclasses_stream_error(self) -> None:
        """PoisonedError is a StreamError — the stream offset is unknown
        after a poison, so the caller must reconnect the same way as any
        other StreamError.
        """
        assert issubclass(PoisonedError, StreamError)
        assert issubclass(PoisonedError, ProtocolError)

    def test_handshake_error_subclasses_protocol_error(self) -> None:
        assert issubclass(HandshakeError, ProtocolError)

    def test_continuation_error_subclasses_protocol_error(self) -> None:
        assert issubclass(ContinuationError, ProtocolError)


class TestServerFailure:
    """ServerFailure carries structured ``code`` and ``message`` fields
    instead of burying them in an f-string that callers must regex.
    """

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
        """ServerFailure must survive ``pickle.dumps`` + ``pickle.loads``.

        Workers that propagate exceptions back to a parent process
        (multiprocessing, concurrent.futures.ProcessPoolExecutor, Celery)
        unpickle by calling ``ServerFailure(*args)`` on the pickled args
        tuple. If args hold a single pre-formatted string, unpickle calls
        ``ServerFailure('[5] db locked')`` against a 2-arg ``__init__``
        and raises ``TypeError``. Mirror the fix that landed for
        ``client.OperationalError``.
        """
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
    """Each raise site in codec.py / buffer.py must raise a specific
    subclass rather than a bare ``ProtocolError``. Callers can then
    distinguish recoverable server errors from fatal stream desync.
    """

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
        """230: a FAILURE response during ROWS continuation is a
        recoverable server error; it must raise ServerFailure (not a
        bare ProtocolError) with ``code`` and ``message`` attributes.
        """
        dec = MessageDecoder(is_request=False)
        dec._continuation_expected = True
        fail_bytes = FailureResponse(code=5, message="database is locked").encode()
        dec.feed(fail_bytes)
        with pytest.raises(ServerFailure) as exc_info:
            dec.decode_continuation()
        assert exc_info.value.code == 5
        assert exc_info.value.message == "database is locked"

    def test_wrong_msg_type_during_continuation_raises_stream_error(self) -> None:
        """230: a non-ROWS message arriving when a ROWS continuation is
        expected means the stream is desynchronized — StreamError, not
        a bare ProtocolError.
        """
        dec = MessageDecoder(is_request=False)
        dec._continuation_expected = True
        # Feed a DbResponse (type=4) where a ROWS continuation was expected
        body = struct.pack("<II", 1, 0)
        header = struct.pack("<IBBH", len(body) // 8, ResponseType.DB, 0, 0)
        dec.feed(header + body)
        with pytest.raises(StreamError, match="ROWS continuation"):
            dec.decode_continuation()

    def test_poisoned_buffer_raises_poisoned_error(self) -> None:
        """230: _check_poisoned() raises PoisonedError (a StreamError).
        The original cause is preserved via __cause__.
        """
        dec = MessageDecoder(is_request=False)
        original = RuntimeError("original decode failure")
        dec._buffer.poison(original)
        with pytest.raises(PoisonedError, match="poisoned"):
            dec.decode()

    def test_poisoned_error_preserves_cause(self) -> None:
        """The original exception that caused the poison must be
        available via ``__cause__`` so callers can inspect it.
        """
        dec = MessageDecoder(is_request=False)
        original = ValueError("oops")
        dec._buffer.poison(original)
        with pytest.raises(PoisonedError) as exc_info:
            dec.decode()
        assert exc_info.value.__cause__ is original


class TestBackwardCompat:
    """Existing callers that catch ``ProtocolError`` broadly must keep
    working — every new subclass inherits from ProtocolError.
    """

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
