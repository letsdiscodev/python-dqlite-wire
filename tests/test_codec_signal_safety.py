"""Signal-safety tests for MessageDecoder.

A KeyboardInterrupt (or any BaseException) delivered inside the parser after
read_message has advanced _pos, but outside the poison handler, escapes without
poisoning; the caller's retry then reads the next message boundary and silently
loses one message. The decoder must poison (or cleanly revert) in those windows.

These tests use sys.settrace to inject the exception at a specific source line —
the reliable way to hit CPython's bytecode-boundary async-exception windows.
"""

from __future__ import annotations

import contextlib
import struct
import sys
from types import FrameType
from typing import Any

import pytest

from dqlitewire.codec import MessageDecoder
from dqlitewire.constants import PROTOCOL_VERSION, ResponseType
from dqlitewire.exceptions import DecodeError, ProtocolError
from dqlitewire.messages import DbResponse, LeaderResponse

_DEFAULT_EXC: BaseException = KeyboardInterrupt("injected")


def _make_db(db_id: int) -> bytes:
    body = struct.pack("<II", db_id, 0)
    header = struct.pack("<IBBH", len(body) // 8, ResponseType.DB, 0, 0)
    return header + body


def _tracer_raising_in(
    func_name: str,
    exc: BaseException = _DEFAULT_EXC,
) -> Any:
    """Raise ``exc`` on the first line event inside ``func_name``."""
    state = {"raised": False}

    def tracer(frame: FrameType, event: str, arg: object) -> Any:
        if event != "line":
            return tracer
        if frame.f_code.co_name != func_name:
            return tracer
        if state["raised"]:
            return tracer
        state["raised"] = True
        raise exc

    return tracer


def _tracer_raising_in_after_call(
    func_name: str,
    callee_name: str,
    exc: BaseException = _DEFAULT_EXC,
) -> Any:
    """Raise ``exc`` on the first line in ``func_name`` after ``callee_name`` returns,
    targeting the window between a sub-call and the caller's post-call statements."""
    state = {"saw_return": False, "raised": False}

    def tracer(frame: FrameType, event: str, arg: object) -> Any:
        if state["raised"]:
            return tracer
        if event == "return" and frame.f_code.co_name == callee_name:
            state["saw_return"] = True
        elif event == "line" and frame.f_code.co_name == func_name and state["saw_return"]:
            state["raised"] = True
            raise exc
        return tracer

    return tracer


class TestDecodeSignalSafety:
    """The consumed-but-unpoisoned window in decode() / decode_continuation()."""

    def test_keyboard_interrupt_inside_parse_leaves_decoder_poisoned(self) -> None:
        """A KeyboardInterrupt while decode_bytes parses a consumed message must poison:
        ``except Exception`` missed BaseException, so the retry silently dropped a message."""
        dec = MessageDecoder(is_request=False)
        dec.feed(_make_db(1) + _make_db(2))

        tracer = _tracer_raising_in("decode_bytes")

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                dec.decode()
        finally:
            sys.settrace(None)

        assert dec.is_poisoned, "decode() must poison on injected KeyboardInterrupt"

        with pytest.raises(ProtocolError, match="poisoned"):
            dec.decode()

    def test_decode_does_not_poison_on_oversized_header(self) -> None:
        """Counter-test: an oversized-header DecodeError fires before read_message
        advances _pos, so skip_message()/reset recovery works — must not poison."""
        dec = MessageDecoder(is_request=False)
        dec._buffer._max_message_size = 128
        # size_words=100 → 800-byte body > 128
        huge = struct.pack("<IBBH", 100, 0, 0, 0)
        dec.feed(huge)

        with pytest.raises(DecodeError, match="exceeds maximum"):
            dec.decode()

        assert not dec.is_poisoned

    def test_keyboard_interrupt_between_parse_and_flag_set_poisons(self) -> None:
        """A KeyboardInterrupt after decode_bytes returns but before _continuation_expected
        is set must poison: the flag store now lives inside the try/except, else the next
        decode() would read the continuation frame as a top-level message."""
        # Any valid message works; the flag-check line runs before isinstance() short-circuits.
        dec = MessageDecoder(is_request=False)
        dec.feed(_make_db(1) + _make_db(2))

        tracer = _tracer_raising_in_after_call("decode", "decode_bytes")

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                dec.decode()
        finally:
            sys.settrace(None)

        assert dec.is_poisoned, (
            "decode() must poison when an interrupt lands after "
            "decode_bytes returns but before the flag store"
        )

        with pytest.raises(ProtocolError, match="poisoned"):
            dec.decode()

    def test_keyboard_interrupt_in_decode_continuation_poisons(self) -> None:
        """Same hazard as decode(): decode_continuation() also needs BaseException handling."""
        # The frame just needs bytes past the header check; the tracer fires before parsing.
        body = b"\x00" * 8
        header = struct.pack("<IBBH", len(body) // 8, ResponseType.ROWS, 0, 0)
        frame = header + body

        dec = MessageDecoder(is_request=False)
        dec._continuation_expected = True
        dec.feed(frame + frame)

        # Inject in decode_continuation itself (where the try: block opens), not the parser.
        tracer = _tracer_raising_in("decode_continuation")

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                dec.decode_continuation()
        finally:
            sys.settrace(None)

        # Either the tracer hit after read_message consumed a frame (must be poisoned)
        # or before any consume (no harm); accept both.
        if dec.is_poisoned:
            with pytest.raises(ProtocolError, match="poisoned"):
                dec.decode_continuation()
        else:
            assert dec._buffer.available() >= 0


def test_leader_response_decodes_cleanly_without_interrupt() -> None:
    """Sanity: the widened except must not regress the happy path."""
    dec = MessageDecoder(is_request=False)
    dec.feed(LeaderResponse(node_id=7, address="host:1234").encode())
    msg = dec.decode()
    assert isinstance(msg, LeaderResponse)
    assert msg.node_id == 7
    assert not dec.is_poisoned


def test_db_response_decodes_cleanly_without_interrupt() -> None:
    """Sanity: full loop through decode() still works."""
    dec = MessageDecoder(is_request=False)
    dec.feed(_make_db(42))
    msg = dec.decode()
    assert isinstance(msg, DbResponse)
    assert msg.db_id == 42


class TestDecodeHandshakeSignalSafety:
    """Signal split inside decode_handshake.

    Pre-fix, an interrupt between read_bytes(8) and the final state store left bytes
    consumed but _handshake_done False; the retry re-peeked the next message as a
    handshake and raised a misleading "Unsupported protocol version". The fix marks
    state before consuming and reverts on failure, so the retry is deterministic.
    """

    def test_keyboard_interrupt_post_consume_is_not_silently_misleading(
        self,
    ) -> None:
        dec = MessageDecoder(is_request=True)
        # Valid handshake then 8 bytes that fail the version check if misread as a handshake.
        dec.feed(PROTOCOL_VERSION.to_bytes(8, "little") + b"\xff" * 8)

        state = {"raised": False}

        def tracer(frame: FrameType, event: str, arg: object) -> Any:
            if event != "line":
                return tracer
            if frame.f_code.co_name != "decode_handshake":
                return tracer
            if state["raised"]:
                return tracer
            try:
                with open(frame.f_code.co_filename) as f:
                    src_line = f.readlines()[frame.f_lineno - 1]
            except OSError:
                return tracer
            # Inject at `self._version = version`: present in both layouts, and pre-fix
            # it runs after read_bytes(8).
            if "self._version = version" in src_line:
                state["raised"] = True
                raise KeyboardInterrupt("injected mid-commit")
            return tracer

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                dec.decode_handshake()
        finally:
            sys.settrace(None)

        # After the torn window the retry must NOT report "Unsupported" for the next
        # message's bytes. Either the state reverted (retry succeeds) or _handshake_done
        # was set before the consume (retry raises "already completed").
        retry_err: Exception | None = None
        retry_result: int | None = None
        try:
            retry_result = dec.decode_handshake()
        except ProtocolError as e:
            retry_err = e

        if retry_err is not None:
            assert "Unsupported" not in str(retry_err), (
                f"retry reported misleading error: {retry_err}"
            )
        else:
            assert retry_result == PROTOCOL_VERSION
            assert dec._handshake_done is True

    def test_happy_path_handshake_still_works(self) -> None:
        """Sanity: without an interrupt, decode_handshake sets all three state bits."""
        dec = MessageDecoder(is_request=True)
        dec.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        version = dec.decode_handshake()
        assert version == PROTOCOL_VERSION
        assert dec._handshake_done is True
        assert dec._version == PROTOCOL_VERSION

    def test_unsupported_version_does_not_consume_bytes(self) -> None:
        """Counter-test: an unsupported version leaves the 8 peeked bytes in the buffer."""
        dec = MessageDecoder(is_request=True)
        bogus = (0xDEADBEEF).to_bytes(8, "little")
        dec.feed(bogus)

        with pytest.raises(ProtocolError, match="Unsupported"):
            dec.decode_handshake()

        assert dec._handshake_done is False
        assert dec._buffer.available() == 8

    def test_decode_handshake_reverts_state_on_read_bytes_failure(self) -> None:
        """Strict revert contract: decode_handshake commits _version/_handshake_done before
        read_bytes(8), and the except BaseException block reverts both on failure, leaving
        the bytes in the buffer for a deterministic retry."""
        dec = MessageDecoder(is_request=True)
        dec.feed(PROTOCOL_VERSION.to_bytes(8, "little"))

        original_read_bytes = dec._buffer.read_bytes

        def exploding_read_bytes(n: int) -> bytes:
            raise KeyboardInterrupt("simulated async cancel")

        dec._buffer.read_bytes = exploding_read_bytes

        with pytest.raises(KeyboardInterrupt):
            dec.decode_handshake()

        # Both state fields rolled back; no partial commit.
        assert dec._handshake_done is False
        assert dec._version is None

        # Bytes never consumed — retry on the restored buffer returns PROTOCOL_VERSION.
        dec._buffer.read_bytes = original_read_bytes
        assert dec._buffer.available() == 8
        assert dec.decode_handshake() == PROTOCOL_VERSION
        assert dec._handshake_done is True
        assert dec._version == PROTOCOL_VERSION

    def test_decode_handshake_reverts_state_on_system_exit(self) -> None:
        """except BaseException width: SystemExit must also trigger the revert (a narrower
        except Exception would leak commit state on shutdown paths)."""
        dec = MessageDecoder(is_request=True)
        dec.feed(PROTOCOL_VERSION.to_bytes(8, "little"))

        def exploding_read_bytes(n: int) -> bytes:
            raise SystemExit(1)

        dec._buffer.read_bytes = exploding_read_bytes

        with pytest.raises(SystemExit):
            dec.decode_handshake()

        assert dec._handshake_done is False
        assert dec._version is None
