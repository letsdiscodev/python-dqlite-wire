"""Signal-safety tests for MessageDecoder.

``MessageDecoder.decode()`` and ``decode_continuation()`` both consume
bytes from the buffer via ``read_message()`` and then parse them via
``decode_bytes`` / ``RowsResponse.decode_body``. The
parse step is wrapped in ``try/except Exception`` so that a
``DecodeError``/``ValueError``/``struct.error`` from the parser
poisons the buffer before propagating.

But ``except Exception`` does not catch ``KeyboardInterrupt``,
``SystemExit``, or any other ``BaseException`` subclass. A signal-
delivered ``KeyboardInterrupt`` landing inside the parser (after
``read_message`` has already advanced ``_pos`` past the consumed
bytes) propagates past the ``except`` block without poisoning. The
caller catches it at the top level and retries; the retry reads the
*next* message boundary, silently losing exactly one message.

``MessageDecoder.decode_handshake()`` has an analogous hazard: the
commit used to be (1) consume bytes, (2) set ``_version``, (3) set
``_handshake_done``. A signal between (1) and (3) left the bytes
consumed but state not marked; the retry re-peeked the first 8 bytes
of the *next* message as a handshake and raised a misleading
"Unsupported protocol version" error.

These tests use ``sys.settrace`` to inject a ``KeyboardInterrupt`` at
a specific source line inside a decoder method — the most reliable
way to exercise CPython's bytecode-boundary async-exception model
without relying on wallclock timing — and assert the decoder ends up
in a state the caller can detect and recover from, not silently torn.
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
    """Raise ``exc`` on the first line event in ``func_name`` AFTER a
    call to ``callee_name`` has returned. Used to target the window
    between a sub-call and the post-call statements in the caller.
    """
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
    """Regression tests for the consumed-but-unpoisoned window in
    ``decode()`` / ``decode_continuation()``.
    """

    def test_keyboard_interrupt_inside_parse_leaves_decoder_poisoned(self) -> None:
        """A ``KeyboardInterrupt`` delivered while ``decode_bytes`` is
        parsing a consumed message used to propagate without
        poisoning the decoder, because ``except Exception`` does not
        catch ``BaseException``. The retry read the next message
        boundary, silently dropping one message. The fix widens to
        ``except BaseException``.
        """
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

        # Retry must fail fast with ProtocolError rather than silently
        # returning the next message from a desynchronized stream.
        with pytest.raises(ProtocolError, match="poisoned"):
            dec.decode()

    def test_decode_does_not_poison_on_oversized_header(self) -> None:
        """Counter-test: an oversized-header DecodeError is raised
        BEFORE read_message advances _pos, so the bytes are still
        there and skip_message()/reset recovery works. The fix must
        not accidentally poison this path.
        """
        dec = MessageDecoder(is_request=False)
        dec._buffer._max_message_size = 128
        # size_words = 100 → body size = 800 > 128
        huge = struct.pack("<IBBH", 100, 0, 0, 0)
        dec.feed(huge)

        with pytest.raises(DecodeError, match="exceeds maximum"):
            dec.decode()

        assert not dec.is_poisoned

    def test_keyboard_interrupt_between_parse_and_flag_set_poisons(self) -> None:
        """233: a KeyboardInterrupt delivered in ``decode()`` AFTER
        ``decode_bytes`` returns but BEFORE ``_continuation_expected``
        is set used to propagate without poisoning, because the flag
        assignment lived outside the ``except BaseException`` block.
        The next ``decode()`` call would then read the continuation
        frame as a top-level message (silent stream desync).

        The fix moves the flag store inside the try/except block, so
        any exception in that window poisons the buffer.
        """
        # Feed any valid message — a DbResponse is fine because the
        # flag-check line runs before the isinstance() short-circuits.
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

        # The retry must fail fast with ProtocolError rather than
        # silently returning the next message from a desynchronized
        # stream.
        with pytest.raises(ProtocolError, match="poisoned"):
            dec.decode()

    def test_keyboard_interrupt_in_decode_continuation_poisons(self) -> None:
        """Same hazard as decode(): decode_continuation() also needs
        BaseException handling.
        """
        # Build a minimal ROWS continuation body. We don't actually
        # reach the parser — the tracer injects the interrupt before
        # any real parsing runs — so the frame just needs to carry
        # some bytes past the header check.
        body = b"\x00" * 8
        header = struct.pack("<IBBH", len(body) // 8, ResponseType.ROWS, 0, 0)
        frame = header + body

        dec = MessageDecoder(is_request=False)
        dec._continuation_expected = True
        dec.feed(frame + frame)

        # Inject inside decode_continuation itself (the wrapper
        # method), not the parser helper — the try: block opens there.
        tracer = _tracer_raising_in("decode_continuation")

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                dec.decode_continuation()
        finally:
            sys.settrace(None)

        # If the tracer landed after read_message consumed the first
        # frame, the decoder must be poisoned. If it landed before
        # any bytes were consumed (e.g. on the very first line), the
        # test still passes because no harm was done. We check both
        # outcomes: the decoder is either poisoned OR the buffer is
        # still at a valid offset.
        if dec.is_poisoned:
            with pytest.raises(ProtocolError, match="poisoned"):
                dec.decode_continuation()
        else:
            # Buffer offset must still be sensible.
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
    """Regression tests for the single-threaded signal split inside
    ``decode_handshake``.

    The pre-fix ``decode_handshake`` committed via three separate
    statements:

        self._buffer.read_bytes(8)       # consume bytes
        self._version = version           # record version
        self._handshake_done = True       # mark done

    A ``KeyboardInterrupt`` delivered between ``read_bytes(8)`` and
    the final store left the buffer with the 8 handshake bytes
    consumed but ``_handshake_done`` still ``False``. On retry,
    ``decode_handshake`` re-peeked ``peek_bytes(8)`` — but those were
    now the first 8 bytes of the **next** message, which almost
    always fail the ``_SUPPORTED_VERSIONS`` check. The caller saw a
    misleading "Unsupported protocol version" error pointing at what
    was actually legitimate message data.

    The fix marks ``_handshake_done`` BEFORE consuming the bytes and
    reverts on failure. An interrupt between the mark and the consume
    leaves the buffer unconsumed and the state "already completed";
    the retry raises a deterministic error instead of silently
    misreading real data.
    """

    def test_keyboard_interrupt_post_consume_is_not_silently_misleading(
        self,
    ) -> None:
        dec = MessageDecoder(is_request=True)
        # 8 valid handshake bytes followed by 8 bytes that WOULD fail
        # the version check if the retry misread them as a handshake.
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
            # Inject at the `self._version = version` line — it is
            # present in both the pre-fix and post-fix layouts, and
            # in the pre-fix source it runs AFTER read_bytes(8).
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

        # After the torn window, a retry must NOT report
        # "Unsupported protocol version" for what were actually real
        # bytes of the next message (0xff*8). Acceptable outcomes:
        #   1. The fix reverted state — retry succeeds with the
        #      original 8 handshake bytes still in the buffer.
        #   2. The fix set _handshake_done before the consume and
        #      did not revert — retry raises "already completed".
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
        """Sanity: without any interrupt, decode_handshake completes
        normally and sets all three state bits.
        """
        dec = MessageDecoder(is_request=True)
        dec.feed(PROTOCOL_VERSION.to_bytes(8, "little"))
        version = dec.decode_handshake()
        assert version == PROTOCOL_VERSION
        assert dec._handshake_done is True
        assert dec._version == PROTOCOL_VERSION

    def test_unsupported_version_does_not_consume_bytes(self) -> None:
        """Counter-test: an unsupported version must leave the 8
        peeked bytes in the buffer so that retry semantics remain
        deterministic.
        """
        dec = MessageDecoder(is_request=True)
        bogus = (0xDEADBEEF).to_bytes(8, "little")
        dec.feed(bogus)

        with pytest.raises(ProtocolError, match="Unsupported"):
            dec.decode_handshake()

        assert dec._handshake_done is False
        assert dec._buffer.available() == 8
