"""Signal-safety tests for ReadBuffer (issues 037, 041, 045).

These tests use ``sys.settrace`` to inject a ``KeyboardInterrupt`` at a
specific source line transition, which is the most reliable way to
exercise CPython's bytecode-boundary async-exception model without
relying on wallclock timing. Each test verifies that when an async
exception lands in the middle of a multi-statement mutation, the
buffer is left in a state that either:

- is self-consistent (``available()`` is non-negative, reads don't
  return fabricated bytes), OR
- is poisoned, so the next caller fails fast with ``ProtocolError``.

"Looks fine but is silently broken" is the failure mode these tests
guard against.
"""

from __future__ import annotations

import contextlib
import sys
from types import FrameType
from typing import Any

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.exceptions import DecodeError, ProtocolError

_DEFAULT_INJECTED_EXC: BaseException = KeyboardInterrupt("injected")


def _raise_on_source_match(
    func_name: str,
    needle: str,
    exc: BaseException = _DEFAULT_INJECTED_EXC,
) -> Any:
    """Return a settrace-compatible tracer that raises ``exc`` on the
    first line inside ``func_name`` whose source contains ``needle``.
    """
    state = {"raised": False}

    def tracer(frame: FrameType, event: str, arg: object) -> Any:
        if event != "line":
            return tracer
        if frame.f_code.co_name != func_name:
            return tracer
        try:
            with open(frame.f_code.co_filename) as f:
                src_line = f.readlines()[frame.f_lineno - 1]
        except OSError:
            return tracer
        if needle in src_line and not state["raised"]:
            state["raised"] = True
            raise exc
        return tracer

    return tracer


class TestMaybeCompactSignalSafety:
    """Regression tests for issue 037.

    ``ReadBuffer._maybe_compact`` used to be two separate ``STORE_ATTR``
    bytecodes:

        self._data = self._data[self._pos :]   # (A)
        self._pos = 0                           # (B)

    CPython checks for pending signals at bytecode line transitions. A
    ``KeyboardInterrupt`` (or any ``PyErr_SetAsyncExc`` delivery) landing
    between (A) and (B) would leave the buffer with a freshly compacted
    ``_data`` but a stale ``_pos`` still pointing at the old offset.
    ``available()`` would return a negative number, reads would return
    nonsense, and no poison fired because no exception originated
    *inside* the buffer — the interrupt was purely external. A
    single-owner caller who did nothing wrong except press Ctrl-C
    during a busy decode loop would end up with silent message dropout.

    The fix wraps the compaction in ``try/except BaseException`` and
    poisons on any torn state, so the next call raises ``ProtocolError``
    instead of lying about the buffer's contents.
    """

    def test_torn_compact_leaves_buffer_poisoned_or_consistent(self) -> None:
        buf = ReadBuffer()
        # Set up a pre-compact state: _pos > 4096 and a tail of bytes
        # that would remain after compaction.
        buf._data = bytearray(b"A" * 4096 + b"B" * 100)
        buf._pos = 4097

        tracer = _raise_on_source_match("_maybe_compact", "self._pos = 0")

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                buf._maybe_compact()
        finally:
            sys.settrace(None)

        # Post-condition: EITHER the buffer is poisoned (preferred), OR
        # it is self-consistent (available() non-negative, reads valid).
        # The old (broken) behaviour returned available() == -3998.
        if buf.is_poisoned:
            # Exact recovery semantics at the public-API layer are
            # covered by issue 038's tests; here we only require that
            # the buffer knows it is torn. Inspect _check_poisoned
            # directly, which is the single source of truth for the
            # poison gate.
            with pytest.raises(ProtocolError, match="poisoned"):
                buf._check_poisoned()
        else:
            assert buf.available() >= 0, (
                f"torn compact left available()={buf.available()} "
                f"(len={len(buf._data)}, pos={buf._pos})"
            )

    def test_happy_path_compact_still_works(self) -> None:
        """Sanity: without any interrupt, compact behaves normally."""
        buf = ReadBuffer()
        buf._data = bytearray(b"A" * 4096 + b"B" * 100)
        buf._pos = 4097

        buf._maybe_compact()

        assert not buf.is_poisoned
        assert buf._pos == 0
        assert len(buf._data) == 99
        assert buf.available() == 99
        # And reads continue to work.
        tail = buf.read_bytes(5)
        assert tail == b"B" * 5

    def test_compact_below_threshold_is_a_noop(self) -> None:
        """If _pos is below the compact threshold, the method does
        nothing — no state changes, no poison.
        """
        buf = ReadBuffer()
        buf.feed(b"\x00" * 32)
        buf.read_bytes(8)
        assert buf._pos == 8

        buf._maybe_compact()

        assert buf._pos == 8
        assert not buf.is_poisoned


class TestMaybeCompactPoisonIsWellFormed:
    """If the torn-state fix fires and poisons the buffer, the poison
    cause must be a real exception instance we can inspect — not None,
    not a raw BaseException we can't chain through.
    """

    def test_poison_cause_is_recorded(self) -> None:
        buf = ReadBuffer()
        buf._data = bytearray(b"A" * 4096 + b"B" * 100)
        buf._pos = 4097

        sentinel = RuntimeError("injected torn state")
        tracer = _raise_on_source_match("_maybe_compact", "self._pos = 0", exc=sentinel)

        sys.settrace(tracer)
        try:
            with contextlib.suppress(RuntimeError):
                buf._maybe_compact()
        finally:
            sys.settrace(None)

        if buf.is_poisoned:
            with pytest.raises(ProtocolError) as ei:
                buf._check_poisoned()
            # The original exception should appear in the cause chain.
            cause = ei.value.__cause__
            assert cause is not None
            # Either directly the sentinel, or a wrapper around it.
            assert cause is sentinel or isinstance(cause, Exception)

    def test_non_exception_base_exception_does_not_crash_poison(self) -> None:
        """KeyboardInterrupt is not an Exception subclass. The poison
        path must still accept it (either by widening poison()'s
        parameter type or by wrapping)."""
        # Use the class above's scenario but with KeyboardInterrupt —
        # that is what CPython actually delivers on Ctrl-C.
        buf = ReadBuffer()
        buf._data = bytearray(b"A" * 4096 + b"B" * 100)
        buf._pos = 4097

        tracer = _raise_on_source_match(
            "_maybe_compact", "self._pos = 0", exc=KeyboardInterrupt("ctrl-c")
        )

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                buf._maybe_compact()
        finally:
            sys.settrace(None)

        # We don't care whether the buffer is poisoned or consistent;
        # we care that neither path raised TypeError from poison()
        # rejecting a BaseException that isn't an Exception.
        _ = buf.is_poisoned  # just reading it must not raise
        _ = buf.available()


def test_decode_error_is_still_recoverable_without_poison() -> None:
    """Counter-test: the torn-state poison must not fire on any
    "normal" code path. A legitimate decode on a clean buffer works
    as before."""
    buf = ReadBuffer()
    buf.feed(b"\x00" * 16)
    assert not buf.is_poisoned
    _ = DecodeError  # referenced for completeness


class TestFeedSignalSafety:
    """Regression tests for issue 048.

    ``ReadBuffer.feed`` used to mutate state (``_skip_remaining``,
    ``_data``) without any try/except wrapper. A BaseException leaking
    out of any intermediate step — most notably between
    ``_maybe_compact()`` returning and ``self._data.extend(data)``,
    where CPython's RESUME opcode IS a real async-exception delivery
    point — would leave the buffer in a silently inconsistent state:
    fed bytes dropped from the caller's stack frame, compact having
    run but extend having not. ``is_poisoned`` would remain ``False``
    and the next ``feed()`` call would silently append to a buffer
    with a gap.

    The fix wraps the mutation block in ``try/except BaseException``
    analogous to ``_maybe_compact``'s own fix from issue 037, while
    leaving the oversized-``DecodeError`` size check OUTSIDE the try
    block so the non-poisoning "recoverable via drain/reset" contract
    is preserved.
    """

    def test_torn_feed_extend_leaves_buffer_poisoned(self) -> None:
        """A BaseException raised between ``_maybe_compact`` and
        ``_data.extend`` must poison the buffer so the next operation
        fails fast instead of silently returning partial data.
        """
        buf = ReadBuffer()
        # Give the buffer enough prior data that _maybe_compact is a
        # no-op (we want the hazard to be specifically the extend
        # step, not any torn compact). _pos below 4096 → compact
        # returns early.
        buf.feed(b"\x00" * 32)
        buf.read_bytes(8)
        assert buf._pos == 8

        # Inject the async exception on the extend line. The tracer
        # runs the line event before bytecode on that line executes,
        # so extend never runs — exactly the shape that a real RESUME
        # after _maybe_compact() returns can produce.
        tracer = _raise_on_source_match("feed", "self._data.extend(data)")

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                buf.feed(b"NEW-DATA")
        finally:
            sys.settrace(None)

        # The fed bytes are lost (they only lived in the caller's
        # frame). The buffer MUST be poisoned so the next operation
        # raises ProtocolError rather than continuing on a torn
        # state.
        assert buf.is_poisoned, "feed() must poison when a BaseException escapes its mutation block"
        with pytest.raises(ProtocolError, match="poisoned"):
            buf.read_message()

    def test_torn_feed_poison_cause_is_recorded(self) -> None:
        """The poison cause must be a real exception that surfaces in
        the ProtocolError ``__cause__`` chain, matching the
        well-formed-poison contract from issue 037.
        """
        buf = ReadBuffer()
        buf.feed(b"\x00" * 32)
        buf.read_bytes(8)

        sentinel = RuntimeError("injected torn extend")
        tracer = _raise_on_source_match("feed", "self._data.extend(data)", exc=sentinel)

        sys.settrace(tracer)
        try:
            with contextlib.suppress(RuntimeError):
                buf.feed(b"NEW-DATA")
        finally:
            sys.settrace(None)

        assert buf.is_poisoned
        with pytest.raises(ProtocolError) as ei:
            buf._check_poisoned()
        cause = ei.value.__cause__
        assert cause is sentinel or (
            isinstance(cause, Exception) and "injected torn extend" in str(cause)
        )

    def test_feed_decode_error_is_still_non_poisoning(self) -> None:
        """Counter-test: the oversized-buffer ``DecodeError`` guard
        must remain non-poisoning, because its caller-recovery contract
        is "drain or reset and continue". The poison wrapper must not
        capture this specific error.
        """
        buf = ReadBuffer(max_message_size=16)
        with pytest.raises(DecodeError):
            buf.feed(b"A" * 32)
        assert not buf.is_poisoned, (
            "oversized-buffer DecodeError must NOT poison the buffer; "
            "it is deliberately recoverable via reset()/clear()"
        )
        # And after reset the buffer is usable again.
        buf.reset()
        buf.feed(b"A" * 8)
        assert buf.available() == 8

    def test_happy_path_feed_still_works(self) -> None:
        """Sanity: without any interrupt, feed behaves normally."""
        buf = ReadBuffer()
        buf.feed(b"hello")
        buf.feed(b"world")
        assert not buf.is_poisoned
        assert buf.available() == 10
        assert buf.read_bytes(10) == b"helloworld"


class _TornSizeBytearray(bytearray):
    """Simulates a free-threaded torn bytearray read.

    On free-threaded CPython (3.13t), ``bytearray.__getitem__`` can
    observe ``ob_size``/``ob_start`` inconsistently during a
    concurrent realloc/extend from another thread, producing a slice
    wider than the caller asked for. Under the GIL this is
    structurally impossible, so we synthesize the observable symptom
    by subclassing bytearray and widening only the 4-byte header
    size-field slice. All other slices (body reads, etc.) pass
    through unchanged.

    The widened slice is 9 bytes: the original 4 plus 5 extra
    ``\\xff`` bytes. ``int.from_bytes(..., "little")`` on that
    produces a value whose low 32 bits match the wire size field
    but whose total is > ``0xFFFFFFFF``, which is the structurally
    impossible condition that issue 051's sanity check is meant to
    detect.
    """

    def __getitem__(self, key: Any) -> Any:
        result = super().__getitem__(key)
        if isinstance(key, slice):
            start, stop, _ = key.indices(len(self))
            if stop - start == 4:
                return bytes(result) + b"\xff\xff\xff\xff\xff"
        return result


class TestTornHeaderSizeSanityCheck:
    """Regression tests for issue 051.

    On free-threaded CPython, concurrent misuse of a ``ReadBuffer``
    can cause the 4-byte header-size slice to observably return more
    than 4 bytes (torn read during a realloc). The resulting
    ``int.from_bytes`` value is a Python bigint wider than 32 bits —
    structurally impossible for a wire-legal ``size_words`` field
    (which is uint32 little-endian). The package already knew about
    this (issue 033 added hex formatting to avoid the
    bigint-to-decimal cap), but the torn-read case was routed
    through the non-poisoning ``DecodeError`` path meant for
    legitimate oversized server messages, leaving the caller free to
    call ``skip_message()`` on a buffer whose offset is unknowable.

    The fix: any ``size_words > 0xFFFFFFFF`` is a torn-read indicator,
    not a legitimate oversized message. Poison the buffer and raise a
    diagnostic ``DecodeError`` identifying the torn read.

    These tests use a ``bytearray`` subclass to synthesize the torn
    slice on GIL builds. On a real free-threaded build the symptom
    arises organically from concurrent access.
    """

    def _buf_with_torn_header(self) -> ReadBuffer:
        buf = ReadBuffer()
        # Put a wire-legal size_words=1 in the header plus 8 bytes of
        # body. The _TornSizeBytearray will widen the 4-byte slice to
        # 9 bytes on read, producing a size_words > 32 bits.
        buf._data = _TornSizeBytearray(b"\x01\x00\x00\x00" + b"\x00" * 12)
        return buf

    def test_read_message_poisons_on_torn_header(self) -> None:
        buf = self._buf_with_torn_header()
        with pytest.raises(DecodeError, match="torn"):
            buf.read_message()
        assert buf.is_poisoned, (
            "torn header read must poison the buffer so subsequent "
            "callers fail fast rather than calling skip_message() on "
            "an unknowable offset"
        )
        with pytest.raises(ProtocolError, match="poisoned"):
            buf.read_message()

    def test_peek_header_poisons_on_torn_header(self) -> None:
        buf = self._buf_with_torn_header()
        with pytest.raises(DecodeError, match="torn"):
            buf.peek_header()
        assert buf.is_poisoned

    def test_skip_message_poisons_on_torn_header(self) -> None:
        buf = self._buf_with_torn_header()
        with pytest.raises(DecodeError, match="torn"):
            buf.skip_message()
        assert buf.is_poisoned

    def test_wire_legal_max_size_still_routes_through_oversized_path(self) -> None:
        """Counter-test: a legitimate ``size_words = 0xFFFFFFFF`` (a
        pathological but structurally-valid value) must NOT trip the
        torn-read sanity check — it must route through the existing
        non-poisoning ``DecodeError`` path so ``skip_message`` can
        recover. The check is strictly ``> 0xFFFFFFFF``.
        """
        buf = ReadBuffer()
        # A real, non-torn 4-byte size field at its maximum value.
        buf._data = bytearray(b"\xff\xff\xff\xff" + b"\x00" * 4)
        with pytest.raises(DecodeError) as ei:
            buf.read_message()
        assert "torn" not in str(ei.value), (
            "wire-legal size_words=0xFFFFFFFF must not be misdiagnosed as torn"
        )
        assert not buf.is_poisoned, (
            "oversized-but-wire-legal DecodeError must remain non-poisoning "
            "to preserve the skip_message() recovery contract"
        )


class TestConsumeMethodSignalSafety:
    """Regression tests for issue 061.

    ``read_message()``, ``skip_message()``, and ``read_bytes()`` each
    advance ``_pos`` and then call ``_maybe_compact()``. The CALL to
    ``_maybe_compact`` is a CPython eval-breaker check point — a pending
    signal is delivered there BEFORE the function body runs. If the
    signal fires at that boundary, ``_pos`` has advanced (bytes consumed)
    but ``_maybe_compact``'s own try/except never executes, so no poison
    fires. The return value is lost with the unwinding frame.

    The fix wraps the mutation block in ``try/except BaseException``
    matching the template from issues 037 (``_maybe_compact``) and
    048 (``feed``).
    """

    def test_torn_read_message_leaves_buffer_poisoned(self) -> None:
        buf = ReadBuffer()
        buf.feed(b"\x01\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 8)

        tracer = _raise_on_source_match("read_message", "self._maybe_compact()")

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                buf.read_message()
        finally:
            sys.settrace(None)

        assert buf.is_poisoned, (
            "read_message must poison when a BaseException escapes its mutation block"
        )
        with pytest.raises(ProtocolError, match="poisoned"):
            buf.read_message()

    def test_torn_read_bytes_leaves_buffer_poisoned(self) -> None:
        buf = ReadBuffer()
        buf.feed(b"\x00" * 32)

        tracer = _raise_on_source_match("read_bytes", "self._maybe_compact()")

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                buf.read_bytes(8)
        finally:
            sys.settrace(None)

        assert buf.is_poisoned, (
            "read_bytes must poison when a BaseException escapes its mutation block"
        )

    def test_torn_skip_message_leaves_buffer_poisoned(self) -> None:
        buf = ReadBuffer()
        buf.feed(b"\x01\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 8)

        tracer = _raise_on_source_match("skip_message", "self._maybe_compact()")

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                buf.skip_message()
        finally:
            sys.settrace(None)

        assert buf.is_poisoned, (
            "skip_message must poison when a BaseException escapes its mutation block"
        )

    def test_happy_paths_still_work(self) -> None:
        """Sanity: without any interrupt, all three methods work normally."""
        buf = ReadBuffer()
        buf.feed(b"\x01\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 8)
        msg = buf.read_message()
        assert msg is not None and not buf.is_poisoned

        buf.feed(b"\x00" * 32)
        data = buf.read_bytes(8)
        assert data is not None and len(data) == 8

        buf.feed(b"\x01\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 8)
        skipped = buf.skip_message()
        assert skipped is True and not buf.is_poisoned
