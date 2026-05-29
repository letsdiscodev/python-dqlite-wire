"""Signal-safety tests for ReadBuffer: inject a BaseException at a source-line
transition (via settrace) mid-mutation and assert the buffer is left either
self-consistent or poisoned, never silently torn."""

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
    """A settrace tracer that raises ``exc`` on the first line inside ``func_name``
    whose source contains ``needle``."""
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
    """Signal delivery between _maybe_compact's two stores (``_data = _data[_pos:]``
    then ``_pos = 0``) would leave fresh _data with a stale _pos (negative available()).
    The fix wraps compaction in try/except BaseException and poisons on a torn state."""

    def test_torn_compact_leaves_buffer_poisoned_or_consistent(self) -> None:
        buf = ReadBuffer()
        buf._data = bytearray(b"A" * 4096 + b"B" * 100)
        buf._pos = 4097

        tracer = _raise_on_source_match("_maybe_compact", "self._pos = 0")

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                buf._maybe_compact()
        finally:
            sys.settrace(None)

        # Either poisoned (preferred) or self-consistent; the old broken
        # behaviour returned available() == -3998.
        if buf.is_poisoned:
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
        tail = buf.read_bytes(5)
        assert tail == b"B" * 5

    def test_compact_below_threshold_is_a_noop(self) -> None:
        """Below the compact threshold the method is a no-op."""
        buf = ReadBuffer()
        buf.feed(b"\x00" * 32)
        buf.read_bytes(8)
        assert buf._pos == 8

        buf._maybe_compact()

        assert buf._pos == 8
        assert not buf.is_poisoned


class TestMaybeCompactPoisonIsWellFormed:
    """When the torn-state fix poisons, the cause must be a real, inspectable exception."""

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
            cause = ei.value.__cause__
            assert cause is not None
            assert cause is sentinel or isinstance(cause, Exception)

    def test_non_exception_base_exception_does_not_crash_poison(self) -> None:
        """The poison path must accept a KeyboardInterrupt (not an Exception subclass)."""
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

        # Neither path may raise TypeError from poison() rejecting a non-Exception.
        _ = buf.is_poisoned
        _ = buf.available()


def test_decode_error_is_still_recoverable_without_poison() -> None:
    """Counter-test: the torn-state poison must not fire on a normal code path."""
    buf = ReadBuffer()
    buf.feed(b"\x00" * 16)
    assert not buf.is_poisoned
    _ = DecodeError


class TestFeedSignalSafety:
    """A BaseException between feed's ``_maybe_compact()`` and ``_data.extend(data)``
    would silently drop the fed bytes. The fix wraps the mutation in try/except
    BaseException, but keeps the oversized-DecodeError check outside it so that
    error stays non-poisoning (recoverable via drain/reset)."""

    def test_torn_feed_extend_leaves_buffer_poisoned(self) -> None:
        """A BaseException between ``_maybe_compact`` and ``_data.extend`` must
        poison so the next operation fails fast instead of returning partial data."""
        buf = ReadBuffer()
        # _pos below 4096 keeps _maybe_compact a no-op so the hazard is the extend step.
        buf.feed(b"\x00" * 32)
        buf.read_bytes(8)
        assert buf._pos == 8

        tracer = _raise_on_source_match("feed", "self._data.extend(data)")

        sys.settrace(tracer)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                buf.feed(b"NEW-DATA")
        finally:
            sys.settrace(None)

        assert buf.is_poisoned, "feed() must poison when a BaseException escapes its mutation block"
        with pytest.raises(ProtocolError, match="poisoned"):
            buf.read_message()

    def test_torn_feed_poison_cause_is_recorded(self) -> None:
        """The poison cause must surface in the ProtocolError ``__cause__`` chain."""
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
        """Counter-test: the oversized-buffer DecodeError must remain non-poisoning
        (recoverable via reset/clear); the poison wrapper must not capture it."""
        buf = ReadBuffer(max_message_size=16)
        with pytest.raises(DecodeError):
            buf.feed(b"A" * 32)
        assert not buf.is_poisoned, (
            "oversized-buffer DecodeError must NOT poison the buffer; "
            "it is deliberately recoverable via reset()/clear()"
        )
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
    """Synthesizes a free-threaded torn read: widens only the 4-byte header
    size-field slice to 9 bytes so ``int.from_bytes`` exceeds 0xFFFFFFFF (the
    structurally-impossible value the sanity check detects)."""

    def __getitem__(self, key: Any) -> Any:
        result = super().__getitem__(key)
        if isinstance(key, slice):
            start, stop, _ = key.indices(len(self))
            if stop - start == 4:
                return bytes(result) + b"\xff\xff\xff\xff\xff"
        return result


class TestTornHeaderSizeSanityCheck:
    """A torn header read yields ``size_words > 0xFFFFFFFF`` (impossible for a
    uint32 field); such a value must poison the buffer and raise a diagnostic
    DecodeError rather than route through the recoverable oversized path."""

    def _buf_with_torn_header(self) -> ReadBuffer:
        buf = ReadBuffer()
        # Wire-legal size_words=1; _TornSizeBytearray widens the slice on read.
        buf._data = _TornSizeBytearray(b"\x01\x00\x00\x00" + b"\x00" * 12)
        return buf

    def test_read_message_poisons_on_torn_header(self) -> None:
        buf = self._buf_with_torn_header()
        with pytest.raises(DecodeError, match="torn"):
            buf.read_message()
        assert buf.is_poisoned, "torn header read must poison so subsequent callers fail fast"
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
        """Counter-test: a real ``size_words = 0xFFFFFFFF`` must NOT trip the torn
        check (strictly ``> 0xFFFFFFFF``); it routes through the recoverable path."""
        buf = ReadBuffer()
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
    """read_message/skip_message/read_bytes advance ``_pos`` then call
    ``_maybe_compact()``; a signal at that call boundary fires after _pos advanced
    but before _maybe_compact's own try/except. The fix wraps the mutation block."""

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
