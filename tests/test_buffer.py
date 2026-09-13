"""Tests for ReadBuffer: feed/skip/compact, oversize rejection and poisoning."""

from __future__ import annotations

import pickle
from typing import cast

import pytest

from dqlitewire.buffer import _COMPACT_THRESHOLD, ReadBuffer
from dqlitewire.codec import MessageDecoder, MessageEncoder
from dqlitewire.constants import HEADER_SIZE, WORD_SIZE, ResponseType, ValueType
from dqlitewire.exceptions import DecodeError, PoisonedError, ProtocolError
from dqlitewire.messages import LeaderRequest, LeaderResponse
from dqlitewire.messages.responses import RowsResponse


class TestReadBuffer:
    def test_empty(self) -> None:
        buf = ReadBuffer()
        assert buf.available() == 0
        assert not buf.has_message()

    def test_rejects_zero_max_message_size(self) -> None:
        with pytest.raises(ValueError, match="max_message_size must be >= 1"):
            ReadBuffer(max_message_size=0)

    def test_rejects_negative_max_message_size(self) -> None:
        with pytest.raises(ValueError, match="max_message_size must be >= 1"):
            ReadBuffer(max_message_size=-1)

    def test_feed(self) -> None:
        buf = ReadBuffer()
        buf.feed(b"hello")
        assert buf.available() == 5

    def test_multiple_feeds(self) -> None:
        buf = ReadBuffer()
        buf.feed(b"hel")
        buf.feed(b"lo")
        assert buf.available() == 5

    def test_read_bytes(self) -> None:
        buf = ReadBuffer()
        buf.feed(b"hello world")
        data = buf.read_bytes(5)
        assert data == b"hello"
        assert buf.available() == 6

    def test_read_bytes_not_enough(self) -> None:
        buf = ReadBuffer()
        buf.feed(b"hi")
        data = buf.read_bytes(5)
        assert data is None
        assert buf.available() == 2

    def test_read_bytes_negative_n_raises(self) -> None:
        """read_bytes with negative n must raise, not corrupt _pos."""
        import pytest

        buf = ReadBuffer()
        buf.feed(b"hello world")
        with pytest.raises(ValueError, match="non-negative"):
            buf.read_bytes(-1)
        assert buf.available() == 11

    def test_peek_bytes_negative_n_raises(self) -> None:
        """peek_bytes with negative n must raise."""
        import pytest

        buf = ReadBuffer()
        buf.feed(b"hello world")
        with pytest.raises(ValueError, match="non-negative"):
            buf.peek_bytes(-1)
        assert buf.available() == 11

    def test_has_message_incomplete(self) -> None:
        buf = ReadBuffer()
        buf.feed(b"\x01\x00\x00\x00\x00\x00\x00\x00")  # header only, body missing
        assert not buf.has_message()

    def test_has_message_complete(self) -> None:
        msg = LeaderRequest()
        encoded = msg.encode()

        buf = ReadBuffer()
        buf.feed(encoded)
        assert buf.has_message()

    def test_peek_header(self) -> None:
        msg = LeaderResponse(node_id=1, address="test")
        encoded = msg.encode()

        buf = ReadBuffer()
        buf.feed(encoded)

        header = buf.peek_header()
        assert header is not None
        size_words, msg_type, schema = header
        assert msg_type == LeaderResponse.MSG_TYPE
        assert schema == 0

    def test_peek_header_not_enough(self) -> None:
        buf = ReadBuffer()
        buf.feed(b"\x00\x00")
        header = buf.peek_header()
        assert header is None

    def test_read_message(self) -> None:
        msg = LeaderRequest()
        encoded = msg.encode()

        buf = ReadBuffer()
        buf.feed(encoded)

        data = buf.read_message()
        assert data == encoded
        assert buf.available() == 0

    def test_read_message_not_complete(self) -> None:
        buf = ReadBuffer()
        buf.feed(b"\x01\x00\x00\x00\x00\x00\x00\x00")  # header says 1 word, body missing
        data = buf.read_message()
        assert data is None

    def test_read_multiple_messages(self) -> None:
        msg1 = LeaderRequest()
        msg2 = LeaderResponse(node_id=1, address="x")

        buf = ReadBuffer()
        buf.feed(msg1.encode() + msg2.encode())

        data1 = buf.read_message()
        assert data1 == msg1.encode()

        data2 = buf.read_message()
        assert data2 == msg2.encode()

    def test_clear(self) -> None:
        buf = ReadBuffer()
        buf.feed(b"data")
        buf.clear()
        assert buf.available() == 0

    def test_clear_also_unpoisons(self) -> None:
        """Regression: clear() must also un-poison (like reset()), not leave a
        half-fresh buffer that still raises ProtocolError."""
        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer()
        buf.feed(b"\x00" * 16)
        buf.poison(DecodeError("boom"))
        assert buf.is_poisoned

        buf.clear()

        assert not buf.is_poisoned
        buf.feed(b"\x01\x00\x00\x00\x00\x00\x00\x00")
        assert buf.available() == 8

    def test_public_api_honors_poison(self) -> None:
        """Every mutating/consuming public method must raise ProtocolError when
        poisoned; observers and recovery primitives stay callable."""
        import pytest

        from dqlitewire.exceptions import DecodeError, ProtocolError

        buf = ReadBuffer()
        buf.poison(DecodeError("original cause"))

        cases: list[tuple[str, object]] = [
            ("feed", lambda: buf.feed(b"x" * 8)),
            ("read_message", lambda: buf.read_message()),
            ("skip_message", lambda: buf.skip_message()),
            ("read_bytes", lambda: buf.read_bytes(4)),
            ("peek_bytes", lambda: buf.peek_bytes(4)),
            ("peek_header", lambda: buf.peek_header()),
        ]
        for name, call in cases:
            with pytest.raises(ProtocolError, match="poisoned") as ei:
                call()  # type: ignore[operator]
            assert isinstance(ei.value.__cause__, DecodeError), name

        _ = buf.available()
        _ = buf.has_message()
        _ = buf.is_poisoned
        _ = buf.is_skipping

        buf.reset()
        assert not buf.is_poisoned
        buf.feed(b"\x00" * 8)

    def test_has_message_is_total_on_oversized(self) -> None:
        """has_message() must not raise — oversized headers surface at consume time
        so the `while has_message(): decode()` pattern needs no try/except."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        header = struct.pack("<IBBH", 1000, 0, 0, 0)  # 8000-byte body > 1024 limit
        buf.feed(header)

        assert buf.has_message() is True

        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()

    def test_peek_header_validates_size(self) -> None:
        """peek_header() must reject oversized headers: its size_words return is
        attacker-controlled and callers might preallocate on it."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        header = struct.pack("<IBBH", 1000, 0, 0, 0)  # 8000-byte body > 1024 limit
        buf.feed(header)

        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.peek_header()

    def test_peek_header_on_valid_header(self) -> None:
        """peek_header() returns the parsed tuple for a valid header without advancing _pos."""
        import struct

        buf = ReadBuffer(max_message_size=1024)
        header = struct.pack("<IBBH", 1, 5, 0, 0)
        buf.feed(header)
        result = buf.peek_header()
        assert result == (1, 5, 0)
        assert buf.peek_header() == (1, 5, 0)

    def test_peek_header_partial_returns_none(self) -> None:
        """With fewer than HEADER_SIZE bytes buffered, peek_header() returns None."""
        buf = ReadBuffer()
        buf.feed(b"\x00\x00")
        assert buf.peek_header() is None

    def test_peek_header_does_not_advance_pos(self) -> None:
        """peek_header() must leave the buffer position untouched."""
        import struct

        buf = ReadBuffer(max_message_size=1024)
        buf.feed(struct.pack("<IBBH", 1, 5, 0, 0))
        assert buf.available() == 8
        buf.peek_header()
        assert buf.available() == 8
        buf.peek_header()
        assert buf.available() == 8

    def test_peek_header_at_exact_size_boundary(self) -> None:
        """total_size == max_message_size is accepted; one byte over raises (off-by-one)."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        # HEADER_SIZE (8) + 1 word (8) == 16 is the exact boundary.
        buf_ok = ReadBuffer(max_message_size=16)
        buf_ok.feed(struct.pack("<IBBH", 1, 0, 0, 0))
        assert buf_ok.peek_header() == (1, 0, 0)

        # size_words=2 -> total_size=24 -> 8 over the limit.
        buf_over = ReadBuffer(max_message_size=16)
        buf_over.feed(struct.pack("<IBBH", 2, 0, 0, 0))
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf_over.peek_header()

    def test_peek_header_and_has_message_disagree_on_oversized(self) -> None:
        """Deliberate asymmetry: has_message() is total (returns True on oversized),
        peek_header() raises since its return is attacker-controlled allocation input."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        buf.feed(struct.pack("<IBBH", 1000, 0, 0, 0))  # 8000 > 1024

        assert buf.has_message() is True

        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.peek_header()

    def test_rejects_oversized_message(self) -> None:
        """Oversized messages surface at read_message(), not has_message()."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        header = struct.pack("<IBBH", 1000, 0, 0, 0)  # 8000-byte body > 1024 limit
        buf.feed(header)
        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()

    def test_skip_message_poisons_after_capped_oversized(self) -> None:
        """A capped oversized skip discards fewer bytes than the peer sent, leaving
        the stream desynchronized, so the buffer must poison once the skip completes."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError, ProtocolError

        buf = ReadBuffer(max_message_size=1024)
        oversized_header = struct.pack("<IBBH", 1000, 0, 0, 0)  # 8000-byte body > 1024

        buf.feed(oversized_header + b"\xab" * 500)

        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()

        assert buf.skip_message() is False  # partial skip

        remaining = buf._skip_remaining
        buf.feed(b"\xab" * remaining)

        assert not buf.is_skipping
        assert buf.is_poisoned

        with pytest.raises(ProtocolError, match="poisoned"):
            buf.read_message()

    def test_capped_oversized_skip_poisons_buffer(self) -> None:
        """Capped skip of oversized message must poison: fewer bytes discarded than
        the peer sent leaves the stream desynchronized."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError, ProtocolError

        buf = ReadBuffer(max_message_size=64)
        oversized_header = struct.pack("<IBBH", 200, 0, 0, 0)  # 1600-byte body >> 64
        buf.feed(oversized_header)

        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()

        buf.skip_message()

        remaining = buf._skip_remaining
        buf.feed(b"\x00" * remaining)
        assert not buf.is_skipping
        assert buf.is_poisoned

        with pytest.raises(ProtocolError, match="poisoned"):
            buf.read_message()

    def test_single_feed_completes_capped_skip_and_poisons(self) -> None:
        """A single feed() that completes a capped skip poisons the buffer."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError, ProtocolError

        buf = ReadBuffer(max_message_size=64)
        oversized_header = struct.pack("<IBBH", 200, 0, 0, 0)  # 1600-byte body >> 64

        buf.feed(oversized_header)
        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()
        assert buf.skip_message() is False
        assert buf.is_skipping

        remaining = buf._skip_remaining
        combined = b"\xcc" * remaining + b"\x00" * 16

        buf.feed(combined)

        assert not buf.is_skipping
        assert buf.is_poisoned

        with pytest.raises(ProtocolError, match="poisoned"):
            buf.read_message()

    def test_skip_oversized_across_multiple_feeds_poisons(self) -> None:
        """Oversized skip across multiple feeds poisons once the capped skip completes."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError, ProtocolError

        buf = ReadBuffer(max_message_size=64)
        oversized_header = struct.pack("<IBBH", 200, 0, 0, 0)  # 1600-byte body >> 64

        buf.feed(oversized_header)
        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()
        assert buf.skip_message() is False

        remaining = buf._skip_remaining
        body = b"\xcc" * remaining
        while body:
            chunk = body[:20]
            body = body[20:]
            buf.feed(chunk)

        assert not buf.is_skipping
        assert buf.is_poisoned

        with pytest.raises(ProtocolError, match="poisoned"):
            buf.feed(b"\x00" * 16)

    def test_is_skipping_property(self) -> None:
        """is_skipping reflects whether an oversized skip is in progress."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=64)
        assert buf.is_skipping is False

        header = struct.pack("<IBBH", 200, 0, 0, 0)  # 1600-byte body > 64 limit
        buf.feed(header)
        assert buf.has_message() is True
        with pytest.raises(DecodeError):
            buf.read_message()

        assert buf.skip_message() is False  # partial skip
        assert buf.is_skipping is True

        buf.feed(b"\x00" * buf._skip_remaining)
        assert buf.is_skipping is False

    def test_clear_resets_skip_state(self) -> None:
        """clear() should cancel any in-progress oversized skip."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=64)
        header = struct.pack("<IBBH", 200, 0, 0, 0)  # 1600-byte body > 64 limit
        buf.feed(header)
        assert buf.has_message() is True
        with pytest.raises(DecodeError):
            buf.read_message()
        buf.skip_message()
        assert buf.is_skipping is True

        buf.clear()
        assert buf.is_skipping is False

    def test_skip_message_empty_buffer(self) -> None:
        """skip_message() on empty buffer returns False."""
        buf = ReadBuffer()
        assert buf.skip_message() is False

    def test_feed_rejects_data_exceeding_max_message_size(self) -> None:
        """feed() should raise DecodeError when buffer exceeds max_message_size."""
        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        buf.feed(b"\x00" * 512)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.feed(b"\x00" * 600)

    def test_read_bytes_triggers_compaction(self) -> None:
        """read_bytes should compact the buffer after consuming enough data."""
        buf = ReadBuffer()
        buf.feed(b"\x00" * 5000)
        buf.read_bytes(4500)
        assert buf._pos == 0
        assert len(buf._data) == 500

    def test_read_message_validates_size_independently(self) -> None:
        """read_message validates size even when has_message wasn't the last call."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        size_words = 200  # 1600-byte body > 1024 limit
        header = struct.pack("<IBBH", size_words, 0, 0, 0)
        body = b"\x00" * (size_words * 8)
        buf._data = bytearray(header + body)
        buf._pos = 0
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()

    def test_skip_message_allows_oversized_for_recovery(self) -> None:
        """skip_message must not raise on oversized (it is the recovery path); when the
        oversized body is fully present it poisons and returns False so the natural
        ``if buf.skip_message(): continue`` pattern stays correct."""
        import struct

        buf = ReadBuffer(max_message_size=1024)
        size_words = 200  # 1600-byte body > 1024 limit
        header = struct.pack("<IBBH", size_words, 0, 0, 0)
        body = b"\x00" * (size_words * 8)
        buf._data = bytearray(header + body)
        buf._pos = 0
        assert buf.skip_message() is False
        assert buf.is_poisoned

    def test_skip_message_waits_for_complete_normal_sized_message(self) -> None:
        """skip_message returns False for an incomplete normal-sized message and must
        not advance past partial data, else the stream corrupts when the rest arrives."""
        import struct

        buf = ReadBuffer(max_message_size=4096)
        header = struct.pack("<IBBH", 5, 0, 0, 0)  # 40-byte body, fits limit
        buf.feed(header + b"\x00" * 16)  # incomplete: only 16 of 40 body bytes
        assert buf.skip_message() is False
        assert buf.available() == 24

    def test_skip_message_caps_remaining_to_max_message_size(self) -> None:
        """_skip_remaining must be capped to max_message_size: an uncapped
        size_words=0xFFFFFFFF (~32 GiB) would silently discard feeds for a long time."""
        import struct

        buf = ReadBuffer(max_message_size=1024)
        header = struct.pack("<IBBH", 0xFFFFFFFF, 0, 0, 0)  # ~32 GiB claimed body
        buf.feed(header)

        buf.skip_message()
        assert buf._skip_remaining <= buf._max_message_size

    def test_feed_compacts_consumed_data(self) -> None:
        """feed() compacts consumed data (when _pos past the threshold) before extending."""
        buf = ReadBuffer(max_message_size=65536)
        buf._data = bytearray(b"\x00" * 5000)
        buf._pos = 4500  # 4500 consumed, 500 remaining
        buf.feed(b"\x01" * 100)
        assert buf._pos == 0
        assert len(buf._data) == 600

    def test_buffer_compaction(self) -> None:
        buf = ReadBuffer()
        msg = LeaderRequest()
        encoded = msg.encode()

        for _ in range(1000):
            buf.feed(encoded)
            buf.read_message()

        buf.feed(encoded)
        assert buf.has_message()

    def test_pickle_raises_typeerror(self) -> None:
        """ReadBuffer must not be picklable."""
        buf = ReadBuffer()
        buf.feed(b"\x00" * 16)
        with pytest.raises(TypeError, match="cannot pickle"):
            pickle.dumps(buf)

    def test_pickle_message_does_not_claim_connection_stream_binding(self) -> None:
        """Rejection message must not invent a "connection stream" binding the
        class does not hold (it is just a bytearray + ints)."""
        buf = ReadBuffer()
        try:
            pickle.dumps(buf)
        except TypeError as e:
            msg = str(e)
        else:
            pytest.fail("expected TypeError")
        assert "connection stream" not in msg, (
            f"rejection message must not claim a connection stream "
            f"binding the class does not hold; got: {msg!r}"
        )
        assert "ReadBuffer" in msg


class TestReadBufferDefensiveChecks:
    """Structural defenses in ``ReadBuffer`` not exercised by any happy-path test."""

    def test_feed_rejects_chunk_larger_than_double_max_message_size(self) -> None:
        """A single feed() chunk must not exceed 2 * max_message_size."""
        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=64)
        with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
            buf.feed(b"\x00" * 129)

    def test_read_message_poisons_with_runtime_error_on_base_exception(self) -> None:
        """A BaseException (KeyboardInterrupt) inside read_message poisons with a
        RuntimeError naming the call site (the BaseException-not-Exception fallback)."""
        from unittest.mock import patch

        buf = ReadBuffer(max_message_size=128)
        encoded = LeaderRequest().encode()
        buf.feed(encoded)

        # Patch ``bytes`` in the buffer module so the raise lands at the slice site.
        import dqlitewire.buffer as buffer_mod

        with (
            patch.object(buffer_mod, "bytes", side_effect=KeyboardInterrupt()),
            pytest.raises(KeyboardInterrupt),
        ):
            buf.read_message()

        assert isinstance(buf._poisoned, RuntimeError)
        assert "read_message interrupted" in str(buf._poisoned)
        assert "KeyboardInterrupt" in str(buf._poisoned)

    def test_skip_message_poisons_with_runtime_error_on_base_exception(self) -> None:
        """Sibling pin for skip_message's BaseException fallback."""
        from unittest.mock import patch

        buf = ReadBuffer(max_message_size=128)
        encoded = LeaderRequest().encode()
        buf.feed(encoded)

        with (
            patch.object(ReadBuffer, "_maybe_compact", side_effect=KeyboardInterrupt()),
            pytest.raises(KeyboardInterrupt),
        ):
            buf.skip_message()

        assert isinstance(buf._poisoned, RuntimeError)
        assert "skip_message interrupted" in str(buf._poisoned)
        assert "KeyboardInterrupt" in str(buf._poisoned)

    def test_read_bytes_poisons_with_runtime_error_on_base_exception(self) -> None:
        """Sibling pin for read_bytes's BaseException fallback."""
        from unittest.mock import patch

        buf = ReadBuffer(max_message_size=128)
        buf.feed(b"\x00" * 16)

        import dqlitewire.buffer as buffer_mod

        with (
            patch.object(buffer_mod, "bytes", side_effect=KeyboardInterrupt()),
            pytest.raises(KeyboardInterrupt),
        ):
            buf.read_bytes(8)

        assert isinstance(buf._poisoned, RuntimeError)
        assert "read_bytes interrupted" in str(buf._poisoned)
        assert "KeyboardInterrupt" in str(buf._poisoned)

    def test_read_message_poisons_with_original_exception_on_exception_subclass(
        self,
    ) -> None:
        """A regular Exception subclass (MemoryError) is stored as-is on
        ``_poisoned``, not wrapped in RuntimeError."""
        from unittest.mock import patch

        buf = ReadBuffer(max_message_size=128)
        encoded = LeaderRequest().encode()
        buf.feed(encoded)

        import dqlitewire.buffer as buffer_mod

        sentinel = MemoryError("simulated allocation failure")
        with (
            patch.object(buffer_mod, "bytes", side_effect=sentinel),
            pytest.raises(MemoryError),
        ):
            buf.read_message()

        assert buf._poisoned is sentinel

    def test_skip_message_poisons_with_original_exception_on_exception_subclass(
        self,
    ) -> None:
        """Sibling pin for skip_message's Exception branch."""
        from unittest.mock import patch

        buf = ReadBuffer(max_message_size=128)
        encoded = LeaderRequest().encode()
        buf.feed(encoded)

        sentinel = MemoryError("simulated allocation failure")
        with (
            patch.object(ReadBuffer, "_maybe_compact", side_effect=sentinel),
            pytest.raises(MemoryError),
        ):
            buf.skip_message()

        assert buf._poisoned is sentinel

    def test_read_bytes_poisons_with_original_exception_on_exception_subclass(
        self,
    ) -> None:
        """Sibling pin for read_bytes's Exception branch."""
        from unittest.mock import patch

        buf = ReadBuffer(max_message_size=128)
        buf.feed(b"\x00" * 16)

        import dqlitewire.buffer as buffer_mod

        sentinel = MemoryError("simulated allocation failure")
        with (
            patch.object(buffer_mod, "bytes", side_effect=sentinel),
            pytest.raises(MemoryError),
        ):
            buf.read_bytes(8)

        assert buf._poisoned is sentinel


# ---- merged from test_buffer_maybe_compact_half_fill_gate.py ----
# ``_maybe_compact`` only compacts when at least half the buffer is consumed,
# bounding a single memcpy to half the buffer (each byte copied at most twice over
# its lifetime) instead of memcpy'ing a near-full buffer for a small consumed prefix.


def test_compact_does_not_fire_when_below_half_fill() -> None:
    """1-MiB payload with ~5 KiB consumed (>4096 but far below half) must not compact."""
    buf = ReadBuffer()
    buf.feed(b"\x00" * (1 << 20))  # 1 MiB
    buf._pos = _COMPACT_THRESHOLD + 1024
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    assert id(buf._data) == pre_data_id, "compact fired when below half-fill; expected no-op"
    assert buf._pos == _COMPACT_THRESHOLD + 1024


def test_compact_fires_when_above_half_fill() -> None:
    """Compact runs when _pos is above 4096 AND more than half is consumed."""
    buf = ReadBuffer()
    total = 20_000
    buf.feed(b"\x00" * total)
    buf._pos = 12_000  # > 4096 AND > total / 2
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    assert id(buf._data) != pre_data_id, "compact did not fire above half-fill; expected compaction"
    assert buf._pos == 0
    assert len(buf._data) == total - 12_000


def test_compact_does_not_fire_when_pos_above_threshold_but_below_half() -> None:
    """Discriminating case: _pos above 4096 (prior gate would compact) but below
    half (new gate must not) — the motivating near-full-buffer scenario."""
    buf = ReadBuffer()
    buf.feed(b"\x00" * (1 << 20))
    buf._pos = _COMPACT_THRESHOLD + 1  # just over the prior gate
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    assert id(buf._data) == pre_data_id, (
        "compact fired with _pos just over 4096 in a 1 MiB buffer; "
        "the half-fill gate should have suppressed the copy"
    )
    assert buf._pos == _COMPACT_THRESHOLD + 1


def test_compact_no_op_below_4096() -> None:
    """The ``_pos <= _COMPACT_THRESHOLD`` guard still short-circuits."""
    buf = ReadBuffer()
    buf.feed(b"\x00" * 10000)
    buf._pos = 100
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    assert id(buf._data) == pre_data_id
    assert buf._pos == 100


def test_compact_amortises_under_burst_fill_pattern() -> None:
    """Under burst-fill, total bytes copied across compacts stays bounded
    (each byte copied at most twice over its lifetime)."""
    buf = ReadBuffer()
    total_fed = 0
    total_copied = 0
    for _ in range(10):
        buf.feed(b"\x00" * (64 * 1024))
        total_fed += 64 * 1024
        # Consume just over half of what's currently buffered.
        consumable = len(buf._data) - buf._pos
        buf._pos += consumable // 2 + 1
        about_to_copy = len(buf._data) - buf._pos
        buf._maybe_compact()
        total_copied += about_to_copy

    assert total_copied <= total_fed, (
        f"copied {total_copied} bytes for {total_fed} fed; "
        "half-fill gate should amortise under burst-fill"
    )


# ---- merged from test_feed_rejection_while_skipping_self_poisons.py ----
# Pin: a feed() size-projection rejection while a skip is in flight
# self-poisons, so the desync'd wire surfaces as PoisonedError rather
# than silently continuing.


def _build_header(declared_words: int) -> bytes:
    size_words_le = declared_words.to_bytes(4, "little")
    return size_words_le + b"\x01" + b"\x00" + b"\x00\x00"


def test_feed_rejection_while_skipping_self_poisons() -> None:
    cap = 64
    buf = ReadBuffer(max_message_size=cap)
    # Feed an oversize header + under-cap body so skip_message arms the
    # deferred-poison path.
    declared_words = 100
    body_bytes_under_cap = WORD_SIZE  # one body word, well under cap
    payload = _build_header(declared_words) + b"\x00" * body_bytes_under_cap
    buf.feed(payload)
    assert buf.skip_message() is False
    assert buf._skip_remaining > 0
    assert not buf.is_poisoned

    # Feed a chunk whose post-skip-discard remainder still exceeds the
    # cap, triggering the projection check; keep it under the early
    # >2*cap gate so that gate doesn't fire first.
    bogus = b"\x00" * (cap + buf._skip_remaining + 1)
    if len(bogus) > 2 * cap:
        bogus = b"\x00" * (2 * cap)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(bogus)

    assert buf.is_poisoned, (
        "feed() must self-poison when its projection check fires while "
        "_skip_remaining > 0; the wire is desync'd and the buffer must "
        "advertise that fact rather than silently continue."
    )


def test_feed_rejection_while_skipping_clears_skip_tracking_fields() -> None:
    """After self-poison, _skip_remaining and _poison_after_skip must be
    cleared so post-poison introspection sees consistent state."""
    cap = 64
    buf = ReadBuffer(max_message_size=cap)
    declared_words = 100
    body_bytes_under_cap = WORD_SIZE
    payload = _build_header(declared_words) + b"\x00" * body_bytes_under_cap
    buf.feed(payload)
    assert buf.skip_message() is False
    assert buf._skip_remaining > 0

    bogus = b"\x00" * (cap + buf._skip_remaining + 1)
    if len(bogus) > 2 * cap:
        bogus = b"\x00" * (2 * cap)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(bogus)

    assert buf.is_poisoned
    assert buf._skip_remaining == 0, (
        "Self-poison branch must clear _skip_remaining; stale value "
        "would mislead any post-poison introspection."
    )
    assert buf._poison_after_skip is None, (
        "Self-poison branch must clear _poison_after_skip; the "
        "deferred-poison path is no longer reachable post-poison."
    )


# ---- merged from test_readbuffer_feed_entry_reject_self_poisons_on_inflight.py ----
# Pin: ``ReadBuffer.feed``'s entry-side oversize reject (chunk >
# 2 * max_message_size) self-poisons when an in-flight body is present,
# so the caller cannot misapply the "safe reset" recovery and desync.


def _make_buffer(cap: int = 64 * 1024) -> ReadBuffer:
    return ReadBuffer(max_message_size=cap)


def test_entry_side_oversize_reject_on_empty_buffer_does_not_poison() -> None:
    """Safe-reset case: rejection with no in-flight body must NOT poison."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    huge = b"\x00" * (2 * cap + 1)
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(huge)
    assert buf.is_poisoned is False, (
        "rejection on an empty buffer must not poison — the documented "
        "safe-reset case still has to hold"
    )
    buf.reset()
    buf.feed(b"\x00" * 16)


def test_entry_side_oversize_reject_with_partial_header_in_buffer_poisons() -> None:
    """In-flight frame present (>= HEADER_SIZE unconsumed): the entry-side
    reject MUST self-poison so safe-reset recovery cannot be misapplied."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    primer = b"\x00" * HEADER_SIZE
    buf.feed(primer)
    assert len(buf._data) - buf._pos >= HEADER_SIZE

    huge = b"\x00" * (2 * cap + 1)
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(huge)

    # A subsequent feed must raise rather than silently accept desynced bytes.
    assert buf.is_poisoned is True
    with pytest.raises(PoisonedError):
        buf.feed(b"\x00" * 16)


def test_entry_side_oversize_reject_with_skip_in_flight_poisons() -> None:
    """Same discipline when ``_skip_remaining > 0`` (skip in flight)."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    buf._skip_remaining = 1024

    huge = b"\x00" * (2 * cap + 1)
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(huge)

    assert buf.is_poisoned is True
    assert buf._skip_remaining == 0


def test_projection_side_reject_with_partial_header_in_buffer_poisons() -> None:
    """Projection-side: a chunk passing the entry-side 2x check but pushing
    projected > cap also self-poisons when an in-flight body is present."""
    cap = 64 * 1024
    buf = _make_buffer(cap)
    primer = b"\x00" * (cap - HEADER_SIZE)
    buf.feed(primer)
    assert len(buf._data) - buf._pos >= HEADER_SIZE

    chunk = b"\x00" * (HEADER_SIZE + 1)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(chunk)

    assert buf.is_poisoned is True


# ---- merged from test_readbuffer_feed_oversize_partial_header_poisons.py ----
# Pin: ``ReadBuffer.feed``'s oversize-reject self-poisons when the
# buffer holds *any* unconsumed pre-header bytes (not just a full 8-byte
# header) — the gate is ``available() > 0``, covering the 1-7-stray-bytes
# case where a peer chunked a header and the second chunk overflowed.


def test_projection_side_oversize_with_partial_header_self_poisons() -> None:
    buf = ReadBuffer(max_message_size=64)
    # Prime with 5 bytes (< HEADER_SIZE=8), then overflow the cap.
    buf.feed(b"\x01\x02\x03\x04\x05")
    assert buf.available() == 5
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(b"\xaa" * 80)
    assert buf.is_poisoned is True


def test_entry_side_oversize_with_partial_header_self_poisons() -> None:
    buf = ReadBuffer(max_message_size=64)
    buf.feed(b"\x01\x02\x03\x04\x05")
    # Exceeds 2x cap: the entry-side reject fires before the projection check.
    with pytest.raises(DecodeError, match="exceeds 2x max_message_size"):
        buf.feed(b"\xaa" * (2 * 64 + 1))
    assert buf.is_poisoned is True


def test_oversize_with_empty_buffer_still_does_not_poison() -> None:
    """Safe-reset case (empty buffer, no skip): DecodeError but no poison."""
    buf = ReadBuffer(max_message_size=64)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        buf.feed(b"\xaa" * 80)
    assert buf.is_poisoned is False
    buf.reset()
    assert buf.is_poisoned is False


# ---- merged from test_skip_message_immediate_poison_returns_false.py ----
# Pin: ``ReadBuffer.skip_message`` returns ``False`` in the
# immediate-poison sub-branch (the entire oversize message body is
# already in the buffer when we call skip_message), even though
# ``_skip_remaining`` reaches zero synchronously.
#
# Pre-fix the method returned ``self._skip_remaining == 0`` which
# evaluated to ``True`` after the synchronous ``self.poison(...)``
# call, contradicting the docstring contract: a caller doing
# ``if buf.skip_message(): continue_decoding()`` would proceed and
# hit ``PoisonedError`` on the next decode with no warning.
#
# The contract is now: ``True`` means "skip succeeded AND the buffer
# remains usable"; ``False`` means "more data needed OR the buffer is
# now poisoned (caller must ``reset()``)".


def _build_skip_header(declared_words: int) -> bytes:
    """Build an 8-byte header declaring ``declared_words`` words of body."""
    size_words_le = declared_words.to_bytes(4, "little")
    type_byte = b"\x01"
    schema_byte = b"\x00"
    extra = b"\x00\x00"
    header = size_words_le + type_byte + schema_byte + extra
    assert len(header) == HEADER_SIZE
    return header


def test_skip_message_immediate_poison_returns_false() -> None:
    """The "immediate-poison" sub-branch fires when the *declared*
    total_size exceeds max_message_size, the body is only partially
    present (bounded by the cap), and ``effective_total == available``.
    In that branch _skip_remaining drops to zero synchronously and
    poison fires inline; pre-fix the method then returned True (the
    raw ``_skip_remaining == 0`` value) which contradicts the
    "next decode can proceed" docstring contract.
    """
    cap = 64
    buf = ReadBuffer(max_message_size=cap)
    # Header declares far more bytes than the cap. We feed exactly
    # ``cap`` bytes (header + body slice that fits under the projection
    # check). At skip_message() time:
    #   total_size = HEADER_SIZE + 100*WORD_SIZE = 808 (>> cap)
    #   effective_total = min(808, 64) = 64
    #   available = 64
    #   skip_now = 64; _skip_remaining = 0; effective_total < total_size
    declared_words = 100
    body_bytes_to_feed = cap - HEADER_SIZE
    payload = _build_skip_header(declared_words) + b"\x00" * body_bytes_to_feed
    assert len(payload) == cap
    buf.feed(payload)

    result = buf.skip_message()
    assert result is False, (
        "skip_message must return False when it synchronously poisons "
        "the buffer; True would falsely advertise that the next decode "
        "can proceed."
    )
    # Confirm the buffer is in fact poisoned now.
    assert buf.is_poisoned
    with pytest.raises(ProtocolError):
        buf.feed(b"\x00" * HEADER_SIZE)


def test_skip_message_returns_false_for_deferred_poison_path() -> None:
    """When the body is only partially in the buffer (deferred poison),
    skip_message also returns False — _skip_remaining > 0 here. Pin
    both branches so the contract is symmetric."""
    cap = 64
    buf = ReadBuffer(max_message_size=cap)
    declared_words = 100
    # Feed header + a few body bytes (less than the cap so deferred path).
    payload = _build_skip_header(declared_words) + b"\x00" * (WORD_SIZE * 2)
    buf.feed(payload)
    assert buf.skip_message() is False
    assert not buf.is_poisoned  # deferred — poison fires when feed completes the skip


# ---- merged from test_oversize_continuation_frame.py ----
# Oversize continuation frames must be rejected before a body read.
#
# ``ReadBuffer`` enforces ``max_message_size`` uniformly on every
# ``feed()`` / ``read_message()`` call regardless of frame ordinal —
# there is no "first vs. continuation" distinction in the buffer
# layer. This module is a structural regression fence: a future
# refactor that split the cap check into a "first-frame only" branch,
# added a per-frame counter with an off-by-one, or weakened the cap
# during continuation mode would silently re-open an amplification
# channel. The tests below pin the contract from the MessageDecoder
# call-site that an in-progress continuation respects the cap.


def _fabricate_oversize_rows_header(body_words: int) -> bytes:
    """Return a valid header that claims a body of ``body_words`` WORDs
    plus enough trailing garbage bytes so the buffer would accept the
    frame if no cap check fired. The body itself is never read — the
    cap check on ``feed()`` triggers first.
    """
    # Header layout: uint32 size_in_words | u8 type | u8 schema | u16 extra.
    header = (
        body_words.to_bytes(4, "little")
        + int(ResponseType.ROWS).to_bytes(1, "little")
        + (0).to_bytes(1, "little")
        + (0).to_bytes(2, "little")
    )
    assert len(header) == HEADER_SIZE
    # A tiny body tail — the buffer rejects before reading this; we
    # only need enough bytes so the buffer sees a complete-ish frame
    # arriving. ``feed()`` raises on size projection, not on content.
    return header + b"\x00" * 8


class TestContinuationFrameOverCap:
    """Cap check must apply to continuation frames identically to
    the initial frame."""

    def test_continuation_header_over_cap_rejected(self) -> None:
        """After a valid first ROWS frame with has_more=True, a
        continuation header declaring a body over the buffer cap must
        raise DecodeError once the decoder attempts to consume it.
        The continuation path does not have ``skip_message()`` recovery,
        so the decoder poisons and forces the caller to reset().
        """
        small_cap = 4096
        decoder = MessageDecoder(max_message_size=small_cap)

        # First frame: valid, has_more=True.
        first = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[1]],
            has_more=True,
        )
        first_bytes = first.encode()
        assert len(first_bytes) < small_cap

        decoder.feed(first_bytes)
        result = cast(RowsResponse, decoder.decode())
        assert result is not None
        assert result.has_more is True

        # Continuation header that claims a body size well over the cap.
        # The header+tail is only a handful of bytes so ``feed()`` accepts
        # it (total projected size is tiny) — the cap check fires when
        # ``decode_continuation()`` attempts to read the declared frame.
        oversize_body_words = (small_cap // WORD_SIZE) + 32
        oversize_frame = _fabricate_oversize_rows_header(oversize_body_words)
        decoder.feed(oversize_frame)

        with pytest.raises(DecodeError, match=r"exceeds maximum"):
            decoder.decode_continuation()

    def test_continuation_within_cap_decodes(self) -> None:
        """A legitimate continuation frame under the cap must decode
        cleanly — boundary fence against an over-eager cap that
        rejects anything in continuation mode.
        """
        small_cap = 64 * 1024
        decoder = MessageDecoder(max_message_size=small_cap)

        first = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[1]],
            has_more=True,
        )
        decoder.feed(first.encode())
        assert decoder.decode() is not None

        # Second frame: small, has_more=False to terminate the stream.
        second = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[2]],
            has_more=False,
        )
        decoder.feed(second.encode())
        cont = decoder.decode_continuation()
        assert cont is not None
        # Narrow the union via isinstance — RowsResponse is the
        # expected continuation shape here; EmptyResponse would mean
        # the second frame was decoded as the wrong type.
        assert isinstance(cont, RowsResponse)
        assert cont.has_more is False
        assert cont.rows == [[2]]


class TestContinuationPoisonedAfterOversize:
    """After an oversize continuation rejection, further decode calls
    must raise PoisonedError — the buffer has no safe recovery path
    during continuation mode."""

    def test_poisoned_after_oversize_continuation(self) -> None:
        small_cap = 4096
        decoder = MessageDecoder(max_message_size=small_cap)

        first = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[1]],
            has_more=True,
        )
        decoder.feed(first.encode())
        decoder.decode()

        oversize_body_words = (small_cap // WORD_SIZE) + 32
        oversize_frame = _fabricate_oversize_rows_header(oversize_body_words)
        decoder.feed(oversize_frame)

        with pytest.raises(DecodeError):
            decoder.decode_continuation()

        # Further operations must see the poisoned state — the
        # continuation path cannot recover in-place via skip_message().
        with pytest.raises(PoisonedError):
            decoder.decode_continuation()


# ---- merged from test_max_message_size_upper_bound.py ----
# Pin: ``max_message_size`` is bounded above by
# ``ReadBuffer.MAX_MESSAGE_SIZE_CEILING`` (``UINT32_MAX`` bytes,
# mirroring the C server's per-frame ceiling at
# ``dqlite-upstream/src/conn.c:169``).
#
# A caller that constructs ``ReadBuffer(max_message_size=2**40)`` (or
# ``MessageEncoder`` / ``MessageDecoder``) would otherwise silently
# disable the composite-frame protection the envelope is supposed to
# provide. The C server hard-caps inbound frames at ``UINT32_MAX``
# bytes; a Python ceiling at the same bound can never refuse a frame
# the C server would accept.


def test_read_buffer_rejects_oversize_max_message_size() -> None:
    with pytest.raises(ValueError, match="UINT32_MAX"):
        ReadBuffer(max_message_size=2**40)


def test_message_encoder_rejects_oversize_max_message_size() -> None:
    with pytest.raises(ValueError, match="UINT32_MAX"):
        MessageEncoder(max_message_size=2**40)


def test_message_decoder_rejects_oversize_max_message_size() -> None:
    with pytest.raises(ValueError, match="UINT32_MAX"):
        MessageDecoder(max_message_size=2**40)


def test_ceiling_itself_is_admissible() -> None:
    """At ``UINT32_MAX`` exactly the constructor still works — the
    upper-bound check is non-strict (``>``, not ``>=``)."""
    buf = ReadBuffer(max_message_size=ReadBuffer.MAX_MESSAGE_SIZE_CEILING)
    assert buf._max_message_size == ReadBuffer.MAX_MESSAGE_SIZE_CEILING


def test_default_and_smaller_values_still_accepted() -> None:
    ReadBuffer(max_message_size=1)
    ReadBuffer(max_message_size=ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE)
    MessageEncoder(max_message_size=ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE)
    MessageDecoder(max_message_size=ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE)
