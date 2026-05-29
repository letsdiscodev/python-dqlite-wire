"""Tests for buffer utilities."""

import pickle

import pytest

from dqlitewire.buffer import ReadBuffer, WriteBuffer
from dqlitewire.messages import LeaderRequest, LeaderResponse


class TestWriteBuffer:
    def test_empty(self) -> None:
        buf = WriteBuffer()
        assert len(buf) == 0
        assert buf.getvalue() == b""

    def test_write(self) -> None:
        buf = WriteBuffer()
        buf.write(b"hello")
        assert len(buf) == 5
        assert buf.getvalue() == b"hello"

    def test_multiple_writes(self) -> None:
        buf = WriteBuffer()
        buf.write(b"hel")
        buf.write(b"lo")
        assert buf.getvalue() == b"hello"

    def test_write_padded(self) -> None:
        buf = WriteBuffer()
        buf.write_padded(b"hi")
        assert len(buf) == 8
        assert buf.getvalue() == b"hi\x00\x00\x00\x00\x00\x00"

    def test_write_padded_exact(self) -> None:
        buf = WriteBuffer()
        buf.write_padded(b"12345678")
        assert len(buf) == 8
        assert buf.getvalue() == b"12345678"

    def test_clear(self) -> None:
        buf = WriteBuffer()
        buf.write(b"data")
        buf.clear()
        assert len(buf) == 0
        assert buf.getvalue() == b""

    def test_write_padded_no_interleaved_tearing_under_contention(self) -> None:
        """Regression: write_padded must emit payload+padding in one extend so a
        thread switch can't interleave another payload into the padding window."""
        import sys
        import threading

        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.0000001)
        try:
            buf = WriteBuffer()

            def worker(payload: bytes) -> None:
                for _ in range(5000):
                    buf.write_padded(payload)

            t1 = threading.Thread(target=worker, args=(b"A" * 5,))
            t2 = threading.Thread(target=worker, args=(b"B" * 5,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            out = buf.getvalue()
            assert len(out) == 2 * 5000 * 8
            for i in range(0, len(out), 8):
                word = out[i : i + 8]
                assert word in (
                    b"AAAAA\x00\x00\x00",
                    b"BBBBB\x00\x00\x00",
                ), f"torn word at offset {i}: {bytes(word)!r}"
        finally:
            sys.setswitchinterval(old_interval)

    def test_pickle_raises_typeerror(self) -> None:
        """WriteBuffer must not be picklable."""
        buf = WriteBuffer()
        buf.write(b"data")
        with pytest.raises(TypeError, match="cannot pickle"):
            pickle.dumps(buf)

    def test_pickle_message_does_not_claim_connection_stream_binding(self) -> None:
        """Rejection message must not invent a "connection stream" binding the
        class does not hold (it is just a bytearray)."""
        buf = WriteBuffer()
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
        assert "WriteBuffer" in msg


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
