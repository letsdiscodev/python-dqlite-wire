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
        buf.write_padded(b"hi")  # 2 bytes, padded to 8
        assert len(buf) == 8
        assert buf.getvalue() == b"hi\x00\x00\x00\x00\x00\x00"

    def test_write_padded_exact(self) -> None:
        buf = WriteBuffer()
        buf.write_padded(b"12345678")  # 8 bytes, no padding needed
        assert len(buf) == 8
        assert buf.getvalue() == b"12345678"

    def test_clear(self) -> None:
        buf = WriteBuffer()
        buf.write(b"data")
        buf.clear()
        assert len(buf) == 0
        assert buf.getvalue() == b""

    def test_write_padded_no_interleaved_tearing_under_contention(self) -> None:
        """Regression for the torn write_padded interleave.

        ``write_padded`` used to do two separate ``bytearray.extend`` calls
        (one for the payload, one for the NUL padding). Under concurrent
        access a thread switch between the two calls could land another
        thread's payload bytes inside the first thread's padding window,
        producing a torn word like ``b'BBBBBAAA'`` instead of a clean
        ``b'AAAAA\\x00\\x00\\x00'`` or ``b'BBBBB\\x00\\x00\\x00'``.

        The fix is to build the padded bytes locally and emit a single
        extend, so no thread can interleave between the payload and its
        padding. This test drives the race at a tight switch interval and
        asserts every output word is one of the two expected shapes.
        """
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
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(buf)


class TestReadBuffer:
    def test_empty(self) -> None:
        buf = ReadBuffer()
        assert buf.available() == 0
        assert not buf.has_message()

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
        # Buffer state must be unchanged
        assert buf.available() == 11

    def test_peek_bytes_negative_n_raises(self) -> None:
        """peek_bytes with negative n must raise, not return empty bytes."""
        import pytest

        buf = ReadBuffer()
        buf.feed(b"hello world")
        with pytest.raises(ValueError, match="non-negative"):
            buf.peek_bytes(-1)
        assert buf.available() == 11

    def test_has_message_incomplete(self) -> None:
        buf = ReadBuffer()
        # Feed only header
        buf.feed(b"\x01\x00\x00\x00\x00\x00\x00\x00")
        assert not buf.has_message()  # Need 1 word of body

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
        buf.feed(b"\x01\x00\x00\x00\x00\x00\x00\x00")  # Header says 1 word
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
        """Regression: clear() must also un-poison.

        ``ReadBuffer`` has two very similar methods: ``clear()`` and
        ``reset()``. ``reset()`` was introduced alongside the poison
        flag as its recovery primitive. ``clear()`` predates the
        poison concept and was never updated — it still resets
        ``_data``/``_pos``/``_skip_remaining`` but leaves
        ``_poisoned`` intact. A caller who reaches for ``clear()``
        expecting it to be "the simpler recovery method" gets a
        half-fresh buffer that still raises ``ProtocolError`` on the
        next operation. The asymmetry is invisible at the API level
        (one-line docstrings, similar names), so the only safe
        behaviour is for both methods to unpoison.
        """
        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer()
        buf.feed(b"\x00" * 16)
        buf.poison(DecodeError("boom"))
        assert buf.is_poisoned

        buf.clear()

        assert not buf.is_poisoned
        # And the buffer must be usable after the clear+reconnect.
        buf.feed(b"\x01\x00\x00\x00\x00\x00\x00\x00")
        assert buf.available() == 8

    def test_public_api_honors_poison(self) -> None:
        """Regression: poison flag must be honored by the public API.

        The poison flag was introduced so that once a decode
        error has desynchronized the stream, subsequent operations
        fail fast with ``ProtocolError`` instead of silently reading
        from an unknown offset. The check was wired into
        ``MessageDecoder.decode`` / ``decode_continuation`` but NOT
        into ``ReadBuffer``'s own public API. A caller holding a raw
        ``ReadBuffer`` (exported from ``dqlitewire`` since
        ``__init__.py``) could keep calling ``feed()``, ``read_message()``,
        etc., after a poison was set and re-desynchronize the stream
        silently.

        Every mutating/consuming public method on ``ReadBuffer`` must
        raise ``ProtocolError`` when poisoned. Pure observers
        (``has_message``, ``available``, ``is_poisoned``, ``is_skipping``)
        and recovery primitives (``clear``, ``reset``, ``poison``)
        must remain callable.
        """
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

        # Observers must remain callable on a poisoned buffer.
        _ = buf.available()
        _ = buf.has_message()
        _ = buf.is_poisoned
        _ = buf.is_skipping

        # Recovery path must still work: reset() unpoisons.
        buf.reset()
        assert not buf.is_poisoned
        buf.feed(b"\x00" * 8)

    def test_has_message_is_total_on_oversized(self) -> None:
        """has_message() must not raise — oversized headers surface at consume time.

        Callers using the documented `while decoder.has_message(): decode()` pattern
        must not have to wrap the check itself in try/except. The raise belongs at
        read_message() / skip_message() / decode(), not at the predicate.
        """
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        # Header claiming a huge body: size_words=1000 (8000 bytes > 1024 limit)
        header = struct.pack("<IBBH", 1000, 0, 0, 0)
        buf.feed(header)

        # The predicate is total: it returns True to signal "something is there
        # to consume" and never raises.
        assert buf.has_message() is True

        # The raise happens at consume time.
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()

    def test_peek_header_validates_size(self) -> None:
        """peek_header() must reject oversized headers, matching has_message().

        peek_header() is a sibling inspection API; returning a raw unvalidated
        size_words lets callers accidentally trust an attacker-controlled value
        (e.g. preallocate based on it). Fail loudly and consistently.
        """
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        # size_words=1000 -> 8000-byte body > 1024 limit
        header = struct.pack("<IBBH", 1000, 0, 0, 0)
        buf.feed(header)

        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.peek_header()

    def test_peek_header_on_valid_header(self) -> None:
        """peek_header() returns the parsed tuple for a valid header and
        does not advance _pos.
        """
        import struct

        buf = ReadBuffer(max_message_size=1024)
        # size_words=1 -> 8-byte body, type=5, schema=0, within limit
        header = struct.pack("<IBBH", 1, 5, 0, 0)
        buf.feed(header)
        result = buf.peek_header()
        assert result == (1, 5, 0)
        # Non-consuming — a second peek returns the same tuple.
        assert buf.peek_header() == (1, 5, 0)

    def test_peek_header_partial_returns_none(self) -> None:
        """With fewer than HEADER_SIZE bytes buffered, peek_header() returns None."""
        buf = ReadBuffer()
        buf.feed(b"\x00\x00")  # only 2 bytes
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
        """A header whose total_size equals max_message_size must be accepted;
        one byte over must raise. Locks in the off-by-one boundary.
        """
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        # max_message_size = 16 means HEADER_SIZE (8) + 1 word (8) is the
        # exact boundary. size_words=1 -> total_size=16 -> ok.
        buf_ok = ReadBuffer(max_message_size=16)
        buf_ok.feed(struct.pack("<IBBH", 1, 0, 0, 0))
        assert buf_ok.peek_header() == (1, 0, 0)

        # size_words=2 -> total_size=24 -> over the limit by 8.
        buf_over = ReadBuffer(max_message_size=16)
        buf_over.feed(struct.pack("<IBBH", 2, 0, 0, 0))
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf_over.peek_header()

    def test_peek_header_and_has_message_disagree_on_oversized(self) -> None:
        """Pinning test for the deliberate asymmetry between the two
        inspection methods on the same buffer.

        has_message() is total (returns True for oversized so the consume
        call surfaces the error). peek_header() raises immediately because
        its return value is an attacker-controlled integer that callers
        might use for allocation. This test locks in the contract so a
        future refactor that "fixes the inconsistency" by making one
        match the other has to confront the trade-off explicitly.
        """
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        buf.feed(struct.pack("<IBBH", 1000, 0, 0, 0))  # 8000 > 1024

        # Predicate: total, returns True.
        assert buf.has_message() is True

        # Inspection-with-payload: raises.
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.peek_header()

    def test_rejects_oversized_message(self) -> None:
        """Oversized messages surface at read_message(), not has_message()."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        # Header claiming a huge body: size_words=1000 (8000 bytes > 1024 limit)
        header = struct.pack("<IBBH", 1000, 0, 0, 0)
        buf.feed(header)
        # has_message() is non-raising; consume-side surfaces the error.
        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()

    def test_skip_message_poisons_after_capped_oversized(self) -> None:
        """After a capped oversized skip, stream is desynchronized and
        the buffer must be poisoned.

        skip_message caps _skip_remaining to max_message_size to prevent
        amplification attacks. Since the peer sent more bytes than the cap,
        the stream is permanently desynchronized. The buffer is poisoned
        once the capped skip completes.
        """
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError, ProtocolError

        buf = ReadBuffer(max_message_size=1024)
        # Header claiming a huge body: size_words=1000 (8000 bytes > 1024 limit)
        oversized_header = struct.pack("<IBBH", 1000, 0, 0, 0)

        # Feed header + first chunk
        buf.feed(oversized_header + b"\xab" * 500)

        # has_message() is total; the raise happens on the consume call.
        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()

        # skip_message should return False — partial skip
        assert buf.skip_message() is False

        # Feed enough bytes to complete the capped skip.
        remaining = buf._skip_remaining
        buf.feed(b"\xab" * remaining)

        # Skip is complete but buffer is poisoned — stream is desynchronized
        assert not buf.is_skipping
        assert buf.is_poisoned

        with pytest.raises(ProtocolError, match="poisoned"):
            buf.read_message()

    def test_capped_oversized_skip_poisons_buffer(self) -> None:
        """121: capped skip of oversized message must poison the buffer.

        When the header claims a body larger than max_message_size, the skip
        is capped to max_message_size bytes. Since the peer sent more bytes
        than were discarded, the stream is permanently desynchronized. The
        buffer must be poisoned to prevent silent garbage reads.
        """
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError, ProtocolError

        buf = ReadBuffer(max_message_size=64)
        # Header claiming 200 words = 1600 bytes body (>> 64 byte limit)
        oversized_header = struct.pack("<IBBH", 200, 0, 0, 0)
        buf.feed(oversized_header)

        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()

        # skip_message triggers capped skip
        buf.skip_message()

        # After the capped skip completes, buffer must be poisoned
        remaining = buf._skip_remaining
        buf.feed(b"\x00" * remaining)
        assert not buf.is_skipping
        assert buf.is_poisoned

        # Further operations should raise ProtocolError
        with pytest.raises(ProtocolError, match="poisoned"):
            buf.read_message()

    def test_single_feed_completes_capped_skip_and_poisons(self) -> None:
        """118/121: single feed() that completes capped skip poisons buffer."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError, ProtocolError

        buf = ReadBuffer(max_message_size=64)
        # 200 words = 1600 bytes body, well over 64-byte limit
        oversized_header = struct.pack("<IBBH", 200, 0, 0, 0)

        buf.feed(oversized_header)
        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()
        assert buf.skip_message() is False
        assert buf.is_skipping

        # Build a single payload: remaining skip bytes + extra data
        remaining = buf._skip_remaining
        combined = b"\xcc" * remaining + b"\x00" * 16

        buf.feed(combined)

        assert not buf.is_skipping
        assert buf.is_poisoned

        with pytest.raises(ProtocolError, match="poisoned"):
            buf.read_message()

    def test_skip_oversized_across_multiple_feeds_poisons(self) -> None:
        """Oversized skip across multiple feeds poisons when complete.

        skip_message caps _skip_remaining to max_message_size, so only that
        many bytes are discarded. Once the capped skip completes, the buffer
        is poisoned because the stream is desynchronized.
        """
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError, ProtocolError

        buf = ReadBuffer(max_message_size=64)
        # 200 words = 1600 bytes body, well over 64-byte limit
        oversized_header = struct.pack("<IBBH", 200, 0, 0, 0)

        # Feed just the header
        buf.feed(oversized_header)
        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()
        assert buf.skip_message() is False

        # Feed capped body in chunks
        remaining = buf._skip_remaining
        body = b"\xcc" * remaining
        while body:
            chunk = body[:20]
            body = body[20:]
            buf.feed(chunk)

        # Capped skip is complete — buffer should be poisoned
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

        # Feed an oversized header (200 words = 1600 bytes > 64 limit)
        header = struct.pack("<IBBH", 200, 0, 0, 0)
        buf.feed(header)
        assert buf.has_message() is True
        with pytest.raises(DecodeError):
            buf.read_message()

        # Partial skip — is_skipping should be True
        assert buf.skip_message() is False
        assert buf.is_skipping is True

        # Feed enough bytes to complete the capped skip
        # (max_message_size=64, header was 8, so _skip_remaining = 64 - 8 = 56)
        buf.feed(b"\x00" * buf._skip_remaining)
        assert buf.is_skipping is False

    def test_clear_resets_skip_state(self) -> None:
        """clear() should cancel any in-progress oversized skip."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=64)
        header = struct.pack("<IBBH", 200, 0, 0, 0)
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
        # First feed is fine
        buf.feed(b"\x00" * 512)
        # Second feed would exceed max
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.feed(b"\x00" * 600)

    def test_read_bytes_triggers_compaction(self) -> None:
        """read_bytes should compact the buffer after consuming enough data."""
        buf = ReadBuffer()
        # Feed more than 4096 bytes worth of data
        buf.feed(b"\x00" * 5000)
        buf.read_bytes(4500)
        # After compaction, internal _pos should be reset
        assert buf._pos == 0
        assert len(buf._data) == 500

    def test_read_message_validates_size_independently(self) -> None:
        """read_message should validate message size even if has_message was not the last call."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        # Craft a message header claiming a huge body (200 words = 1600 bytes > 1024 limit)
        size_words = 200
        header = struct.pack("<IBBH", size_words, 0, 0, 0)
        # Feed header + enough body data to make it "complete"
        body = b"\x00" * (size_words * 8)
        buf._data = bytearray(header + body)
        buf._pos = 0
        # Bypass has_message and call read_message directly — it calls has_message internally
        # which should raise
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()

    def test_skip_message_allows_oversized_for_recovery(self) -> None:
        """skip_message should NOT reject oversized messages — it's the recovery mechanism."""
        import struct

        buf = ReadBuffer(max_message_size=1024)
        # Craft a message header claiming a huge body (200 words = 1600 bytes > 1024 limit)
        size_words = 200
        header = struct.pack("<IBBH", size_words, 0, 0, 0)
        body = b"\x00" * (size_words * 8)
        buf._data = bytearray(header + body)
        buf._pos = 0
        # skip_message should succeed (it's the recovery tool for oversized messages)
        assert buf.skip_message() is True

    def test_skip_message_waits_for_complete_normal_sized_message(self) -> None:
        """skip_message should return False for incomplete normal-sized messages.

        If a message fits within max_message_size but hasn't fully arrived,
        skip_message must not advance past partial data — doing so would
        corrupt the stream when the remaining bytes arrive later.
        """
        import struct

        buf = ReadBuffer(max_message_size=4096)
        # Header claiming 5 words (40 bytes body), total 48 bytes — fits in limit
        header = struct.pack("<IBBH", 5, 0, 0, 0)
        # Feed only header + 16 bytes of body (incomplete: need 40)
        buf.feed(header + b"\x00" * 16)
        # Should return False since message is incomplete but not oversized
        assert buf.skip_message() is False
        # Buffer position should not have changed
        assert buf.available() == 24  # 8 header + 16 partial body

    def test_skip_message_caps_remaining_to_max_message_size(self) -> None:
        """skip_message should not set _skip_remaining beyond max_message_size.

        A malicious header with size_words=0xFFFFFFFF claims a ~32 GiB body.
        Without capping, _skip_remaining would be ~32 GiB, causing all
        subsequent feed() calls to silently discard data for an extremely
        long time (8-byte header → 32 GiB data loss amplification).
        """
        import struct

        buf = ReadBuffer(max_message_size=1024)
        # Craft header claiming ~32 GiB body
        header = struct.pack("<IBBH", 0xFFFFFFFF, 0, 0, 0)
        buf.feed(header)

        buf.skip_message()
        # _skip_remaining should be capped to max_message_size, not ~32 GiB
        assert buf._skip_remaining <= buf._max_message_size

    def test_feed_compacts_consumed_data(self) -> None:
        """feed() should compact consumed data before extending the buffer.

        When _pos is past the compaction threshold, feed() should compact
        the buffer to reclaim consumed bytes before adding new data.
        """
        buf = ReadBuffer(max_message_size=65536)
        # Simulate a state where data was consumed but not yet compacted
        # (e.g., by external manipulation or edge case in read patterns)
        buf._data = bytearray(b"\x00" * 5000)
        buf._pos = 4500  # 4500 consumed, 500 remaining
        # feed() should compact (since _pos > 4096) before extending
        buf.feed(b"\x01" * 100)
        assert buf._pos == 0
        assert len(buf._data) == 600  # 500 remaining + 100 new

    def test_buffer_compaction(self) -> None:
        buf = ReadBuffer()
        # Feed a lot of small messages to trigger compaction
        msg = LeaderRequest()
        encoded = msg.encode()

        for _ in range(1000):
            buf.feed(encoded)
            buf.read_message()

        # Buffer should still work
        buf.feed(encoded)
        assert buf.has_message()

    def test_pickle_raises_typeerror(self) -> None:
        """ReadBuffer must not be picklable."""
        buf = ReadBuffer()
        buf.feed(b"\x00" * 16)
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(buf)
