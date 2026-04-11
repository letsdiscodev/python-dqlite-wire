"""Tests for buffer utilities."""

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

    def test_skip_message_recovers_from_oversized(self) -> None:
        """After skipping an oversized message, stream should not be corrupted.

        skip_message caps _skip_remaining to max_message_size to prevent
        amplification attacks. The test feeds at most max_message_size bytes
        of oversized body, then verifies the next message decodes correctly.
        """
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        # Header claiming a huge body: size_words=1000 (8000 bytes > 1024 limit)
        oversized_header = struct.pack("<IBBH", 1000, 0, 0, 0)
        valid_msg = LeaderResponse(node_id=1, address="node1:9001")
        valid_encoded = valid_msg.encode()

        # Feed header + first chunk
        buf.feed(oversized_header + b"\xab" * 500)

        # has_message() is total; the raise happens on the consume call.
        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()

        # skip_message should return False — partial skip
        assert buf.skip_message() is False

        # Feed enough bytes to complete the capped skip + valid message.
        # _skip_remaining is capped to max_message_size (1024), and skip_message
        # already consumed the 508 bytes in the buffer (8 header + 500 body),
        # so _skip_remaining = 1024 - 508 = 516 bytes.
        buf.feed(b"\xab" * 516 + valid_encoded)

        # The capped oversized bytes should be discarded; valid message readable
        assert buf.has_message()
        data = buf.read_message()
        assert data is not None
        assert data == valid_encoded

    def test_skip_oversized_across_multiple_feeds(self) -> None:
        """Oversized message bytes should be discarded across multiple feed() calls.

        skip_message caps _skip_remaining to max_message_size, so only that
        many bytes are discarded (not the full claimed body size).
        """
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=64)
        # 200 words = 1600 bytes body, well over 64-byte limit
        oversized_header = struct.pack("<IBBH", 200, 0, 0, 0)
        valid_msg = LeaderRequest()
        valid_encoded = valid_msg.encode()

        # Feed just the header
        buf.feed(oversized_header)
        assert buf.has_message() is True
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.read_message()
        assert buf.skip_message() is False

        # Feed capped body in chunks (max_message_size=64, header was 8 bytes,
        # so _skip_remaining = 64 - 8 = 56 bytes)
        remaining = buf._skip_remaining
        body = b"\xcc" * remaining
        while body:
            chunk = body[:20]
            body = body[20:]
            buf.feed(chunk)

        # Now feed the valid message
        buf.feed(valid_encoded)
        assert buf.has_message()
        data = buf.read_message()
        assert data == valid_encoded

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
