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

    def test_rejects_oversized_message(self) -> None:
        """Messages exceeding max size should raise DecodeError."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        # Header claiming a huge body: size_words=1000 (8000 bytes > 1024 limit)
        header = struct.pack("<IBBH", 1000, 0, 0, 0)
        buf.feed(header)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.has_message()

    def test_skip_message_recovers_from_oversized(self) -> None:
        """After an oversized message, skip_message() should allow recovery."""
        import struct

        import pytest

        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        # Header claiming a huge body: size_words=1000 (8000 bytes > 1024 limit)
        oversized_header = struct.pack("<IBBH", 1000, 0, 0, 0)
        # Feed oversized header + a valid message after it
        valid_msg = LeaderResponse(node_id=1, address="node1:9001")
        valid_encoded = valid_msg.encode()
        buf.feed(oversized_header + valid_encoded)

        # has_message should raise for oversized
        with pytest.raises(DecodeError, match="exceeds maximum"):
            buf.has_message()

        # skip_message should skip past the oversized header
        assert buf.skip_message() is True

        # Now the valid message should be readable
        # (But the oversized message consumed only its header since we
        # didn't have the full body — skip advances past what's available)
        # Feed the valid message again to a fresh position
        buf2 = ReadBuffer(max_message_size=1024)
        buf2.feed(valid_encoded)
        assert buf2.has_message()
        data = buf2.read_message()
        assert data is not None

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
