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
