"""``WriteBuffer.write``/``write_padded`` and ``decode_message`` accept any
bytes-like input (bytes, bytearray, memoryview)."""

from dqlitewire.buffer import WriteBuffer
from dqlitewire.codec import decode_message
from dqlitewire.messages.responses import EmptyResponse


def test_write_accepts_bytearray() -> None:
    buf = WriteBuffer()
    buf.write(bytearray(b"hello"))
    assert buf.getvalue() == b"hello"


def test_write_accepts_memoryview() -> None:
    buf = WriteBuffer()
    buf.write(memoryview(b"hello"))
    assert buf.getvalue() == b"hello"


def test_write_padded_accepts_bytearray() -> None:
    buf = WriteBuffer()
    buf.write_padded(bytearray(b"hi"))
    assert buf.getvalue() == b"hi" + b"\x00" * 6


def test_write_padded_accepts_memoryview() -> None:
    buf = WriteBuffer()
    buf.write_padded(memoryview(b"hi"))
    assert buf.getvalue() == b"hi" + b"\x00" * 6


def test_decode_message_accepts_memoryview() -> None:
    msg_bytes = EmptyResponse().encode()
    msg = decode_message(memoryview(msg_bytes))
    assert isinstance(msg, EmptyResponse)


def test_decode_message_accepts_bytearray() -> None:
    msg_bytes = bytearray(EmptyResponse().encode())
    msg = decode_message(msg_bytes)
    assert isinstance(msg, EmptyResponse)
