"""``WriteBuffer.write`` and ``write_padded`` accept any bytes-like
input (``bytes``, ``bytearray``, ``memoryview``).

The runtime always supported this via ``bytearray.extend``, but the
type annotation declared ``data: bytes`` which lied to type-checker
users. Sibling decoders (``decode_text``, ``decode_blob``) consistently
typed as ``bytes | memoryview``; align here for consistency.

``MessageDecoder.decode_bytes`` and the module-level ``decode_message``
helper similarly widened.
"""

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
    assert buf.getvalue() == b"hi" + b"\x00" * 6  # padded to word boundary


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
