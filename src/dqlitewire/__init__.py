"""Pure Python wire protocol implementation for dqlite."""

from dqlitewire.buffer import ReadBuffer, WriteBuffer
from dqlitewire.codec import MessageDecoder, MessageEncoder
from dqlitewire.constants import (
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_LEGACY,
    ROW_DONE_BYTE,
    ROW_DONE_MARKER,
    ROW_PART_BYTE,
    ROW_PART_MARKER,
    RequestType,
    ResponseType,
    ValueType,
)
from dqlitewire.exceptions import DecodeError, EncodeError, ProtocolError

__all__ = [
    "MessageDecoder",
    "MessageEncoder",
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_LEGACY",
    "ProtocolError",
    "DecodeError",
    "EncodeError",
    "ReadBuffer",
    "ROW_DONE_BYTE",
    "ROW_DONE_MARKER",
    "ROW_PART_BYTE",
    "ROW_PART_MARKER",
    "WriteBuffer",
    "RequestType",
    "ResponseType",
    "ValueType",
]

__version__ = "0.1.0"
