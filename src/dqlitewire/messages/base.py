"""Base message types for dqlite wire protocol."""

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from dqlitewire.constants import HEADER_SIZE, WORD_SIZE
from dqlitewire.exceptions import DecodeError


@dataclass
class Header:
    """Message header.

    Format (8 bytes):
    - size: uint32 - Size of message body in words (8-byte units)
    - type: uint8 - Message type code
    - schema: uint8 - Schema version (currently always 0)
    - reserved: uint16 - Reserved (always 0)
    """

    size_words: int
    msg_type: int
    schema: int = 0
    reserved: int = 0

    def encode(self) -> bytes:
        """Encode header to bytes."""
        return struct.pack(
            "<IBBH",
            self.size_words,
            self.msg_type,
            self.schema,
            self.reserved,
        )

    @classmethod
    def decode(cls, data: bytes) -> "Header":
        """Decode header from bytes."""
        if len(data) < HEADER_SIZE:
            raise DecodeError(f"Need {HEADER_SIZE} bytes for header, got {len(data)}")
        size_words, msg_type, schema, reserved = struct.unpack("<IBBH", data[:HEADER_SIZE])
        return cls(size_words, msg_type, schema, reserved)

    @property
    def body_size(self) -> int:
        """Size of message body in bytes."""
        return self.size_words * WORD_SIZE


class Message(ABC):
    """Base class for all protocol messages."""

    MSG_TYPE: ClassVar[int]
    SCHEMA: ClassVar[int] = 0

    @abstractmethod
    def encode_body(self) -> bytes:
        """Encode message body (without header)."""
        ...

    def _get_schema(self) -> int:
        """Return the schema version for the header. Override for per-instance schema."""
        return self.SCHEMA

    def encode(self) -> bytes:
        """Encode complete message with header."""
        body = self.encode_body()
        # Ensure body is word-aligned
        if len(body) % WORD_SIZE != 0:
            body += b"\x00" * (WORD_SIZE - (len(body) % WORD_SIZE))
        size_words = len(body) // WORD_SIZE
        header = Header(size_words, self.MSG_TYPE, schema=self._get_schema())
        return header.encode() + body

    @classmethod
    @abstractmethod
    def decode_body(cls, data: bytes) -> "Message":
        """Decode message from body data (without header)."""
        ...
