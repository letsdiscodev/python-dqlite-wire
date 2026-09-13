"""Base message types for dqlite wire protocol."""

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, final

from dqlitewire.constants import HEADER_SIZE, WORD_SIZE
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.types import _is_int_not_bool

__all__ = [
    "Header",
    "Message",
    "require_consumed",
    "require_length",
    "require_schema",
]


@final
@dataclass(frozen=True, slots=True)
class Header:
    """Message header (8 bytes, ``<IBBH``): size_words, msg_type, schema, reserved.

    Construction accepts any uint8 schema; the per-message-type ceiling is
    enforced at decode-dispatch (in ``codec.py``) to avoid a circular import.
    ``reserved`` is written as 0 and ignored on decode, as go-dqlite does.
    """

    size_words: int
    msg_type: int
    schema: int = 0
    reserved: int = 0

    def __post_init__(self) -> None:
        if not _is_int_not_bool(self.reserved) or not 0 <= self.reserved < 2**16:
            raise EncodeError(f"Header reserved field {self.reserved!r} out of range for uint16")
        # Range-validate at construction for precise errors instead of an
        # opaque ``struct.error`` at encode. ``bool`` is rejected first
        # because ``True == 1`` would coerce to a valid uint8.
        if not _is_int_not_bool(self.size_words):
            raise EncodeError(
                f"Header size_words must be int, got {type(self.size_words).__name__}"
            )
        if not 0 <= self.size_words < 2**32:
            raise EncodeError(f"Header size_words {self.size_words} out of range for uint32")
        if not _is_int_not_bool(self.msg_type):
            raise EncodeError(f"Header msg_type must be int, got {type(self.msg_type).__name__}")
        if not 0 <= self.msg_type < 2**8:
            raise EncodeError(f"Header msg_type {self.msg_type} out of range for uint8")
        if not _is_int_not_bool(self.schema):
            raise EncodeError(f"Header schema must be int, got {type(self.schema).__name__}")
        if not 0 <= self.schema < 2**8:
            raise EncodeError(f"Header schema {self.schema} out of range for uint8")

    def encode(self) -> bytes:
        try:
            return struct.pack(
                "<IBBH",
                self.size_words,
                self.msg_type,
                self.schema,
                self.reserved,
            )
        except struct.error as e:
            raise EncodeError(f"Failed to encode header: {e}") from e

    @classmethod
    def decode(cls, data: bytes) -> "Header":
        if len(data) < HEADER_SIZE:
            raise DecodeError(f"Need {HEADER_SIZE} bytes for header, got {len(data)}")
        size_words, msg_type, schema, reserved = struct.unpack("<IBBH", data[:HEADER_SIZE])
        return cls(size_words, msg_type, schema, reserved)

    @property
    def body_size(self) -> int:
        """Message body size in bytes (canonical words-to-bytes accessor)."""
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
        """Schema version for the header; override for per-instance schema."""
        return self.SCHEMA

    def encode(self) -> bytes:
        """Encode complete message with header."""
        body = self.encode_body()
        # Subclass encoders must self-pad to word alignment; fail loudly
        # here rather than silently padding and masking a subclass bug.
        if len(body) % WORD_SIZE != 0:
            raise EncodeError(
                f"{type(self).__name__}.encode_body() returned "
                f"{len(body)} bytes; must be {WORD_SIZE}-aligned"
            )
        size_words = len(body) // WORD_SIZE
        header = Header(size_words, self.MSG_TYPE, schema=self._get_schema())
        return header.encode() + body

    @classmethod
    @abstractmethod
    def decode_body(cls, data: bytes, schema: int = 0) -> "Message":
        """Decode message from body data (without header)."""
        ...


def require_schema(cls: type, schema: int, allowed: tuple[int, ...] = (0,)) -> None:
    if schema not in allowed:
        raise DecodeError(f"{cls.__name__} unsupported schema version {schema}")


def require_length(cls: type, data: bytes, size: int, *, exactly: bool = False) -> None:
    if len(data) != size:
        qualifier = "exactly " if exactly else ""
        raise DecodeError(f"{cls.__name__} body must be {qualifier}{size} bytes, got {len(data)}")


def require_consumed(cls: type, data: bytes, offset: int, after: str = "") -> None:
    if offset != len(data):
        suffix = f" after {after}" if after else ""
        raise DecodeError(f"{cls.__name__} has {len(data) - offset} trailing bytes{suffix}")
