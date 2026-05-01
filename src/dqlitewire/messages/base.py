"""Base message types for dqlite wire protocol."""

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from dqlitewire.constants import HEADER_SIZE, WORD_SIZE
from dqlitewire.exceptions import DecodeError, EncodeError


@dataclass(frozen=True, slots=True)
class Header:
    """Message header.

    Format (8 bytes):
    - size: uint32 - Size of message body in words (8-byte units)
    - type: uint8 - Message type code
    - schema: uint8 - Schema version (0 or 1; V1 extends param tuples and StmtResponse)
    - reserved: uint16 - Reserved (always 0)
    """

    size_words: int
    msg_type: int
    schema: int = 0
    reserved: int = 0

    def __post_init__(self) -> None:
        # Enforce ``reserved == 0`` symmetrically on encode and decode.
        # The decoder rejects non-zero ``reserved`` so peer corruption
        # surfaces as DecodeError; without this construction-time
        # check, ``Header(reserved=42).encode()`` would produce wire
        # bytes the same decoder rejects, breaking encode→decode
        # round-trip identity. Upstream C (message.h) keeps the field
        # zero in every emit path.
        if self.reserved != 0:
            raise EncodeError(f"Header reserved field must be 0, got {self.reserved}")
        # Range-validate the remaining three fields at construction
        # time so an invalid value surfaces here (with a precise
        # field name + observed value) rather than at ``encode()``
        # time as an opaque ``struct.error → EncodeError`` wrap.
        # ``bool`` is rejected first because ``True == 1`` would
        # silently coerce to a valid uint8 and mask caller bugs.
        if isinstance(self.size_words, bool) or not isinstance(self.size_words, int):
            raise EncodeError(
                f"Header size_words must be int, got {type(self.size_words).__name__}"
            )
        if not 0 <= self.size_words < 2**32:
            raise EncodeError(f"Header size_words {self.size_words} out of range for uint32")
        if isinstance(self.msg_type, bool) or not isinstance(self.msg_type, int):
            raise EncodeError(f"Header msg_type must be int, got {type(self.msg_type).__name__}")
        if not 0 <= self.msg_type < 2**8:
            raise EncodeError(f"Header msg_type {self.msg_type} out of range for uint8")
        if isinstance(self.schema, bool) or not isinstance(self.schema, int):
            raise EncodeError(f"Header schema must be int, got {type(self.schema).__name__}")
        if not 0 <= self.schema < 2**8:
            raise EncodeError(f"Header schema {self.schema} out of range for uint8")

    def encode(self) -> bytes:
        """Encode header to bytes."""
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
        """Decode header from bytes."""
        if len(data) < HEADER_SIZE:
            raise DecodeError(f"Need {HEADER_SIZE} bytes for header, got {len(data)}")
        try:
            size_words, msg_type, schema, reserved = struct.unpack("<IBBH", data[:HEADER_SIZE])
        except struct.error as e:  # pragma: no cover
            # Defensive: ``struct.unpack`` of the fixed-size ``<IBBH``
            # format on a guaranteed-8-byte slice cannot fail with
            # ``struct.error`` — the length check above ensures the
            # slice has exactly ``HEADER_SIZE`` bytes. Kept as a
            # belt-and-braces guard against future format changes.
            raise DecodeError(f"Failed to decode header: {e}") from e
        # Upstream C (message.h) reserves the trailing uint16 and every
        # current server writes 0. Reject non-zero values so peer
        # corruption or a future schema extension surfaces as a clean
        # DecodeError instead of silently carrying bits we cannot
        # re-emit. Matches LeaderRequest.decode_body's strict check.
        if reserved != 0:
            raise DecodeError(f"Header reserved field must be 0, got {reserved}")
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
        # Every built-in Message subclass already emits a word-aligned
        # body (text / blob / params / tuple encoders all pad
        # themselves). The previous silent ``body += b"\x00" * pad``
        # hid subclass bugs by the time the misshapen body reached the
        # peer. Upstream's C encoder asserts ``len % 8 == 0``
        # (``dqlite_assert(_n % 8 == 0)`` in ``gateway.c`` SUCCESS /
        # failure macros); match the invariant at encode time so
        # regressions fail loudly here rather than surfacing as a
        # strict-decode rejection at the peer.
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
