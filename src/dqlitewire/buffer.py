"""Buffer utilities for streaming protocol data."""

from dqlitewire.constants import HEADER_SIZE, WORD_SIZE
from dqlitewire.exceptions import DecodeError


class WriteBuffer:
    """Buffer for building wire protocol messages."""

    def __init__(self) -> None:
        self._data = bytearray()

    def write(self, data: bytes) -> None:
        """Append data to buffer."""
        self._data.extend(data)

    def write_padded(self, data: bytes) -> None:
        """Append data with padding to word boundary."""
        self._data.extend(data)
        remainder = len(data) % WORD_SIZE
        if remainder:
            self._data.extend(b"\x00" * (WORD_SIZE - remainder))

    def getvalue(self) -> bytes:
        """Get buffer contents."""
        return bytes(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        """Clear the buffer."""
        self._data.clear()


class ReadBuffer:
    """Buffer for reading wire protocol messages from a stream.

    Handles partial reads and message framing.
    """

    DEFAULT_MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64 MiB

    def __init__(self, max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE) -> None:
        self._data = bytearray()
        self._pos = 0
        self._max_message_size = max_message_size
        self._skip_remaining = 0

    def feed(self, data: bytes) -> None:
        """Add received data to the buffer.

        If an oversized message is being skipped (after skip_message() returned
        False), incoming bytes are silently discarded until the full oversized
        message has been consumed.
        """
        if self._skip_remaining > 0:
            discard = min(len(data), self._skip_remaining)
            data = data[discard:]
            self._skip_remaining -= discard
            if not data:
                return
        self._maybe_compact()
        unconsumed = len(self._data) - self._pos + len(data)
        if unconsumed > self._max_message_size:
            raise DecodeError(f"Buffer size {unconsumed} exceeds maximum {self._max_message_size}")
        self._data.extend(data)

    def has_message(self) -> bool:
        """Check if a complete message is available."""
        available = len(self._data) - self._pos

        if available < HEADER_SIZE:
            return False

        # Read size from header (first 4 bytes = size in words)
        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        total_size = HEADER_SIZE + (size_words * WORD_SIZE)

        if total_size > self._max_message_size:
            raise DecodeError(
                f"Message size {total_size} bytes exceeds maximum {self._max_message_size}"
            )

        return available >= total_size

    def peek_header(self) -> tuple[int, int, int] | None:
        """Peek at the message header without consuming it.

        Returns (size_in_words, message_type, schema_version) or None if not enough data.
        """
        available = len(self._data) - self._pos

        if available < HEADER_SIZE:
            return None

        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        msg_type = self._data[self._pos + 4]
        schema_version = self._data[self._pos + 5]

        return size_words, msg_type, schema_version

    def read_message(self) -> bytes | None:
        """Read a complete message from the buffer.

        Returns the message data (including header) or None if not enough data.
        """
        if not self.has_message():
            return None

        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        total_size = HEADER_SIZE + (size_words * WORD_SIZE)

        message = bytes(self._data[self._pos : self._pos + total_size])
        self._pos += total_size
        self._maybe_compact()

        return message

    def skip_message(self) -> bool:
        """Skip the current message in the buffer.

        For normal-sized messages (within max_message_size), waits until the
        full message is available before skipping. For oversized messages that
        exceed max_message_size, discards available bytes and tracks the
        remainder via ``_skip_remaining``; subsequent ``feed()`` calls will
        silently discard the remaining oversized bytes.

        Returns True if a message was fully skipped, False if not enough data
        for a header, a normal-sized message is still incomplete, or an
        oversized message skip is still in progress (check ``is_skipping``).
        """
        available = len(self._data) - self._pos
        if available < HEADER_SIZE:
            return False

        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        total_size = HEADER_SIZE + (size_words * WORD_SIZE)

        if total_size <= self._max_message_size:
            # Normal-sized message: only skip when complete to avoid
            # stream corruption from partially consumed messages.
            if available < total_size:
                return False
            self._pos += total_size
        else:
            # Oversized message: discard what we have and track remaining
            # bytes to be discarded in subsequent feed() calls.
            skip_now = min(total_size, available)
            self._pos += skip_now
            self._skip_remaining = total_size - skip_now

        self._maybe_compact()
        return self._skip_remaining == 0

    def read_bytes(self, n: int) -> bytes | None:
        """Read exactly n bytes from the buffer.

        Returns None if not enough data available.
        """
        available = len(self._data) - self._pos
        if available < n:
            return None

        data = bytes(self._data[self._pos : self._pos + n])
        self._pos += n
        self._maybe_compact()
        return data

    def _maybe_compact(self) -> None:
        """Compact buffer if we've consumed a lot."""
        if self._pos > 4096:
            self._data = self._data[self._pos :]
            self._pos = 0

    def available(self) -> int:
        """Return number of bytes available to read."""
        return len(self._data) - self._pos

    @property
    def is_skipping(self) -> bool:
        """True if still discarding bytes from an oversized message."""
        return self._skip_remaining > 0

    def clear(self) -> None:
        """Clear the buffer."""
        self._data.clear()
        self._pos = 0
        self._skip_remaining = 0
