"""Buffer utilities for streaming protocol data."""

from dqlitewire.constants import HEADER_SIZE, WORD_SIZE


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

    def __init__(self) -> None:
        self._data = bytearray()
        self._pos = 0

    def feed(self, data: bytes) -> None:
        """Add received data to the buffer."""
        self._data.extend(data)

    def has_message(self) -> bool:
        """Check if a complete message is available."""
        available = len(self._data) - self._pos

        if available < HEADER_SIZE:
            return False

        # Read size from header (first 4 bytes = size in words)
        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        total_size = HEADER_SIZE + (size_words * WORD_SIZE)

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

        # Compact buffer if we've consumed a lot
        if self._pos > 4096:
            self._data = self._data[self._pos :]
            self._pos = 0

        return message

    def read_bytes(self, n: int) -> bytes | None:
        """Read exactly n bytes from the buffer.

        Returns None if not enough data available.
        """
        available = len(self._data) - self._pos
        if available < n:
            return None

        data = bytes(self._data[self._pos : self._pos + n])
        self._pos += n
        return data

    def available(self) -> int:
        """Return number of bytes available to read."""
        return len(self._data) - self._pos

    def clear(self) -> None:
        """Clear the buffer."""
        self._data.clear()
        self._pos = 0
