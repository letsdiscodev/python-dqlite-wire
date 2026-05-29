"""Buffer utilities for streaming protocol data."""

from typing import ClassVar, Final, NoReturn

from dqlitewire.constants import HEADER_SIZE, WORD_SIZE
from dqlitewire.exceptions import DecodeError, PoisonedError

__all__ = [
    "ReadBuffer",
    "WriteBuffer",
]

_COMPACT_THRESHOLD: Final[int] = 4096


class WriteBuffer:
    """Buffer for building wire protocol messages.

    NOT thread-safe: single-owner (one thread/coroutine). ``write_padded``
    guards only against a torn payload/pad interleave, not general concurrency.
    """

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — instances "
            f"hold mutable buffered bytes under a single-owner discipline; "
            f"share by re-creating in the target process."
        )

    def __init__(self) -> None:
        self._data = bytearray()

    def write(self, data: bytes | bytearray | memoryview) -> None:
        self._data.extend(data)

    def write_padded(self, data: bytes | bytearray | memoryview) -> None:
        """Append data with NUL padding to the next word boundary."""
        remainder = len(data) % WORD_SIZE
        if remainder:
            # Single ``extend`` of a pre-built local so payload and pad
            # cannot be torn apart by a concurrent writer, and to avoid
            # the second allocation of a ``bytes(data) + pad`` concat.
            chunk = bytearray(data)
            chunk += b"\x00" * (WORD_SIZE - remainder)
            self._data.extend(chunk)
        else:
            self._data.extend(data)

    def getvalue(self) -> bytes:
        return bytes(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


class ReadBuffer:
    """Buffer for reading wire protocol messages from a stream; handles
    partial reads and framing.

    NOT thread-safe: single-owner (one thread/coroutine). Concurrent misuse
    silently corrupts data (lost ``_pos`` updates, torn ``_pos``/``_data``
    snapshots across ``_maybe_compact()``) — ``poison()`` cannot detect this.
    """

    DEFAULT_MAX_MESSAGE_SIZE: ClassVar[int] = 64 * 1024 * 1024  # 64 MiB
    # Mirrors the C server's per-frame ceiling at dqlite-upstream/src/conn.c:169
    # (rejects any single frame above UINT32_MAX bytes), so a misconfigured
    # max_message_size above it can never disable the composite-frame guard.
    MAX_MESSAGE_SIZE_CEILING: ClassVar[int] = 0xFFFFFFFF

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — instances "
            f"hold per-stream-position mutable state under a single-owner "
            f"discipline; share by re-creating in the target process."
        )

    def __init__(self, max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE) -> None:
        if max_message_size < 1:
            raise ValueError(f"max_message_size must be >= 1, got {max_message_size}")
        if max_message_size > self.MAX_MESSAGE_SIZE_CEILING:
            raise ValueError(
                f"max_message_size must be <= {self.MAX_MESSAGE_SIZE_CEILING} "
                f"(UINT32_MAX bytes; the C server's per-frame ceiling at "
                f"dqlite-upstream/src/conn.c:169 rejects any single frame above this "
                f"bound), got {max_message_size}"
            )
        self._data = bytearray()
        self._pos = 0
        self._max_message_size = max_message_size
        self._skip_remaining = 0
        self._poisoned: Exception | None = None
        self._poison_after_skip: Exception | None = None

    @property
    def is_poisoned(self) -> bool:
        """True if a mid-stream error has marked this buffer unrecoverable."""
        return self._poisoned is not None

    def poison(self, error: Exception) -> None:
        """Mark the buffer unrecoverable; every method raises until reset().

        Call when a decode error means the stream offset is no longer trustworthy
        (bytes were consumed but parsing failed afterwards).
        """
        if self._poisoned is None:
            self._poisoned = error

    def reset(self) -> None:
        """Clear buffer state and un-poison. Use after a reconnect."""
        self._data.clear()
        self._pos = 0
        self._skip_remaining = 0
        self._poisoned = None
        self._poison_after_skip = None

    def _check_poisoned(self) -> None:
        if self._poisoned is not None:
            raise PoisonedError(
                "buffer is poisoned; call reset() and reconnect"
            ) from self._poisoned

    def _check_torn_size(self, size_words: int) -> None:
        """Poison and raise if size_words exceeds the 4-byte field width.

        A value > 0xFFFFFFFF can only come from a torn read (concurrent realloc
        on a free-threaded build), distinct from a legitimate oversized message.
        """
        if size_words > 0xFFFFFFFF:
            err = DecodeError(
                f"torn header read: size_words={size_words:#x} (>32 bits, "
                "indicates concurrent misuse on a free-threaded build)"
            )
            self.poison(err)
            raise err

    def feed(self, data: bytes) -> None:
        """Add received data to the buffer; discard bytes while mid-skip of an
        oversized message.

        Raises ``ProtocolError`` if poisoned. Raises a non-poisoning
        ``DecodeError`` if the buffer would exceed ``max_message_size``. The
        rejected ``data`` is never appended: ``reset()`` recovery is only safe
        when the buffer was empty with no skip in flight — otherwise the reject
        self-poisons because the buffered prefix is unconsumed wire state that
        ``reset()`` would clobber and desync. (The read-side oversize path in
        ``read_message`` stays recoverable via ``skip_message``.)
        """
        self._check_poisoned()
        # Reject a chunk > 2x max_message_size at entry (legal messages fit one
        # cap, bursty reads occasionally double) before upstream allocation grows
        # unbounded; the size check stays OUTSIDE the try below so a safe-reset
        # caller is not poisoned.
        if len(data) > 2 * self._max_message_size:
            err = DecodeError(
                f"feed data ({len(data)}) exceeds 2x max_message_size "
                f"({self._max_message_size}); split into smaller chunks"
            )
            # Self-poison when any in-flight wire state is present (mid-skip, or
            # any unconsumed buffered bytes) so the safe-reset case cannot be
            # misapplied — reset() would clobber the prefix and mis-frame.
            inflight_state = (len(self._data) - self._pos) > 0
            if self._skip_remaining > 0 or inflight_state:
                if self._skip_remaining > 0:
                    self._skip_remaining = 0
                    self._poison_after_skip = None
                self.poison(err)
            raise err
        # Projection check before any mutation (must not poison). Account for
        # the would-be skip-discard so a buffer mid-skip can accept bytes whose
        # post-discard remainder fits within max_message_size.
        if self._skip_remaining > 0:
            effective_len = max(0, len(data) - self._skip_remaining)
        else:
            effective_len = len(data)
        projected = len(self._data) - self._pos + effective_len
        if projected > self._max_message_size:
            err = DecodeError(f"Buffer size {projected} exceeds maximum {self._max_message_size}")
            # Self-poison when any in-flight wire state is present (symmetric
            # with the entry-side reject above) so reset() recovery cannot be
            # misapplied — it would clobber the prefix and mis-frame.
            inflight_state = (len(self._data) - self._pos) > 0
            if self._skip_remaining > 0 or inflight_state:
                if self._skip_remaining > 0:
                    # Clear skip-tracking for consistent post-poison state.
                    self._skip_remaining = 0
                    self._poison_after_skip = None
                self.poison(err)
            raise err
        try:
            if self._skip_remaining > 0:
                discard = min(len(data), self._skip_remaining)
                data = data[discard:]
                self._skip_remaining -= discard
                # Fire deferred poison once the capped skip completes: the stream
                # is desynchronized, so further reads would be garbage.
                if self._skip_remaining == 0 and self._poison_after_skip is not None:
                    self._poisoned = self._poison_after_skip
                    self._poison_after_skip = None
                if not data:
                    return
            self._maybe_compact()
            self._data.extend(data)
        except BaseException as e:
            # Torn state here is unrecoverable; poison so the next caller fails
            # fast rather than reading from a buffer with a gap (first-error-wins).
            if self._poisoned is None:
                if isinstance(e, Exception):
                    self._poisoned = e
                else:
                    self._poisoned = RuntimeError(f"feed interrupted: {type(e).__name__}")
            raise

    def has_message(self) -> bool:
        """Check if a complete message is available; total (never raises).

        Only ``size_words`` is inspected. An oversized header reports ``True`` so
        the caller's ``while has_message():`` loop reaches the consume call, where
        the error actually surfaces; ``True`` does not imply the subsequent
        decode succeeds (e.g. a non-zero ``reserved`` field).
        """
        available = len(self._data) - self._pos

        if available < HEADER_SIZE:
            return False

        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        if size_words > 0xFFFFFFFF:  # pragma: no cover
            # Torn read from concurrent slice-widening; report "consumable" so
            # read_message() poisons. Caps at 0xFFFFFFFF, so unreachable in tests.
            return True
        total_size = HEADER_SIZE + (size_words * WORD_SIZE)

        if total_size > self._max_message_size:
            return True

        return available >= total_size

    def peek_header(self) -> tuple[int, int, int] | None:
        """Peek the header without consuming it.

        Returns (size_in_words, message_type, schema_version) or None if short.
        Raises ``DecodeError`` if the claimed size exceeds ``max_message_size``
        (raised immediately, unlike ``has_message()``, because the returned size
        is attacker-controlled and a caller may use it for allocation/routing).
        ``msg_type``/``schema_version`` are raw: a successful peek does not imply
        the subsequent ``Header.decode``/codec schema check will succeed.
        Raises ``ProtocolError`` if poisoned.
        """
        self._check_poisoned()
        available = len(self._data) - self._pos

        if available < HEADER_SIZE:
            return None

        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        self._check_torn_size(size_words)
        total_size = HEADER_SIZE + (size_words * WORD_SIZE)
        if total_size > self._max_message_size:
            # Hex format: a torn bigint's decimal form can exceed CPython's
            # 4300-digit int-to-str limit and make this f-string itself raise.
            raise DecodeError(
                f"Message size {total_size:#x} bytes exceeds maximum {self._max_message_size}"
            )

        msg_type = self._data[self._pos + 4]
        schema_version = self._data[self._pos + 5]

        return size_words, msg_type, schema_version

    def read_message(self) -> bytes | None:
        """Read a complete message from the buffer.

        Returns the message data (including header) or None if not enough data.
        Raises ``DecodeError`` if the next buffered header claims a message
        larger than ``max_message_size``. Use ``skip_message()`` to recover.
        Raises ``ProtocolError`` if the buffer is poisoned.
        """
        self._check_poisoned()
        available = len(self._data) - self._pos
        if available < HEADER_SIZE:
            return None

        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        self._check_torn_size(size_words)
        total_size = HEADER_SIZE + (size_words * WORD_SIZE)

        if total_size > self._max_message_size:
            # Hex format: a torn bigint's decimal form can exceed CPython's
            # 4300-digit int-to-str limit and make this f-string itself raise.
            raise DecodeError(
                f"Message size {total_size:#x} bytes exceeds maximum {self._max_message_size}"
            )

        if available < total_size:
            return None

        try:
            message = bytes(self._data[self._pos : self._pos + total_size])
            self._pos += total_size
            self._maybe_compact()
        except BaseException as e:
            if self._poisoned is None:
                if isinstance(e, Exception):
                    self._poisoned = e
                else:
                    self._poisoned = RuntimeError(f"read_message interrupted: {type(e).__name__}")
            raise

        return message

    def skip_message(self) -> bool:
        """Skip the current message; True only if fully skipped AND still usable.

        Normal messages skip only when complete. Oversized messages discard
        available bytes and track the rest via ``_skip_remaining`` for ``feed()``
        to drain. Returns False on a short header, an incomplete message, an
        in-progress skip (``is_skipping``), or a synchronous cap-poison.
        Raises ``ProtocolError`` if poisoned.
        """
        self._check_poisoned()
        available = len(self._data) - self._pos
        if available < HEADER_SIZE:
            return False

        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        self._check_torn_size(size_words)
        total_size = HEADER_SIZE + (size_words * WORD_SIZE)

        # Skip a normal-sized message only when complete (partial skip corrupts
        # the stream).
        if total_size <= self._max_message_size and available < total_size:
            return False

        try:
            if total_size <= self._max_message_size:
                self._pos += total_size
            else:
                # Oversized: discard what we have, track the rest for feed().
                # Cap to max_message_size to block amplification (an 8-byte
                # header claiming ~32 GiB discarding that much real data).
                effective_total = min(total_size, self._max_message_size)
                skip_now = min(effective_total, available)
                self._pos += skip_now
                self._skip_remaining = effective_total - skip_now
                # Cap < actual message: poison, but defer while feed() still
                # has bytes to discard (it poisons once _skip_remaining hits 0).
                if effective_total < total_size:
                    if self._skip_remaining == 0:
                        # Capped skip already done — poison now and return False
                        # (buffer unusable) so the caller's
                        # ``if skip_message(): continue`` does not proceed.
                        self.poison(
                            DecodeError(
                                f"Oversized message skip capped to "
                                f"{effective_total} of {total_size} bytes; "
                                f"stream is desynchronized. Call reset()."
                            )
                        )
                        self._maybe_compact()
                        return False
                    else:
                        self._poison_after_skip = DecodeError(
                            f"Oversized message skip capped to "
                            f"{effective_total} of {total_size} bytes; "
                            f"stream is desynchronized. Call reset()."
                        )
            self._maybe_compact()
        except BaseException as e:
            if self._poisoned is None:
                if isinstance(e, Exception):
                    self._poisoned = e
                else:
                    self._poisoned = RuntimeError(f"skip_message interrupted: {type(e).__name__}")
            raise

        return self._skip_remaining == 0

    def read_bytes(self, n: int) -> bytes | None:
        """Read exactly n bytes, or None if fewer are available.

        Raises ``ProtocolError`` if poisoned, ``ValueError`` if *n* is negative.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self._check_poisoned()
        available = len(self._data) - self._pos
        if available < n:
            return None

        try:
            data = bytes(self._data[self._pos : self._pos + n])
            self._pos += n
            self._maybe_compact()
        except BaseException as e:
            if self._poisoned is None:
                if isinstance(e, Exception):
                    self._poisoned = e
                else:
                    self._poisoned = RuntimeError(f"read_bytes interrupted: {type(e).__name__}")
            raise
        return data

    def peek_bytes(self, n: int) -> bytes | None:
        """Return the next n bytes without advancing, or None if fewer available.

        Non-consuming counterpart to ``read_bytes(n)``: validate before consuming.
        Raises ``ProtocolError`` if poisoned, ``ValueError`` if *n* is negative.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self._check_poisoned()
        available = len(self._data) - self._pos
        if available < n:
            return None
        return bytes(self._data[self._pos : self._pos + n])

    def _maybe_compact(self) -> None:
        """Compact the buffer once enough has been consumed.

        Catch BaseException and poison: an async exception (KeyboardInterrupt /
        PyErr_SetAsyncExc) between the ``_data`` and ``_pos`` stores would leave
        a compacted ``_data`` with a stale ``_pos`` and silently corrupt reads.
        """
        if self._pos <= _COMPACT_THRESHOLD:
            return
        # Only compact when at least half consumed, so each byte is copied at
        # most twice over its lifetime (bounds worst-case single-compact CPU).
        if self._pos < len(self._data) // 2:
            return
        try:
            new_data = self._data[self._pos :]
            self._data = new_data
            self._pos = 0
        except BaseException as e:
            if self._poisoned is None:
                if isinstance(e, Exception):
                    self._poisoned = e
                else:
                    self._poisoned = RuntimeError(f"buffer compact interrupted: {type(e).__name__}")
            raise

    def available(self) -> int:
        return len(self._data) - self._pos

    @property
    def is_skipping(self) -> bool:
        """True if still discarding bytes from an oversized message."""
        return self._skip_remaining > 0

    def clear(self) -> None:
        """Alias for ``reset()`` (clears state and un-poisons)."""
        self.reset()
