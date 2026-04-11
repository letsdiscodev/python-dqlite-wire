"""Buffer utilities for streaming protocol data."""

from dqlitewire.constants import HEADER_SIZE, WORD_SIZE
from dqlitewire.exceptions import DecodeError, ProtocolError


class WriteBuffer:
    """Buffer for building wire protocol messages."""

    def __init__(self) -> None:
        self._data = bytearray()

    def write(self, data: bytes) -> None:
        """Append data to buffer."""
        self._data.extend(data)

    def write_padded(self, data: bytes) -> None:
        """Append data with padding to word boundary.

        The padded bytes are built locally and emitted via a single
        ``bytearray.extend`` so that under accidental concurrent misuse
        (see issue 021 for the single-owner contract) two threads'
        payloads and padding cannot interleave. This still does not make
        ``WriteBuffer`` thread-safe in any strong sense, but it removes
        the torn payload/pad split that used to be visible to callers.
        """
        remainder = len(data) % WORD_SIZE
        if remainder:
            self._data.extend(data + b"\x00" * (WORD_SIZE - remainder))
        else:
            self._data.extend(data)

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
        self._poisoned: Exception | None = None

    @property
    def is_poisoned(self) -> bool:
        """True if a mid-stream error has marked this buffer unrecoverable."""
        return self._poisoned is not None

    def poison(self, error: Exception) -> None:
        """Mark the buffer as unrecoverable.

        Call this when a decode error means the stream offset is no longer
        trustworthy (parsing consumed bytes but failed afterwards). Once
        poisoned, every public method raises ``ProtocolError`` until
        ``reset()`` is called.
        """
        if self._poisoned is None:
            self._poisoned = error

    def reset(self) -> None:
        """Clear buffer state and un-poison. Use after a reconnect."""
        self._data.clear()
        self._pos = 0
        self._skip_remaining = 0
        self._poisoned = None

    def _check_poisoned(self) -> None:
        if self._poisoned is not None:
            raise ProtocolError(
                "buffer is poisoned; call reset() and reconnect"
            ) from self._poisoned

    def feed(self, data: bytes) -> None:
        """Add received data to the buffer.

        If an oversized message is being skipped (after skip_message() returned
        False), incoming bytes are silently discarded until the full oversized
        message has been consumed.

        Raises ``ProtocolError`` if the buffer is poisoned — callers must
        ``reset()`` (or ``clear()``) before feeding further data.
        Raises ``DecodeError`` (non-poisoning, recoverable via
        ``reset()``/``clear()``) if the resulting buffer size would
        exceed ``max_message_size``.

        Signal-safety note (issue 048): the mutation block below is
        wrapped in ``try/except BaseException`` so that any async
        exception leaking out — most notably between the
        ``_maybe_compact()`` return and the subsequent
        ``_data.extend(data)``, a reachable RESUME delivery point in
        3.11+ — poisons the buffer. The ``DecodeError`` size check
        is deliberately kept OUTSIDE the try block so that the
        documented "recoverable oversized-buffer" contract is
        preserved.
        """
        self._check_poisoned()
        # Size check BEFORE any mutation: the DecodeError raised here
        # must NOT poison the buffer, because its caller-recovery
        # contract is "drain or reset() and continue". The check
        # accounts for the would-be skip-discard so that a buffer
        # in skip mode can legitimately accept incoming bytes whose
        # post-discard remainder fits within max_message_size.
        if self._skip_remaining > 0:
            effective_len = max(0, len(data) - self._skip_remaining)
        else:
            effective_len = len(data)
        projected = len(self._data) - self._pos + effective_len
        if projected > self._max_message_size:
            raise DecodeError(f"Buffer size {projected} exceeds maximum {self._max_message_size}")
        try:
            if self._skip_remaining > 0:
                discard = min(len(data), self._skip_remaining)
                data = data[discard:]
                self._skip_remaining -= discard
                if not data:
                    return
            self._maybe_compact()
            self._data.extend(data)
        except BaseException as e:
            # Any torn state here is unrecoverable — subsequent
            # callers MUST see poison rather than silently reading
            # from a buffer with a gap. ``poison()`` is first-error-
            # wins, so if ``_maybe_compact`` already poisoned with
            # its own cause, that original cause is preserved.
            if self._poisoned is None:
                if isinstance(e, Exception):
                    self._poisoned = e
                else:
                    self._poisoned = RuntimeError(f"feed interrupted: {type(e).__name__}")
            raise

    def has_message(self) -> bool:
        """Check if a complete message is available.

        This predicate is total: it never raises. An oversized header
        (claiming a body larger than ``max_message_size``) is reported as
        ``True`` so that the caller's ``while decoder.has_message(): ...``
        loop proceeds to the consume call, where the error actually surfaces.
        The raise lives in ``read_message()`` / ``skip_message()``, not here.
        """
        available = len(self._data) - self._pos

        if available < HEADER_SIZE:
            return False

        # Read size from header (first 4 bytes = size in words)
        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        # Torn-read sanity (issue 051): if the slice was widened by
        # a concurrent realloc, report "something to consume" so the
        # caller's while-loop proceeds to read_message(), which then
        # poisons. has_message() itself is a total predicate and
        # must not raise.
        if size_words > 0xFFFFFFFF:
            return True
        total_size = HEADER_SIZE + (size_words * WORD_SIZE)

        if total_size > self._max_message_size:
            # Report "something to consume" — the consume call will raise.
            return True

        return available >= total_size

    def peek_header(self) -> tuple[int, int, int] | None:
        """Peek at the message header without consuming it.

        Returns (size_in_words, message_type, schema_version) or None if not
        enough data. Raises ``DecodeError`` if the header claims a total
        message size larger than ``max_message_size``.

        Note the deliberate asymmetry with ``has_message()``: that predicate
        is total (returns ``True`` for oversized so the documented
        ``while has_message(): decode()`` loop proceeds to the consume call,
        which then raises). ``peek_header()`` instead raises immediately
        because its return value is an attacker-controlled integer that a
        caller might use directly for allocation, timeout, or routing
        decisions — silently returning an unbounded value would defeat the
        ``max_message_size`` guard the caller is presumably relying on. The
        error message and exception type match ``read_message()`` so the
        consume-side recovery path (``skip_message()``) applies the same way.

        Raises ``ProtocolError`` if the buffer is poisoned.
        """
        self._check_poisoned()
        available = len(self._data) - self._pos

        if available < HEADER_SIZE:
            return None

        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        # Sanity check for torn reads (issue 051). The wire size
        # field is exactly 4 bytes (uint32 little-endian), so any
        # value > 0xFFFFFFFF cannot come from a well-formed header
        # — it can only come from a ``bytearray`` slice that
        # observed torn ``ob_size``/``ob_start`` during a concurrent
        # realloc on a free-threaded build, returning more than 4
        # bytes. Distinguish this from legitimate oversized messages
        # so the non-poisoning ``DecodeError`` recovery contract
        # still applies to real wire-oversized messages while torn
        # reads poison the buffer.
        if size_words > 0xFFFFFFFF:
            err = DecodeError(
                f"torn header read: size_words={size_words:#x} (>32 bits, "
                "indicates concurrent misuse on a free-threaded build)"
            )
            self.poison(err)
            raise err
        total_size = HEADER_SIZE + (size_words * WORD_SIZE)
        if total_size > self._max_message_size:
            # Format size in hex: under concurrent misuse (see issue 033)
            # `total_size` can be a torn bigint whose decimal form exceeds
            # CPython's 4300-digit int-to-str limit, which would make this
            # f-string itself raise ValueError. Hex formatting has no cap.
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
        # Sanity check for torn reads (issue 051). The wire size
        # field is exactly 4 bytes (uint32 little-endian), so any
        # value > 0xFFFFFFFF cannot come from a well-formed header
        # — it can only come from a ``bytearray`` slice that
        # observed torn ``ob_size``/``ob_start`` during a concurrent
        # realloc on a free-threaded build, returning more than 4
        # bytes. Distinguish this from legitimate oversized messages
        # so the non-poisoning ``DecodeError`` recovery contract
        # still applies to real wire-oversized messages while torn
        # reads poison the buffer.
        if size_words > 0xFFFFFFFF:
            err = DecodeError(
                f"torn header read: size_words={size_words:#x} (>32 bits, "
                "indicates concurrent misuse on a free-threaded build)"
            )
            self.poison(err)
            raise err
        total_size = HEADER_SIZE + (size_words * WORD_SIZE)

        if total_size > self._max_message_size:
            # Format size in hex: under concurrent misuse (see issue 033)
            # `total_size` can be a torn bigint whose decimal form exceeds
            # CPython's 4300-digit int-to-str limit, which would make this
            # f-string itself raise ValueError. Hex formatting has no cap.
            raise DecodeError(
                f"Message size {total_size:#x} bytes exceeds maximum {self._max_message_size}"
            )

        if available < total_size:
            return None

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

        Raises ``ProtocolError`` if the buffer is poisoned.
        """
        self._check_poisoned()
        available = len(self._data) - self._pos
        if available < HEADER_SIZE:
            return False

        size_words = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
        # Sanity check for torn reads (issue 051). The wire size
        # field is exactly 4 bytes (uint32 little-endian), so any
        # value > 0xFFFFFFFF cannot come from a well-formed header
        # — it can only come from a ``bytearray`` slice that
        # observed torn ``ob_size``/``ob_start`` during a concurrent
        # realloc on a free-threaded build, returning more than 4
        # bytes. Distinguish this from legitimate oversized messages
        # so the non-poisoning ``DecodeError`` recovery contract
        # still applies to real wire-oversized messages while torn
        # reads poison the buffer.
        if size_words > 0xFFFFFFFF:
            err = DecodeError(
                f"torn header read: size_words={size_words:#x} (>32 bits, "
                "indicates concurrent misuse on a free-threaded build)"
            )
            self.poison(err)
            raise err
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
            # Cap to max_message_size to prevent amplification attacks where
            # an 8-byte header claiming a ~32 GiB body would cause that much
            # legitimate data to be silently discarded.
            effective_total = min(total_size, self._max_message_size)
            skip_now = min(effective_total, available)
            self._pos += skip_now
            self._skip_remaining = effective_total - skip_now

        self._maybe_compact()
        return self._skip_remaining == 0

    def read_bytes(self, n: int) -> bytes | None:
        """Read exactly n bytes from the buffer.

        Returns None if not enough data available.
        Raises ``ProtocolError`` if the buffer is poisoned.
        """
        self._check_poisoned()
        available = len(self._data) - self._pos
        if available < n:
            return None

        data = bytes(self._data[self._pos : self._pos + n])
        self._pos += n
        self._maybe_compact()
        return data

    def peek_bytes(self, n: int) -> bytes | None:
        """Return the next n bytes without advancing the read position.

        Returns None if fewer than n bytes are available. Symmetric with
        ``read_bytes(n)`` but non-consuming — use this when you need to
        validate the bytes before deciding whether to consume them.

        Raises ``ProtocolError`` if the buffer is poisoned.
        """
        self._check_poisoned()
        available = len(self._data) - self._pos
        if available < n:
            return None
        return bytes(self._data[self._pos : self._pos + n])

    def _maybe_compact(self) -> None:
        """Compact buffer if we've consumed a lot.

        Signal-safety note (issue 037): this method mutates two
        attributes — ``_data`` and ``_pos`` — which compile to two
        ``STORE_ATTR`` bytecodes. CPython checks for pending signals
        at bytecode line transitions, so a ``KeyboardInterrupt`` (or
        any ``PyErr_SetAsyncExc`` delivery) landing between the two
        stores used to leave the buffer with a freshly compacted
        ``_data`` but a stale ``_pos`` still pointing at the old
        offset. ``available()`` would return a negative number, reads
        would silently return nonsense, and no poison fired because
        no exception originated inside the buffer — the interrupt was
        purely external. A single-owner caller pressing Ctrl-C during
        a busy decode loop would end up with silent message dropout.

        We cannot make the two stores atomic at the Python level, but
        we can catch ``BaseException`` and poison so that the next
        caller fails fast with ``ProtocolError`` instead of reading
        from an inconsistent offset.
        """
        if self._pos <= 4096:
            return
        try:
            new_data = self._data[self._pos :]
            self._data = new_data
            self._pos = 0
        except BaseException as e:
            # Any torn state here is unrecoverable — the next caller
            # MUST see poison, not a silently inconsistent buffer.
            # ``poison()`` is first-error-wins and its body is a
            # single ``STORE_ATTR`` so cannot itself be split.
            if self._poisoned is None:
                if isinstance(e, Exception):
                    self._poisoned = e
                else:
                    self._poisoned = RuntimeError(f"buffer compact interrupted: {type(e).__name__}")
            raise

    def available(self) -> int:
        """Return number of bytes available to read."""
        return len(self._data) - self._pos

    @property
    def is_skipping(self) -> bool:
        """True if still discarding bytes from an oversized message."""
        return self._skip_remaining > 0

    def clear(self) -> None:
        """Clear buffer state and un-poison.

        Equivalent to ``reset()``. Kept as a convenience alias because
        ``clear()`` predates the poison concept (issue 026) and was
        briefly inconsistent with ``reset()`` — it used to leave the
        ``_poisoned`` flag intact, which meant a caller who reached
        for ``clear()`` as a recovery primitive got a half-fresh
        buffer that still raised ``ProtocolError`` on the next
        operation (issue 040).
        """
        self.reset()
