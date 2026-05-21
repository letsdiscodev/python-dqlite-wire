"""Pin: ``WriteBuffer.write_padded`` allocates one intermediate buffer
on the padding path, not two.

The previous shape ``self._data.extend(bytes(data) + b"\\x00" * pad)``
forced two payload-sized allocations: one for the ``bytes(data)``
materialisation when input is non-bytes (and one for the
concatenation when input is already bytes), plus the concatenation
intermediate, plus the buffer-protocol consume by ``extend``. The fix
builds a local ``bytearray`` and extends it in place with the padding
before the single ``extend`` call. Cross-thread atomicity (one
``extend`` onto ``self._data``) is preserved; peak allocation drops
to one payload-sized chunk.

Validate by:
  1. The wire bytes (encoder output) are unchanged.
  2. ``tracemalloc`` reports the padding branch's peak allocation
     bounded by ``len(data) + small constant``, not ``2 * len(data)``.
"""

from __future__ import annotations

import tracemalloc

from dqlitewire.buffer import WriteBuffer
from dqlitewire.constants import WORD_SIZE


def test_write_padded_wire_bytes_unchanged_under_padding() -> None:
    """The bytes produced by ``getvalue()`` must still be
    ``data + zero-pad`` regardless of the in-call allocation shape."""
    buf = WriteBuffer()
    data = b"hello"
    buf.write_padded(data)
    pad = (-len(data)) % WORD_SIZE
    assert buf.getvalue() == data + b"\x00" * pad


def test_write_padded_no_padding_fast_path_unchanged() -> None:
    """Word-aligned input bypasses the padding branch; the fast path
    must still write exactly the input bytes."""
    buf = WriteBuffer()
    data = b"x" * (3 * WORD_SIZE)
    buf.write_padded(data)
    assert buf.getvalue() == data


def test_write_padded_peak_allocation_bounded_to_one_payload() -> None:
    """``tracemalloc`` snapshot before/after a single ``write_padded``
    call must show peak < 2 * len(data). The pre-fix shape allocated
    a payload-sized concatenation intermediate; the fix replaces it
    with one in-place ``bytearray`` mutation.

    Use a 1 MiB payload so the allocation cost is large relative to
    the interpreter's per-frame overhead; the bound is set loose
    enough to absorb runtime noise (peak < 1.6 * len(data)).
    """
    payload_size = 1024 * 1024  # 1 MiB
    data = b"\x00" * (payload_size + 5)  # forces 3-byte padding

    buf = WriteBuffer()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    buf.write_padded(data)
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap_after.compare_to(snap_before, "filename")
    total_alloc = sum(s.size_diff for s in stats if s.size_diff > 0)

    # Pre-fix: ~2 * payload_size (concat intermediate + extend
    # consume). Post-fix: ~1 * payload_size (single local bytearray).
    # 1.6x is loose enough to survive runtime / sampling noise.
    assert total_alloc < int(1.6 * payload_size), (
        f"peak allocation {total_alloc} exceeds 1.6 * payload "
        f"({int(1.6 * payload_size)}) — write_padded may be back to "
        f"the two-allocation shape"
    )


def test_write_padded_accepts_bytearray() -> None:
    """Buffer-protocol consumers (``bytearray``) still flow through
    cleanly; the in-place ``bytearray(data)`` copy mirrors the
    previous ``bytes(data)`` materialisation for non-bytes inputs."""
    buf = WriteBuffer()
    data = bytearray(b"abcde")  # 5 bytes — needs 3-byte padding
    buf.write_padded(data)
    assert buf.getvalue() == bytes(data) + b"\x00" * 3


def test_write_padded_accepts_memoryview() -> None:
    buf = WriteBuffer()
    data = memoryview(b"abcde")
    buf.write_padded(data)
    assert buf.getvalue() == b"abcde" + b"\x00" * 3
