"""Pin: ``WriteBuffer.write_padded`` peak-allocates one payload-sized
buffer on the padding path, not two."""

from __future__ import annotations

import tracemalloc

from dqlitewire.buffer import WriteBuffer
from dqlitewire.constants import WORD_SIZE


def test_write_padded_wire_bytes_unchanged_under_padding() -> None:
    buf = WriteBuffer()
    data = b"hello"
    buf.write_padded(data)
    pad = (-len(data)) % WORD_SIZE
    assert buf.getvalue() == data + b"\x00" * pad


def test_write_padded_no_padding_fast_path_unchanged() -> None:
    """Word-aligned input bypasses the padding branch."""
    buf = WriteBuffer()
    data = b"x" * (3 * WORD_SIZE)
    buf.write_padded(data)
    assert buf.getvalue() == data


def test_write_padded_peak_allocation_bounded_to_one_payload() -> None:
    """Padding-branch peak allocation must stay under 2 * len(data)."""
    payload_size = 1024 * 1024
    data = b"\x00" * (payload_size + 5)  # forces 3-byte padding

    buf = WriteBuffer()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    buf.write_padded(data)
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap_after.compare_to(snap_before, "filename")
    total_alloc = sum(s.size_diff for s in stats if s.size_diff > 0)

    # ~1 payload post-fix vs ~2 pre-fix; 1.6x absorbs sampling noise.
    assert total_alloc < int(1.6 * payload_size), (
        f"peak allocation {total_alloc} exceeds 1.6 * payload "
        f"({int(1.6 * payload_size)}) — write_padded may be back to "
        f"the two-allocation shape"
    )


def test_write_padded_accepts_bytearray() -> None:
    buf = WriteBuffer()
    data = bytearray(b"abcde")
    buf.write_padded(data)
    assert buf.getvalue() == bytes(data) + b"\x00" * 3


def test_write_padded_accepts_memoryview() -> None:
    buf = WriteBuffer()
    data = memoryview(b"abcde")
    buf.write_padded(data)
    assert buf.getvalue() == b"abcde" + b"\x00" * 3
