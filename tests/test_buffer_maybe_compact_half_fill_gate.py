"""Pin: ``Buffer._maybe_compact`` only compacts when at least half
the buffer has been consumed.

The prior gate fired on ``self._pos > _COMPACT_THRESHOLD`` (4096
bytes). That allowed a single compact to memcpy a multi-MiB tail
when a near-full buffer happened to have just over 4 KiB consumed
— up to ~64 MiB synchronous loop CPU on a max-cap buffer.

The half-fill gate amortises the worst-case copy at 2× per byte:
each byte is copied at most twice over its lifetime in the buffer.
Worst-case single-compact memcpy is now bounded at half the
buffer size.

The slow-drip "never-compacts" concern is bounded by the buffer's
maximum message size: ``feed`` rejects oversized frames upstream
(``DecodeError`` once the projected size exceeds
``max_message_size``), so ``len(_data)`` cannot grow without bound
while the half-fill gate withholds compaction.
"""

from __future__ import annotations

from dqlitewire.buffer import _COMPACT_THRESHOLD, ReadBuffer


def test_compact_does_not_fire_when_below_half_fill() -> None:
    """A buffer with a 1-MiB payload and 5-KiB consumed (>4096 but
    far below half-fill) must NOT compact yet."""
    buf = ReadBuffer()
    buf.feed(b"\x00" * (1 << 20))  # 1 MiB
    # Consume just over the prior gate's threshold (4096) but well
    # below half-fill (~512 KiB).
    buf._pos = _COMPACT_THRESHOLD + 1024
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    # Under the half-fill gate, _data is NOT reallocated; pos
    # stays where it was.
    assert id(buf._data) == pre_data_id, "compact fired when below half-fill; expected no-op"
    assert buf._pos == _COMPACT_THRESHOLD + 1024


def test_compact_fires_when_above_half_fill() -> None:
    """When more than half of the buffer is consumed AND _pos is
    above the pre-existing 4096 threshold, compact runs.
    """
    buf = ReadBuffer()
    total = 20_000  # well above _COMPACT_THRESHOLD * 2
    buf.feed(b"\x00" * total)
    buf._pos = 12_000  # > 4096 AND > total / 2 (= 10_000)
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    # Compact fired: _data is a fresh bytearray, _pos reset to 0.
    assert id(buf._data) != pre_data_id, "compact did not fire above half-fill; expected compaction"
    assert buf._pos == 0
    assert len(buf._data) == total - 12_000


def test_compact_does_not_fire_when_pos_above_threshold_but_below_half() -> None:
    """The new gate's discriminating case: _pos is above the prior
    4096-byte threshold (so the prior shape would compact) but below
    half of the buffer (so the new shape must NOT compact). This is
    the case that motivated the fix — a near-full buffer with a
    small consumed prefix would have memcpy'd nearly the whole
    buffer for negligible gain.
    """
    buf = ReadBuffer()
    buf.feed(b"\x00" * (1 << 20))  # 1 MiB
    buf._pos = _COMPACT_THRESHOLD + 1  # just over the prior gate
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    assert id(buf._data) == pre_data_id, (
        "compact fired with _pos just over 4096 in a 1 MiB buffer; "
        "the half-fill gate should have suppressed the copy"
    )
    assert buf._pos == _COMPACT_THRESHOLD + 1


def test_compact_no_op_below_4096() -> None:
    """The pre-existing ``_pos <= _COMPACT_THRESHOLD`` guard still
    short-circuits when very little is consumed."""
    buf = ReadBuffer()
    buf.feed(b"\x00" * 10000)
    buf._pos = 100  # under threshold
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    assert id(buf._data) == pre_data_id
    assert buf._pos == 100


def test_compact_amortises_under_burst_fill_pattern() -> None:
    """Under a burst-fill pattern (large message arrives all at
    once, then small consumed prefix), total bytes copied across
    compacts is bounded — each byte is copied at most twice over
    its lifetime in the buffer.
    """
    buf = ReadBuffer()
    total_fed = 0
    total_copied = 0
    for _ in range(10):
        buf.feed(b"\x00" * (64 * 1024))
        total_fed += 64 * 1024
        # Consume just over half of what's currently in the buffer.
        consumable = len(buf._data) - buf._pos
        buf._pos += consumable // 2 + 1
        # Compact copy size equals the bytes that remain at the time
        # the compact fires (the half not yet consumed).
        about_to_copy = len(buf._data) - buf._pos
        buf._maybe_compact()
        total_copied += about_to_copy

    assert total_copied <= total_fed, (
        f"copied {total_copied} bytes for {total_fed} fed; "
        "half-fill gate should amortise under burst-fill"
    )
