"""``_maybe_compact`` only compacts when at least half the buffer is consumed,
bounding a single memcpy to half the buffer (each byte copied at most twice over
its lifetime) instead of memcpy'ing a near-full buffer for a small consumed prefix."""

from __future__ import annotations

from dqlitewire.buffer import _COMPACT_THRESHOLD, ReadBuffer


def test_compact_does_not_fire_when_below_half_fill() -> None:
    """1-MiB payload with ~5 KiB consumed (>4096 but far below half) must not compact."""
    buf = ReadBuffer()
    buf.feed(b"\x00" * (1 << 20))  # 1 MiB
    buf._pos = _COMPACT_THRESHOLD + 1024
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    assert id(buf._data) == pre_data_id, "compact fired when below half-fill; expected no-op"
    assert buf._pos == _COMPACT_THRESHOLD + 1024


def test_compact_fires_when_above_half_fill() -> None:
    """Compact runs when _pos is above 4096 AND more than half is consumed."""
    buf = ReadBuffer()
    total = 20_000
    buf.feed(b"\x00" * total)
    buf._pos = 12_000  # > 4096 AND > total / 2
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    assert id(buf._data) != pre_data_id, "compact did not fire above half-fill; expected compaction"
    assert buf._pos == 0
    assert len(buf._data) == total - 12_000


def test_compact_does_not_fire_when_pos_above_threshold_but_below_half() -> None:
    """Discriminating case: _pos above 4096 (prior gate would compact) but below
    half (new gate must not) — the motivating near-full-buffer scenario."""
    buf = ReadBuffer()
    buf.feed(b"\x00" * (1 << 20))
    buf._pos = _COMPACT_THRESHOLD + 1  # just over the prior gate
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    assert id(buf._data) == pre_data_id, (
        "compact fired with _pos just over 4096 in a 1 MiB buffer; "
        "the half-fill gate should have suppressed the copy"
    )
    assert buf._pos == _COMPACT_THRESHOLD + 1


def test_compact_no_op_below_4096() -> None:
    """The ``_pos <= _COMPACT_THRESHOLD`` guard still short-circuits."""
    buf = ReadBuffer()
    buf.feed(b"\x00" * 10000)
    buf._pos = 100
    pre_data_id = id(buf._data)

    buf._maybe_compact()

    assert id(buf._data) == pre_data_id
    assert buf._pos == 100


def test_compact_amortises_under_burst_fill_pattern() -> None:
    """Under burst-fill, total bytes copied across compacts stays bounded
    (each byte copied at most twice over its lifetime)."""
    buf = ReadBuffer()
    total_fed = 0
    total_copied = 0
    for _ in range(10):
        buf.feed(b"\x00" * (64 * 1024))
        total_fed += 64 * 1024
        # Consume just over half of what's currently buffered.
        consumable = len(buf._data) - buf._pos
        buf._pos += consumable // 2 + 1
        about_to_copy = len(buf._data) - buf._pos
        buf._maybe_compact()
        total_copied += about_to_copy

    assert total_copied <= total_fed, (
        f"copied {total_copied} bytes for {total_fed} fed; "
        "half-fill gate should amortise under burst-fill"
    )
