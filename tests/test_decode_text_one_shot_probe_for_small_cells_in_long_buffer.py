"""``decode_text`` uses the one-shot path for small TEXT cells even when the
remaining buffer is large. It probes the first
``min(data_len, max_size+1, _TEXT_ONE_SHOT_MAX)`` bytes; only cells exceeding
that probe window escalate to the chunked path. Peak transient stays <= 64 KiB.
"""

from __future__ import annotations

from unittest import mock

from dqlitewire.types import decode_text


def _build_body(cells: list[str]) -> bytes:
    """Build a TEXT-cells body matching the encoder shape: utf-8 + NUL +
    pad-to-word per cell."""

    def pad(n: int) -> int:
        return (-n) & 7

    out = bytearray()
    for cell in cells:
        utf8 = cell.encode("utf-8")
        encoded = utf8 + b"\x00"
        out.extend(encoded)
        out.extend(b"\x00" * pad(len(encoded)))
    return bytes(out)


def test_short_cells_in_long_buffer_use_one_shot_path() -> None:
    """Every short cell in a long buffer round-trips via the one-shot path
    (the next test directly asserts the chunked path never fires)."""
    # 4000 short cells (~64 KB total) spans the prior data_len > 64 KiB regime.
    cells = [f"cell-{i:08d}" for i in range(4000)]
    body = _build_body(cells)

    view = memoryview(body)
    offset = 0
    decoded: list[str] = []
    while offset < len(view):
        text, consumed = decode_text(view[offset:])
        decoded.append(text)
        offset += consumed

    assert decoded == cells, "all cells must round-trip"


def test_chunked_path_only_fires_for_cells_exceeding_probe_window() -> None:
    """The chunked path must NOT fire for short cells in a long buffer."""
    cells = [f"x-{i}" for i in range(1000)]
    body = _build_body(cells)
    # Tail-pad with 1 MiB of NULs so data_len > 64 KiB for the first cells.
    padded = body + b"\x00" * (1 << 20)
    view = memoryview(padded)

    # Detect chunked-path entry by swapping _TEXT_SCAN_CHUNK for a sentinel
    # that counts the index/add ops the chunked path performs on it.
    chunked_path_entered = 0
    import dqlitewire.types as types_mod

    real_scan_chunk = types_mod._TEXT_SCAN_CHUNK

    class _ChunkedSentinel:
        def __index__(self) -> int:
            nonlocal chunked_path_entered
            chunked_path_entered += 1
            return real_scan_chunk

        def __add__(self, other: int) -> int:
            nonlocal chunked_path_entered
            chunked_path_entered += 1
            return real_scan_chunk + other

        def __radd__(self, other: int) -> int:
            nonlocal chunked_path_entered
            chunked_path_entered += 1
            return other + real_scan_chunk

    with mock.patch.object(types_mod, "_TEXT_SCAN_CHUNK", _ChunkedSentinel()):
        offset = 0
        decoded: list[str] = []
        for _ in range(len(cells)):
            text, consumed = decode_text(view[offset:])
            decoded.append(text)
            offset += consumed

    assert decoded == cells, "all short cells must decode correctly"
    assert chunked_path_entered == 0, (
        f"chunked path entered {chunked_path_entered} times for "
        "short cells in a long buffer; expected zero (one-shot probe "
        "should cover every cell)"
    )


def test_chunked_path_still_fires_for_genuinely_long_cells() -> None:
    """Regression: a cell that exceeds the one-shot probe window
    must still decode correctly via the chunked path."""
    from dqlitewire.types import _TEXT_ONE_SHOT_MAX

    long_text = "x" * (_TEXT_ONE_SHOT_MAX + 100)
    body = _build_body([long_text])

    text, consumed = decode_text(memoryview(body))
    assert text == long_text
    assert consumed == len(body)
