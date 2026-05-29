"""Performance regression tests for decode paths.

Bodies are wrapped in ``memoryview`` at ``decode_body`` entry so per-row
slicing is O(1) (was ``data[offset:]`` tail copies, O(N²)). Timing here is
comparative (large/small ratio) to stay machine-independent.
"""

from __future__ import annotations

import time

from dqlitewire.constants import HEADER_SIZE, ValueType
from dqlitewire.messages.responses import RowsResponse


def _build_rows(n: int) -> bytes:
    msg = RowsResponse(
        column_names=["a"],
        column_types=[ValueType.INTEGER],
        row_types=[[ValueType.INTEGER]] * n,
        rows=[[i] for i in range(n)],
    )
    return msg.encode()[HEADER_SIZE:]


def _time_decode(body: bytes, max_rows: int) -> float:
    # max_rows cap triggers at ``>=``, so pass a value above the row count.
    start = time.perf_counter()
    RowsResponse.decode_body(body, max_rows=max_rows + 1)
    return time.perf_counter() - start


class TestRowsDecodePerformance:
    def test_rows_decode_scales_linearly(self) -> None:
        """228: decoding 10x rows should take ~10x time, not ~100x.

        30x ratio allowed for CI noise while still catching O(N²) regression.
        """
        small = _build_rows(5_000)
        large = _build_rows(50_000)

        # Warm up lazy imports so first-call overhead doesn't dominate.
        _time_decode(small, max_rows=5_000)

        t_small = _time_decode(small, max_rows=5_000)
        t_large = _time_decode(large, max_rows=50_000)

        # Guard against tiny denominators on very fast machines.
        t_small = max(t_small, 1e-4)
        ratio = t_large / t_small

        assert ratio < 30, (
            f"decode ratio for 10x rows is {ratio:.1f}x "
            f"(t_small={t_small * 1000:.1f}ms, "
            f"t_large={t_large * 1000:.1f}ms); expected linear "
            f"scaling (<10x ideally, <30x as a safety margin). "
            f"Quadratic regression likely."
        )

    def test_rows_decode_many_rows_completes_quickly(self) -> None:
        """228: sanity — decoding 20k rows must finish under 2 seconds."""
        body = _build_rows(20_000)
        elapsed = _time_decode(body, max_rows=20_000)
        assert elapsed < 2.0, (
            f"decoding 20k rows took {elapsed:.2f}s; expected < 0.5s "
            f"on a linear decoder. Quadratic regression likely."
        )


class TestDecodeTextLongValues:
    """``decode_text`` must support arbitrarily long memoryview text values
    (a prior fix capped the scan window at 4 KiB, breaking long TEXT cells).
    """

    def test_decode_text_memoryview_8kib_roundtrips(self) -> None:
        from dqlitewire.types import decode_text, encode_text

        long_text = "x" * 8192
        encoded = encode_text(long_text)
        text, consumed = decode_text(memoryview(encoded))
        assert text == long_text
        assert consumed == len(encoded)

    def test_decode_text_memoryview_1mib_roundtrips(self) -> None:
        from dqlitewire.types import decode_text, encode_text

        long_text = "a" * (1024 * 1024)
        encoded = encode_text(long_text)
        start = time.perf_counter()
        text, consumed = decode_text(memoryview(encoded))
        elapsed = time.perf_counter() - start
        assert text == long_text
        assert consumed == len(encoded)
        assert elapsed < 0.5, (
            f"decoding 1 MiB memoryview text took {elapsed:.2f}s; "
            f"chunked scan should complete well under 0.5s"
        )

    def test_rows_response_with_long_text_values_roundtrips(self) -> None:
        """Full RowsResponse with 8 KiB TEXT cells decodes (prior 4 KiB cap broke this)."""
        long_text = "y" * 8192
        msg = RowsResponse(
            column_names=["payload"],
            column_types=[ValueType.TEXT],
            row_types=[[ValueType.TEXT]] * 5,
            rows=[[long_text] for _ in range(5)],
        )
        encoded = msg.encode()[HEADER_SIZE:]
        decoded = RowsResponse.decode_body(encoded)
        assert len(decoded.rows) == 5
        for row in decoded.rows:
            assert row[0] == long_text
