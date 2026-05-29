"""``decode_row_header`` validates all 8 marker bytes (like the upstream C
sentinel), not just the first byte (Go). Its docstring must say so, so a
contributor doesn't relax the check and accept torn markers like ``0xff 0x00...``.
"""

from __future__ import annotations

from dqlitewire.exceptions import DecodeError
from dqlitewire.tuples import decode_row_header


def test_decode_row_header_torn_marker_rejected_not_silently_consumed() -> None:
    """A torn marker (first byte 0xFF, trailing bytes diverge) must NOT be
    treated as DONE — it falls through to the type-header decode path."""
    # Go's first-byte check would accept this as DONE; we must not.
    torn = b"\xff\x00\x00\x00\x00\x00\x00\x00"
    try:
        result = decode_row_header(torn, column_count=1)
    except DecodeError:
        # Narrow catch so a refactor-introduced non-decode error propagates.
        return
    types_or_marker, _ = result
    from dqlitewire.tuples import RowMarker

    assert types_or_marker is not RowMarker.DONE, (
        "Torn marker (first byte 0xFF, rest zeros) must not be accepted as DONE"
    )
