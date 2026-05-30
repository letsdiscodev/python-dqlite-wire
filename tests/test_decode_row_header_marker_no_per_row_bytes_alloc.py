"""``decode_row_header`` and the zero-column fast-path in
``RowsResponse.decode_body`` detect DONE / PART markers correctly from both
memoryview- and bytes-backed input, and reject torn markers.
"""

from __future__ import annotations


def test_decode_row_header_marker_detection_works_with_memoryview_input() -> None:
    """DONE and PART markers are identified from memoryview-backed bytes."""
    from dqlitewire.constants import ROW_DONE_BYTE, ROW_PART_BYTE
    from dqlitewire.tuples import RowMarker, decode_row_header

    done_buf = memoryview(bytes([ROW_DONE_BYTE]) * 8 + b"\x00" * 8)
    part_buf = memoryview(bytes([ROW_PART_BYTE]) * 8 + b"\x00" * 8)

    result, consumed = decode_row_header(done_buf, column_count=1)
    assert result is RowMarker.DONE
    assert consumed == 8

    result, consumed = decode_row_header(part_buf, column_count=1)
    assert result is RowMarker.PART
    assert consumed == 8


def test_decode_row_header_marker_detection_works_with_bytes_input() -> None:
    """bytes input still works (the rewrite collapses both paths into one
    direct equality)."""
    from dqlitewire.constants import ROW_DONE_BYTE, ROW_PART_BYTE
    from dqlitewire.tuples import RowMarker, decode_row_header

    done_buf = bytes([ROW_DONE_BYTE]) * 8
    part_buf = bytes([ROW_PART_BYTE]) * 8

    result, consumed = decode_row_header(done_buf, column_count=1)
    assert result is RowMarker.DONE
    assert consumed == 8

    result, consumed = decode_row_header(part_buf, column_count=1)
    assert result is RowMarker.PART
    assert consumed == 8


def test_decode_row_header_rejects_torn_marker_under_memoryview_input() -> None:
    """A torn marker must NOT be accepted as DONE — it falls through to the
    type-decode arm (Go checks only the first byte; we validate all 8)."""
    from dqlitewire.exceptions import DecodeError
    from dqlitewire.tuples import RowMarker, decode_row_header

    # First byte's low nibble 0x0f is an invalid type code -> DecodeError.
    torn = memoryview(b"\xff" + b"\x00" * 15)
    try:
        result, _ = decode_row_header(torn, column_count=1)
    except DecodeError:
        return
    assert result is not RowMarker.DONE, (
        "torn marker incorrectly accepted as DONE under memoryview input"
    )
