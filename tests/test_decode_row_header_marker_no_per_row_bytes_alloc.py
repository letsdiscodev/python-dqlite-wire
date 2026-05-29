"""``decode_row_header`` and the zero-column fast-path in
``RowsResponse.decode_body`` detect DONE / PART markers via direct
``memoryview[:8] == _ROW_*_MARKER`` equality (buffer-protocol __eq__),
without a per-row ``bytes(data[:8])`` copy.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from dqlitewire import tuples as tuples_mod
from dqlitewire.messages import responses as responses_mod


def _decode_row_header_source() -> str:
    return textwrap.dedent(inspect.getsource(tuples_mod.decode_row_header))


def _rows_response_decode_body_source() -> str:
    return textwrap.dedent(inspect.getsource(responses_mod.RowsResponse.decode_body))


def _has_bytes_call_on_view_prefix(src: str) -> bool:
    """True if the source contains any ``bytes(<expr>[...])`` call."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "bytes" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Subscript):
                return True
            if isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Slice):
                return True
    return False


def test_decode_row_header_marker_check_does_not_materialise_per_row_bytes() -> None:
    """The marker-detection arm of ``decode_row_header`` must NOT contain
    ``bytes(<slice>)``."""
    src = _decode_row_header_source()
    # Scan only the marker-check arm at the top; tolerate bytes(...) later.
    head = "\n".join(src.splitlines()[:25])
    assert not _has_bytes_call_on_view_prefix(head), (
        "decode_row_header's marker arm allocates bytes(data[:8]) per "
        "row; replace with direct ``data[:WORD_SIZE] == _ROW_*_MARKER`` "
        "equality (memoryview supports buffer-protocol __eq__ against "
        "bytes-like constants without copying)"
    )


def test_rows_response_decode_body_zero_column_marker_does_not_materialise_bytes() -> None:
    """The zero-column fast-path in ``RowsResponse.decode_body`` must NOT
    call ``bytes(view[...])`` to feed the marker comparison."""
    src = _rows_response_decode_body_source()
    assert "marker = bytes(" not in src, (
        "RowsResponse.decode_body's zero-column marker check still uses "
        "bytes(view[offset : offset + WORD_SIZE]); replace with direct "
        "``view[offset : offset + WORD_SIZE] == _ROW_*_MARKER`` (memoryview "
        "buffer-protocol equality avoids the per-frame copy)"
    )


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
