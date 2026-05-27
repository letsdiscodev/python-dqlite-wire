"""Pin: ``decode_row_header`` (and the zero-column fast-path in
``RowsResponse.decode_body``) detect DONE / PART markers via
direct ``memoryview[:8] == _ROW_*_MARKER`` equality, without
materialising a per-row ``bytes(data[:8])``.

The prior shape called ``bytes(data[:8])`` once per row purely to
satisfy a ``bytes == bytes`` comparison against
``_ROW_DONE_MARKER`` / ``_ROW_PART_MARKER``. ``memoryview``
already implements ``__eq__`` against bytes-like objects via the
buffer protocol — the explicit copy was redundant.

The C reference (``dqlite-upstream/src/client/protocol.c``'s
``peekUint64``) compares the 8-byte sentinel as an integer with
zero heap allocation; the Python equivalent that mirrors this
property is the direct ``memoryview[:8] == bytes_constant``.

A 10k-row response paid 10001 ``bytes()`` calls + 10001 × ~64 B
of transient allocation purely to support the comparison. This
is a pure cleanup mirroring the established
``_NIBBLE_TO_VALUETYPE`` precedent (per-cell lookup-table
replaces per-cell constructor call) in the same module.
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
    """Walk the source AST and look for any ``bytes(<expr>[...])``
    or ``bytes(<expr>[:N])`` call. The fix removes ALL such calls
    from the marker-detection arm of these two functions.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "bytes" and node.args:
            arg = node.args[0]
            # ``bytes(view[:WORD_SIZE])`` or ``bytes(data[:8])``.
            if isinstance(arg, ast.Subscript):
                return True
            # ``bytes(view[offset : offset + WORD_SIZE])``.
            if isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Slice):
                return True
    return False


def test_decode_row_header_marker_check_does_not_materialise_per_row_bytes() -> None:
    """The marker-detection arm of ``decode_row_header`` must NOT
    contain ``bytes(<slice>)`` — direct ``view[:WORD_SIZE] ==
    constant`` is the canonical shape.
    """
    src = _decode_row_header_source()
    # Restrict scan to the first ~20 lines (the marker-check arm
    # at the top of the function); we tolerate ``bytes(...)`` in
    # any subsequent code if a future refactor adds one elsewhere.
    head = "\n".join(src.splitlines()[:25])
    assert not _has_bytes_call_on_view_prefix(head), (
        "decode_row_header's marker arm allocates bytes(data[:8]) per "
        "row; replace with direct ``data[:WORD_SIZE] == _ROW_*_MARKER`` "
        "equality (memoryview supports buffer-protocol __eq__ against "
        "bytes-like constants without copying)"
    )


def test_rows_response_decode_body_zero_column_marker_does_not_materialise_bytes() -> None:
    """The zero-column fast-path in ``RowsResponse.decode_body``
    must NOT call ``bytes(view[offset : offset + WORD_SIZE])`` to
    feed the marker comparison.
    """
    src = _rows_response_decode_body_source()
    # The zero-column branch is short; scan the whole body so we
    # catch any other bytes-on-view-prefix calls too. But there is
    # ALSO a legitimate ``bytes(view[offset : offset + size])``
    # call on the FilesResponse-style content-copy path that
    # belongs to a different function. ``RowsResponse.decode_body``
    # itself should not contain a marker-shape ``bytes(view[...])``
    # call after the fix.
    #
    # Specific assertion: there must be no
    # ``marker = bytes(view[...])`` shape in the source.
    assert "marker = bytes(" not in src, (
        "RowsResponse.decode_body's zero-column marker check still uses "
        "bytes(view[offset : offset + WORD_SIZE]); replace with direct "
        "``view[offset : offset + WORD_SIZE] == _ROW_*_MARKER`` (memoryview "
        "buffer-protocol equality avoids the per-frame copy)"
    )


def test_decode_row_header_marker_detection_works_with_memoryview_input() -> None:
    """Behavioural pin: feed memoryview-backed marker bytes and
    verify both DONE and PART markers are correctly identified.
    Regression guard for the equality-comparison rewrite.
    """
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
    """Behavioural pin: bytes input still works (the prior code
    had an explicit ``if isinstance(data, memoryview)`` branch;
    the rewrite collapses both code paths into one direct
    equality)."""
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
    """Strict-marker negative pin: a torn marker like ``0xff 0x11
    0x00..`` must NOT be accepted as DONE — it falls through to
    the type-decode arm (Go reference checks only the first byte;
    Python intentionally validates all 8). The fall-through path
    either returns a non-marker type list or raises ``DecodeError``
    on the invalid nibble; both prove the torn marker was rejected.
    """
    from dqlitewire.exceptions import DecodeError
    from dqlitewire.tuples import RowMarker, decode_row_header

    # ``0xff`` followed by zero-pad — the first byte has nibble 0x0f
    # (invalid type code), so the type-decode arm raises DecodeError.
    torn = memoryview(b"\xff" + b"\x00" * 15)
    try:
        result, _ = decode_row_header(torn, column_count=1)
    except DecodeError:
        # Marker check rejected, fall-through raised on invalid nibble.
        return
    # If it didn't raise, it must NOT have returned the DONE sentinel.
    assert result is not RowMarker.DONE, (
        "torn marker incorrectly accepted as DONE under memoryview input"
    )
