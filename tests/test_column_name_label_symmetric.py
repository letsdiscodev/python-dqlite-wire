"""RowsResponse column-name diagnostic labels must be byte-identical (lowercase
"column name") across the encode and decode paths, so one monitoring match lifts
both halves of the round-trip.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.messages.responses import _MAX_COLUMN_NAME_SIZE, RowsResponse
from dqlitewire.types import encode_uint64


def test_encode_side_column_name_uses_lowercase_label() -> None:
    oversize = "x" * (_MAX_COLUMN_NAME_SIZE + 1)
    with pytest.raises(EncodeError) as exc:
        RowsResponse(column_names=[oversize], rows=[]).encode_body()
    # Lowercase only — the title-case form must not slip back in.
    assert "column name" in str(exc.value)
    assert "Column name" not in str(exc.value)


def _build_rows_frame_with_oversize_col_name(name_size: int) -> bytes:
    """Hand-code an oversize column-name body (bypassing the encode-side cap) to drive
    the decode-side label. Body: uint64 column_count, then per column a padded
    NUL-terminated UTF-8 string, then the row terminator."""
    # decode_text scans for the NUL; with no terminator in max_size+1 bytes the
    # cap-exceeded diagnostic fires with the field label.
    payload = b"a" * name_size + b"\x00"
    pad = (-len(payload)) % 8
    text_bytes = payload + b"\x00" * pad
    # Trailing zero word passes the column-count vs remaining-body bounds check.
    return encode_uint64(1) + text_bytes + b"\x00" * 8


def test_decode_side_column_name_uses_lowercase_label() -> None:
    body = _build_rows_frame_with_oversize_col_name(_MAX_COLUMN_NAME_SIZE + 1)
    with pytest.raises(DecodeError) as exc:
        RowsResponse.decode_body(body)
    assert "column name" in str(exc.value)
    assert "Column name" not in str(exc.value)


def test_encode_and_decode_labels_are_byte_identical() -> None:
    """The same lowercase token appears in both the encode and decode diagnostics."""
    oversize = "x" * (_MAX_COLUMN_NAME_SIZE + 1)
    with pytest.raises(EncodeError) as enc_exc:
        RowsResponse(column_names=[oversize], rows=[]).encode_body()

    body = _build_rows_frame_with_oversize_col_name(_MAX_COLUMN_NAME_SIZE + 1)
    with pytest.raises(DecodeError) as dec_exc:
        RowsResponse.decode_body(body)

    enc_text = str(enc_exc.value)
    dec_text = str(dec_exc.value)
    assert "column name" in enc_text
    assert "column name" in dec_text
