"""Pin: ``RowsResponse`` column-name diagnostic labels are byte-identical
across the encode and decode paths.

Every other variable-length text field in the package emits an identical
``label=`` string from both ``encode_text`` and ``decode_text`` call
sites so a monitoring rule keyed on the exact diagnostic text can lift
both halves of the round-trip with one match. ``"Column name"``
(title case) on the encode side vs ``"column name"`` (lowercase) on the
decode side was the lone discipline drift; this pin makes the
asymmetry surface as a failing test if it ever returns.
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
    """Build a synthetic body that hand-codes an oversize column-name
    field — bypassing the encode-side cap so the decode-side label
    discipline can be exercised directly. The wire format for a
    ``RowsResponse`` body is: uint64 column_count, then for each
    column a NUL-terminated UTF-8 string (no length prefix) padded
    to 8-byte alignment, then the row stream terminator.

    ``decode_text`` scans for the NUL terminator; ``max_size+1``
    bytes are materialised for the scan and if no NUL is found in
    that window, the cap-exceeded error fires. A reserve for the
    end marker is appended so the column-count bounds check
    (``column_count > (remaining - WORD_SIZE) // 8``) passes.
    """
    # Oversize NUL-terminated payload — the scan will exceed max_size
    # before finding a terminator, surfacing the cap diagnostic with
    # the field label.
    payload = b"a" * name_size + b"\x00"
    pad = (-len(payload)) % 8
    text_bytes = payload + b"\x00" * pad
    # Append an 8-byte zero word as a stand-in for the end marker /
    # remaining body — its only purpose here is to pass the column-
    # count vs remaining-body bounds check at the top of decode_body.
    return encode_uint64(1) + text_bytes + b"\x00" * 8


def test_decode_side_column_name_uses_lowercase_label() -> None:
    body = _build_rows_frame_with_oversize_col_name(_MAX_COLUMN_NAME_SIZE + 1)
    with pytest.raises(DecodeError) as exc:
        RowsResponse.decode_body(body)
    assert "column name" in str(exc.value)
    assert "Column name" not in str(exc.value)


def test_encode_and_decode_labels_are_byte_identical() -> None:
    """Pin the symmetric-label discipline: the exact same lowercase
    token appears in both diagnostics."""
    oversize = "x" * (_MAX_COLUMN_NAME_SIZE + 1)
    with pytest.raises(EncodeError) as enc_exc:
        RowsResponse(column_names=[oversize], rows=[]).encode_body()

    body = _build_rows_frame_with_oversize_col_name(_MAX_COLUMN_NAME_SIZE + 1)
    with pytest.raises(DecodeError) as dec_exc:
        RowsResponse.decode_body(body)

    # Both diagnostics share the lowercase "column name" token so a
    # grep on a log line lifts both call sites.
    enc_text = str(enc_exc.value)
    dec_text = str(dec_exc.value)
    assert "column name" in enc_text
    assert "column name" in dec_text
