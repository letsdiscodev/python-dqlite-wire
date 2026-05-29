"""Pin: ROWS continuation frames must keep the initial frame's column_names
tuple. Only the initial names are reported, so a peer that holds
column_count constant but rotates names would silently mislabel per-row
data; the decoder raises DecodeError."""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder, MessageEncoder
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import RowsResponse


def _encoded_rows(column_names: list[str], has_more: bool) -> bytes:
    rows = RowsResponse(
        column_names=column_names,
        rows=[list(range(len(column_names)))],
        has_more=has_more,
    )
    return MessageEncoder().encode(rows)


def test_continuation_column_names_drift_raises() -> None:
    decoder = MessageDecoder(is_request=False)
    initial_bytes = _encoded_rows(["id", "name"], has_more=True)
    decoder.feed(initial_bytes)
    initial = decoder.decode()
    assert isinstance(initial, RowsResponse)
    assert decoder._continuation_column_names == ("id", "name")

    cont_bytes = _encoded_rows(["name", "id"], has_more=False)
    decoder.feed(cont_bytes)
    with pytest.raises(DecodeError, match="column name drift"):
        decoder.decode_continuation()


def test_continuation_column_names_renamed_raises() -> None:
    """Same-count but renamed columns (server bug or hostile relabel) must
    raise."""
    decoder = MessageDecoder(is_request=False)
    initial_bytes = _encoded_rows(["a", "b"], has_more=True)
    decoder.feed(initial_bytes)
    decoder.decode()

    cont_bytes = _encoded_rows(["x", "y"], has_more=False)
    decoder.feed(cont_bytes)
    with pytest.raises(DecodeError, match="column name drift"):
        decoder.decode_continuation()


def test_continuation_column_names_match_passes() -> None:
    """Identical names accepted; snapshot cleared on the final frame."""
    decoder = MessageDecoder(is_request=False)
    initial_bytes = _encoded_rows(["a", "b", "c"], has_more=True)
    decoder.feed(initial_bytes)
    decoder.decode()

    cont_bytes = _encoded_rows(["a", "b", "c"], has_more=False)
    decoder.feed(cont_bytes)
    cont = decoder.decode_continuation()
    assert cont is not None
    assert decoder._continuation_column_names is None
