"""Pin: ROWS continuation frames must have the same column_names
tuple as the initial frame.

The user-facing ``column_names`` accessor reports the initial frame's
names only (the protocol-layer drain merges only ``rows``). A buggy /
hostile peer that holds ``column_count`` constant but rotates the
names list per continuation frame would silently mis-attribute the
per-row data — e.g. emit ``["id","name"]`` initially and
``["name","id"]`` on a continuation, with the rest of the rows now
silently labelled wrong.

Sibling-shaped to the column-count drift check; same strict-decode
posture established by the BOOLEAN narrowing, NULL-zero distinction,
and ``Header.reserved == 0`` policies.
"""

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
    """A continuation frame with a permuted (same-count) column-names
    tuple must raise DecodeError."""
    decoder = MessageDecoder(is_request=False)
    # Initial frame: 2 columns ["id", "name"]
    initial_bytes = _encoded_rows(["id", "name"], has_more=True)
    decoder.feed(initial_bytes)
    initial = decoder.decode()
    assert isinstance(initial, RowsResponse)
    assert decoder._continuation_column_names == ("id", "name")

    # Continuation frame: same count, but names permuted.
    cont_bytes = _encoded_rows(["name", "id"], has_more=False)
    decoder.feed(cont_bytes)
    with pytest.raises(DecodeError, match="column name drift"):
        decoder.decode_continuation()


def test_continuation_column_names_renamed_raises() -> None:
    """A continuation frame whose names tuple has the same count but
    different strings (a server bug or hostile relabel) must raise."""
    decoder = MessageDecoder(is_request=False)
    initial_bytes = _encoded_rows(["a", "b"], has_more=True)
    decoder.feed(initial_bytes)
    decoder.decode()

    cont_bytes = _encoded_rows(["x", "y"], has_more=False)
    decoder.feed(cont_bytes)
    with pytest.raises(DecodeError, match="column name drift"):
        decoder.decode_continuation()


def test_continuation_column_names_match_passes() -> None:
    """A continuation frame with identical names is accepted; the
    snapshot is cleared on the final frame."""
    decoder = MessageDecoder(is_request=False)
    initial_bytes = _encoded_rows(["a", "b", "c"], has_more=True)
    decoder.feed(initial_bytes)
    decoder.decode()

    cont_bytes = _encoded_rows(["a", "b", "c"], has_more=False)
    decoder.feed(cont_bytes)
    cont = decoder.decode_continuation()
    assert cont is not None
    assert decoder._continuation_column_names is None
