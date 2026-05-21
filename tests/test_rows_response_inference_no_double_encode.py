"""Pin: ``RowsResponse.encode_body``'s inference fallback path runs
``encode_value`` once per cell, not twice.

Previously ``_get_row_types``'s inference branch called
``encode_value(v)[1]`` per cell — running the full encode pipeline
(length-cap checks, BLOB materialisation, UTF-8 encode) just to extract
the type tag from the returned tuple — and then ``encode_body`` ran
``encode_value(v, vtype)`` again to consume the bytes. For BLOB-heavy
frames the cost was 2 * N * M ``encode_value`` calls instead of N * M;
each large BLOB cell was materialised twice. The fix introduces
``_infer_value_type`` (type-only ladder, no encode work) and uses it
on the inference path.

The pin counts ``encode_value`` invocations on the inference path and
asserts it equals one per cell, not two.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from dqlitewire.constants import ValueType
from dqlitewire.messages.responses import RowsResponse
from dqlitewire.types import WireValue


def test_inference_path_runs_encode_value_once_per_cell() -> None:
    """For a 3-row x 2-col inference frame, ``encode_value`` runs 6
    times total (one per cell), not 12 (the pre-fix double-encode
    shape). The type-only helper handles the inference pass; the
    actual emission still calls ``encode_value`` once per cell via
    ``encode_row_values`` in ``tuples.py``."""
    rows: list[list[WireValue]] = [[b"a", "x"], [b"b", "y"], [b"c", "z"]]
    msg = RowsResponse(column_names=["blob", "text"], rows=rows)

    from dqlitewire.types import encode_value as real_encode_value

    call_count = 0

    def counting_encode_value(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return real_encode_value(*args, **kwargs)

    # Patch the emission-side call site (in tuples.py); the inference
    # path no longer calls encode_value at all, so the count reflects
    # only the emission pass.
    with patch("dqlitewire.tuples.encode_value", side_effect=counting_encode_value):
        msg.encode_body()

    # 3 rows * 2 cols = 6 cells. Pre-fix: the inference pass also
    # called encode_value, doubling the count to 12. Post-fix: only
    # the emission pass remains.
    assert call_count == 6, f"expected 6 encode_value calls, got {call_count}"


def test_inference_path_round_trip_unchanged() -> None:
    """The fix changes the call count, not the wire bytes. Verify
    inference still picks the right types and the round-trip succeeds."""
    rows: list[list[WireValue]] = [[b"a"], [1], ["text"], [1.5], [True], [None]]
    msg = RowsResponse(column_names=["c"], rows=rows)
    encoded = msg.encode_body()
    decoded = RowsResponse.decode_body(encoded)
    # Inference picks BLOB for bytes, INTEGER for int (not BOOLEAN —
    # the bool is checked first per the helper's ladder), etc. The
    # decoded values reflect what the inference picked.
    assert decoded.rows[0] == [b"a"]
    assert decoded.rows[1] == [1]
    assert decoded.rows[2] == ["text"]
    assert decoded.rows[3] == [1.5]
    # bool infers to BOOLEAN, which round-trips as int 1 (SQLite stores
    # both as integer column value regardless of the tag) — documented
    # in encode_value's docstring.
    assert decoded.rows[4] == [1]
    assert decoded.rows[5] == [None]


def test_infer_value_type_helper_directly() -> None:
    """Direct unit coverage for the helper."""
    from dqlitewire.types import _infer_value_type

    assert _infer_value_type(None) == ValueType.NULL
    assert _infer_value_type(True) == ValueType.BOOLEAN
    assert _infer_value_type(False) == ValueType.BOOLEAN
    assert _infer_value_type(42) == ValueType.INTEGER
    assert _infer_value_type(1.5) == ValueType.FLOAT
    assert _infer_value_type("text") == ValueType.TEXT
    assert _infer_value_type(b"bytes") == ValueType.BLOB
    assert _infer_value_type(bytearray(b"ba")) == ValueType.BLOB
    assert _infer_value_type(memoryview(b"mv")) == ValueType.BLOB
