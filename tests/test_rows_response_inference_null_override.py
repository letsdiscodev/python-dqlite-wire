"""Pin: ``RowsResponse._get_row_types`` applies the None→NULL override
uniformly across all three type-selection paths (``row_types`` set,
``column_types`` set, or inference from values).

The None→NULL override is critical because Go's per-row type header
emits the NULL nibble when the value is None regardless of the declared
column type. The override used to be skipped on the inference branch,
relying on ``_infer_value_type(None) == ValueType.NULL`` to produce
the right answer accidentally. A future helper that learned a
typed-null shape would silently regress the inference path.
"""

from __future__ import annotations

from dqlitewire.constants import ValueType
from dqlitewire.messages.responses import RowsResponse
from dqlitewire.types import WireValue


def test_none_in_row_with_row_types_set_overrides_to_null() -> None:
    row: list[WireValue] = [42, None]
    msg = RowsResponse(
        column_names=["a", "b"],
        row_types=[[ValueType.INTEGER, ValueType.TEXT]],
        rows=[row],
    )
    types = msg._get_row_types(0, row)
    assert types == [ValueType.INTEGER, ValueType.NULL]


def test_none_in_row_with_column_types_set_overrides_to_null() -> None:
    row: list[WireValue] = [42, None]
    msg = RowsResponse(
        column_names=["a", "b"],
        column_types=[ValueType.INTEGER, ValueType.TEXT],
        rows=[row],
    )
    types = msg._get_row_types(0, row)
    assert types == [ValueType.INTEGER, ValueType.NULL]


def test_none_in_row_with_inference_falls_through_to_null() -> None:
    """Inference branch: no ``column_types`` / ``row_types`` declared.
    The None→NULL override applies uniformly with the other branches
    (was previously implicit via ``_infer_value_type(None) == NULL``;
    now applied at the same site for defensive symmetry)."""
    row: list[WireValue] = [42, None]
    msg = RowsResponse(column_names=["a", "b"], rows=[row])
    types = msg._get_row_types(0, row)
    assert types == [ValueType.INTEGER, ValueType.NULL]


def test_no_none_values_does_not_change_types() -> None:
    """Negative pin: the override loop must NOT touch non-None cells."""
    row: list[WireValue] = [42, "x"]
    msg = RowsResponse(
        column_names=["a", "b"],
        column_types=[ValueType.INTEGER, ValueType.TEXT],
        rows=[row],
    )
    types = msg._get_row_types(0, row)
    assert types == [ValueType.INTEGER, ValueType.TEXT]
