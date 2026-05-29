"""Pin: every name in ``dqlitewire.constants.__all__`` is re-exported from
the package root, guarding against drift if a constant is added to the
subpackage list without a matching root re-export."""

from __future__ import annotations

import dqlitewire
from dqlitewire import constants


def test_constants_all_promoted_to_root_all() -> None:
    missing = sorted(set(constants.__all__) - set(dqlitewire.__all__))
    assert not missing, f"constants.__all__ items missing from dqlitewire.__all__: {missing}"


def test_constants_all_attribute_reachable_at_root() -> None:
    missing = [name for name in constants.__all__ if not hasattr(dqlitewire, name)]
    assert not missing, (
        f"constants.__all__ items not attribute-reachable on dqlitewire: {sorted(missing)}"
    )


def test_header_size_and_word_size_reachable_at_root() -> None:
    assert dqlitewire.HEADER_SIZE == constants.HEADER_SIZE
    assert dqlitewire.WORD_SIZE == constants.WORD_SIZE


def test_row_marker_reachable_at_root() -> None:
    """``RowMarker`` (from ``tuples.py``) is also reachable from the root."""
    from dqlitewire import tuples

    assert dqlitewire.RowMarker is tuples.RowMarker
