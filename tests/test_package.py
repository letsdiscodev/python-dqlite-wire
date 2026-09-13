"""Package-level hygiene pins: re-exports, logger handler, no __future__ annotations in src."""

from __future__ import annotations

import logging
import pathlib

import dqlitewire
from dqlitewire import constants

# ---- merged from test_no_future_annotations_in_src.py ----
# Guard: no src/dqlitewire file carries ``from __future__ import annotations`` (3.13 floor).


def test_no_future_annotations_in_dqlitewire_src() -> None:
    root = pathlib.Path(dqlitewire.__file__).parent
    offenders: list[str] = []
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "from __future__ import annotations" in text:
            offenders.append(str(py.relative_to(root)))
    assert not offenders, (
        f"unexpected `from __future__ import annotations` imports in dqlitewire/src/: {offenders}"
    )


# ---- merged from test_top_level_logger_has_null_handler.py ----
# The top-level package logger has a logging.NullHandler attached at import,
# per the Python logging HOWTO convention for libraries.


def test_top_level_logger_has_null_handler() -> None:
    logger = logging.getLogger("dqlitewire")
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers), (
        "library top-level logger must have a NullHandler attached per "
        "Python logging HOWTO convention (every well-behaved library — "
        "psycopg, aiosqlite, asyncpg, urllib3 — does this)"
    )


# ---- merged from test_tuples_types_in_package_all.py ----
# Pin: ``dqlitewire.tuples`` and ``dqlitewire.types`` are reachable from
# the package root.
#
# ``dqlitewire.__all__`` includes ``"messages"`` so ``dqlitewire.messages``
# is part of the documented public API. ``tuples`` and ``types`` are
# referenced by the package docstring at ``__init__.py`` (e.g.
# ``encode_value(v)[1]``) and must be reachable through the package
# root for the same parity — otherwise the helpers are reachable only
# via ``from dqlitewire.tuples import ...`` which is fine but
# undocumented as a public entry point. This module is the drift
# defence against ``__all__`` losing those entries.


def test_tuples_submodule_reachable_via_package_root() -> None:
    from dqlitewire import tuples

    assert tuples.encode_params_tuple is not None
    assert "tuples" in dqlitewire.__all__


def test_types_submodule_reachable_via_package_root() -> None:
    from dqlitewire import types

    assert types.encode_value is not None
    assert "types" in dqlitewire.__all__


# ---- merged from test_constants_promoted_to_root.py ----
# Pin: every name in ``dqlitewire.constants.__all__`` is re-exported from
# the package root, guarding against drift if a constant is added to the
# subpackage list without a matching root re-export.


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
