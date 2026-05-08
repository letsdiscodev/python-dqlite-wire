"""Pin: ``dqlitewire.tuples`` and ``dqlitewire.types`` are reachable from
the package root.

ISSUE-1450 (`done/`) extended ``dqlitewire.__all__`` to include
``"messages"``. The same parallel was missing for ``tuples`` and
``types`` despite ``encode_value(v)[1]`` being referenced by the
package docstring at ``__init__.py:54``. Without the entries the
helpers are reachable only via ``from dqlitewire.tuples import ...``
which is fine but undocumented as a public entry point.
"""

from __future__ import annotations

import dqlitewire


def test_tuples_submodule_reachable_via_package_root() -> None:
    from dqlitewire import tuples

    assert tuples.encode_params_tuple is not None
    assert "tuples" in dqlitewire.__all__


def test_types_submodule_reachable_via_package_root() -> None:
    from dqlitewire import types

    assert types.encode_value is not None
    assert "types" in dqlitewire.__all__
