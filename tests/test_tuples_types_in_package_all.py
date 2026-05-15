"""Pin: ``dqlitewire.tuples`` and ``dqlitewire.types`` are reachable from
the package root.

``dqlitewire.__all__`` includes ``"messages"`` so ``dqlitewire.messages``
is part of the documented public API. ``tuples`` and ``types`` are
referenced by the package docstring at ``__init__.py`` (e.g.
``encode_value(v)[1]``) and must be reachable through the package
root for the same parity — otherwise the helpers are reachable only
via ``from dqlitewire.tuples import ...`` which is fine but
undocumented as a public entry point. This module is the drift
defence against ``__all__`` losing those entries.
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
