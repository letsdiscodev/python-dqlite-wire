"""Each dqlitewire submodule must declare ``__all__`` so
``from dqlitewire.<sub> import *`` does not leak private helpers.

Mirrors the pattern enforced by ``dqliteclient`` (see
``test_submodule_all.py`` in that package). Without per-submodule
``__all__``, ``from dqlitewire.constants import *`` leaks ``IntEnum``,
``Final``, regex internals, etc., and breaks the discipline of
declaring the public surface at the source.

Also asserts every name listed in ``dqlitewire.__all__`` is reachable
from at least one submodule's ``__all__`` (i.e., the parent re-export
has a single source-of-truth submodule).
"""

from __future__ import annotations

import importlib

import pytest

import dqlitewire

_SUBMODULES = [
    "dqlitewire.buffer",
    "dqlitewire.codec",
    "dqlitewire.constants",
    "dqlitewire.exceptions",
    "dqlitewire.messages",
    "dqlitewire.messages.base",
    "dqlitewire.messages.requests",
    "dqlitewire.messages.responses",
    "dqlitewire.truncate",
    "dqlitewire.tuples",
    "dqlitewire.types",
]


@pytest.mark.parametrize("modname", _SUBMODULES)
def test_submodule_declares_all(modname: str) -> None:
    mod = importlib.import_module(modname)
    assert hasattr(mod, "__all__"), f"{modname} is missing __all__"
    exported = mod.__all__
    assert isinstance(exported, list | tuple), (
        f"{modname}.__all__ must be list/tuple, got {type(exported).__name__}"
    )
    for name in exported:
        assert isinstance(name, str), f"{modname}.__all__ entries must be strings; got {name!r}"
        assert hasattr(mod, name), f"{modname}.__all__ lists {name!r} but it is not defined"


def test_parent_all_is_covered_by_submodules() -> None:
    """Every name in ``dqlitewire.__all__`` is reachable from at
    least one submodule's ``__all__`` (modulo package-local names
    like ``__version__`` and the submodule re-exports themselves)."""
    submodule_names: set[str] = set()
    for modname in _SUBMODULES:
        mod = importlib.import_module(modname)
        submodule_names.update(getattr(mod, "__all__", []))

    # Names that legitimately live only at the package root.
    package_local = {
        "__version__",
        "messages",
        "tuples",
        "types",
    }

    missing: list[str] = []
    for name in dqlitewire.__all__:
        if name in package_local:
            continue
        if name not in submodule_names:
            missing.append(name)
    assert not missing, (
        f"names re-exported by dqlitewire.__all__ are not in any submodule "
        f"__all__: {sorted(missing)}"
    )
