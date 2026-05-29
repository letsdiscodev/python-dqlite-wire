"""Every dqlitewire submodule declares __all__ (so `import *` leaks no private
helpers), and every name in dqlitewire.__all__ is reachable from some submodule's."""

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
