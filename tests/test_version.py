"""Guard against drift between pyproject.toml version and __version__."""

import pathlib
import tomllib

import dqlitewire


def test_pyproject_matches_package_version() -> None:
    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        metadata = tomllib.load(f)
    assert metadata["project"]["version"] == dqlitewire.__version__


def test_version_is_in_all() -> None:
    """Pin ``__version__`` as a public, re-exported attribute.

    PEP 396 convention: ``__version__`` is the canonical module-level
    handle for version discovery and belongs in ``__all__`` so
    ``from dqlitewire import *`` binds it and static-analysis tools
    that read ``__all__`` as the source of truth see it as public.
    """
    assert "__version__" in dqlitewire.__all__
