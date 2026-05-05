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


def test_header_and_message_exported_from_package() -> None:
    """``Message`` and ``Header`` are used in the public
    ``encode_message`` annotation, so they must be importable from the
    top-level package rather than only from ``dqlitewire.messages.base``.
    """
    import dqlitewire
    from dqlitewire.messages.base import Header, Message

    assert dqlitewire.Header is Header
    assert dqlitewire.Message is Message
    assert "Header" in dqlitewire.__all__
    assert "Message" in dqlitewire.__all__


def test_all_entries_are_resolvable_attributes() -> None:
    """Every name in ``__all__`` must resolve to an attribute on the
    package. A refactor that dropped or renamed a re-exported symbol
    while leaving its name in ``__all__`` (or vice versa) silently
    breaks ``from dqlitewire import X`` for downstream consumers
    (python-dqlite-client / python-dqlite-dbapi / sqlalchemy-dqlite
    all pull cross-package symbols from this surface)."""
    for name in dqlitewire.__all__:
        assert hasattr(dqlitewire, name), (
            f"__all__ lists {name!r} but module has no such attribute"
        )
    # Sanity: __all__ contains no duplicates.
    assert len(dqlitewire.__all__) == len(set(dqlitewire.__all__))
