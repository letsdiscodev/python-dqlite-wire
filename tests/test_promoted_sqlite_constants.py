"""Pin: the 10 promoted SQLite primary codes and the 11 extended
``SQLITE_CONSTRAINT_*`` codes match stdlib ``sqlite3`` bit-for-bit
and are reachable via the wire package's top-level surface.

The 10 primaries were previously duplicated as private literals in
the dbapi cursor classifier (``_SQLITE_*`` module-locals); they now
live in ``dqlitewire.constants`` alongside the rest of the SQLite
primary set. The extended ``SQLITE_CONSTRAINT_*`` family was
previously absent — the integer values arrived at the dbapi caller
via the wire round-trip but the symbolic names were missing on both
layers, so cross-driver porting code branching on
``e.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE`` had to
``import sqlite3`` separately.
"""

from __future__ import annotations

import sqlite3

import pytest

import dqlitewire

# Promoted primary codes (formerly cursor.py-private literals).
PROMOTED_PRIMARIES = [
    "SQLITE_INTERNAL",
    "SQLITE_TOOBIG",
    "SQLITE_CONSTRAINT",
    "SQLITE_MISMATCH",
    "SQLITE_MISUSE",
    "SQLITE_NOLFS",
    "SQLITE_AUTH",
    "SQLITE_RANGE",
    "SQLITE_NOTICE",
    "SQLITE_WARNING",
]

# Extended SQLITE_CONSTRAINT_* family (high byte 1..11).
CONSTRAINT_FAMILY = [
    "SQLITE_CONSTRAINT_CHECK",
    "SQLITE_CONSTRAINT_COMMITHOOK",
    "SQLITE_CONSTRAINT_FOREIGNKEY",
    "SQLITE_CONSTRAINT_FUNCTION",
    "SQLITE_CONSTRAINT_NOTNULL",
    "SQLITE_CONSTRAINT_PRIMARYKEY",
    "SQLITE_CONSTRAINT_TRIGGER",
    "SQLITE_CONSTRAINT_UNIQUE",
    "SQLITE_CONSTRAINT_VTAB",
    "SQLITE_CONSTRAINT_ROWID",
    "SQLITE_CONSTRAINT_PINNED",
]


@pytest.mark.parametrize("name", PROMOTED_PRIMARIES)
def test_promoted_primary_matches_stdlib(name: str) -> None:
    """Each promoted primary equals stdlib's value bit-for-bit."""
    assert getattr(dqlitewire, name) == getattr(sqlite3, name), (
        f"{name} drift: dqlitewire={getattr(dqlitewire, name)} "
        f"vs stdlib sqlite3={getattr(sqlite3, name)}"
    )


@pytest.mark.parametrize("name", CONSTRAINT_FAMILY)
def test_constraint_extended_matches_stdlib(name: str) -> None:
    """Each extended SQLITE_CONSTRAINT_* equals stdlib's value."""
    assert getattr(dqlitewire, name) == getattr(sqlite3, name), (
        f"{name} drift: dqlitewire={getattr(dqlitewire, name)} "
        f"vs stdlib sqlite3={getattr(sqlite3, name)}"
    )


@pytest.mark.parametrize("name", PROMOTED_PRIMARIES + CONSTRAINT_FAMILY)
def test_constants_in_package_all(name: str) -> None:
    """Promoted constants appear in ``dqlitewire.__all__``."""
    assert name in dqlitewire.__all__


def test_constraint_extended_share_primary_byte() -> None:
    """Every extended ``SQLITE_CONSTRAINT_*`` masks to
    ``SQLITE_CONSTRAINT`` (the dbapi classifier's grouping
    invariant)."""
    for name in CONSTRAINT_FAMILY:
        value = getattr(dqlitewire, name)
        assert value & 0xFF == dqlitewire.SQLITE_CONSTRAINT, (
            f"{name}={value} primary-byte differs from "
            f"SQLITE_CONSTRAINT={dqlitewire.SQLITE_CONSTRAINT}"
        )
