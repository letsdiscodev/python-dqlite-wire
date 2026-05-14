"""Pin: ``BARE_DATABASE_ERROR_CODES`` is the SSOT for the slot-fatal
SQLite primary codes the dqlite server actually emits as bare
``DatabaseError`` per PEP 249 §9 routing.

The set is consumed across packages as a contract: SA's
``sqlalchemydqlite.base._BARE_DBE_DISCONNECT_CODES`` derives from it
directly to decide which conn slots are pre-ping slot-fatal vs
recoverable. Adding or removing a code is a deliberate cross-package
behaviour change; pin both the membership and the ``frozenset`` shape
so silent drift surfaces at the wire layer before propagating to SA.
"""

from __future__ import annotations

from dqlitewire import (
    BARE_DATABASE_ERROR_CODES,
    SQLITE_CORRUPT,
    SQLITE_FORMAT,
    SQLITE_NOTADB,
)


def test_bare_database_error_codes_membership() -> None:
    """SSOT pin: the slot-fatal set is exactly the three codes
    dqlite-server actually emits as bare ``DatabaseError`` per PEP 249
    §9 routing. Adding / removing a code is a deliberate cross-package
    behaviour change — update both the constant and this pin together.
    """
    assert (
        frozenset(
            {
                SQLITE_CORRUPT,  # 11
                SQLITE_FORMAT,  # 24
                SQLITE_NOTADB,  # 26
            }
        )
        == BARE_DATABASE_ERROR_CODES
    )


def test_bare_database_error_codes_numeric_values() -> None:
    """SSOT pin on the absolute numeric codes so a future rename of any
    of the three ``SQLITE_*`` aliases would not silently change the
    cross-package contract."""
    assert frozenset({11, 24, 26}) == BARE_DATABASE_ERROR_CODES


def test_bare_database_error_codes_is_frozenset() -> None:
    """Pin the type so consumers can rely on ``frozenset`` semantics
    (hashable, immutable, subset-comparable). A future regression to
    plain ``set`` would break consumers that include the constant in
    their own frozensets or use it as a dict key. SA's
    ``_BARE_DBE_DISCONNECT_CODES <= some_set`` arms depend on the
    frozenset shape."""
    assert isinstance(BARE_DATABASE_ERROR_CODES, frozenset)
