"""Pin ``BARE_DATABASE_ERROR_CODES``: the SSOT cross-package contract (SA
derives its slot-fatal disconnect codes from it), so pin membership and
frozenset shape against silent drift."""

from __future__ import annotations

from dqlitewire import (
    BARE_DATABASE_ERROR_CODES,
    SQLITE_CORRUPT,
    SQLITE_FORMAT,
    SQLITE_NOTADB,
)


def test_bare_database_error_codes_membership() -> None:
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
    """Pin absolute numeric codes so a future ``SQLITE_*`` alias rename can't
    silently change the contract."""
    assert frozenset({11, 24, 26}) == BARE_DATABASE_ERROR_CODES


def test_bare_database_error_codes_is_frozenset() -> None:
    """Pin frozenset shape: SA uses the constant as a dict key / in subset
    comparisons, which a regression to plain ``set`` would break."""
    assert isinstance(BARE_DATABASE_ERROR_CODES, frozenset)
