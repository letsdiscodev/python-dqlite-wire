"""LEADER_ERROR_CODES has modern (40/41) and legacy (32/33) sub-codes; old flips stay retryable."""

from __future__ import annotations

from dqlitewire import LEADER_ERROR_CODES
from dqlitewire.constants import (
    SQLITE_IOERR,
    SQLITE_IOERR_LEADERSHIP_LOST,
    SQLITE_IOERR_LEADERSHIP_LOST_LEGACY,
    SQLITE_IOERR_NOT_LEADER,
    SQLITE_IOERR_NOT_LEADER_LEGACY,
)


def test_modern_codes_are_in_set() -> None:
    assert SQLITE_IOERR_NOT_LEADER == 10250
    assert SQLITE_IOERR_LEADERSHIP_LOST == 10506
    assert SQLITE_IOERR_NOT_LEADER in LEADER_ERROR_CODES
    assert SQLITE_IOERR_LEADERSHIP_LOST in LEADER_ERROR_CODES


def test_legacy_codes_are_in_set() -> None:
    assert SQLITE_IOERR_NOT_LEADER_LEGACY == 8202
    assert SQLITE_IOERR_LEADERSHIP_LOST_LEGACY == 8458
    assert SQLITE_IOERR_NOT_LEADER_LEGACY in LEADER_ERROR_CODES
    assert SQLITE_IOERR_LEADERSHIP_LOST_LEGACY in LEADER_ERROR_CODES


def test_set_contains_exactly_the_four_codes() -> None:
    """Cardinality pin so a future revert that drops the legacy pair
    fails deterministically."""
    assert len(LEADER_ERROR_CODES) == 4
    assert SQLITE_IOERR_NOT_LEADER in LEADER_ERROR_CODES
    assert SQLITE_IOERR_LEADERSHIP_LOST in LEADER_ERROR_CODES
    assert SQLITE_IOERR_NOT_LEADER_LEGACY in LEADER_ERROR_CODES
    assert SQLITE_IOERR_LEADERSHIP_LOST_LEGACY in LEADER_ERROR_CODES


def test_legacy_codes_compute_from_byte_shift() -> None:
    """Tie the constants to the byte-shift formula so a future
    refactor that re-derives the values can be sanity-checked."""
    assert SQLITE_IOERR_NOT_LEADER_LEGACY == SQLITE_IOERR | (32 << 8)
    assert SQLITE_IOERR_LEADERSHIP_LOST_LEGACY == SQLITE_IOERR | (33 << 8)
