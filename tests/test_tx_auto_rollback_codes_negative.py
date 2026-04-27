"""Negative pin: codes upstream emits via ``failure(req, ...)`` that
must NOT trigger the client's auto-rollback tracker clear.

The positive set (ABORT/NOMEM/INTERRUPT/IOERR/CORRUPT/FULL) is
already pinned. This file pins the negative space — adding a code
to the auto-rollback set without flipping the parametrize here will
trip the test, forcing a deliberate review.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import (
    DQLITE_NOTFOUND,
    DQLITE_PARSE,
    SQLITE_BUSY,
    SQLITE_ERROR,
    SQLITE_NOTFOUND,
    SQLITE_PROTOCOL,
    TX_AUTO_ROLLBACK_PRIMARY_CODES,
)


@pytest.mark.parametrize(
    "code",
    [
        SQLITE_ERROR,  # gateway.c emits for nonempty statement tail, etc.
        SQLITE_BUSY,  # engine-side OR Raft-side; carved out at client layer
        SQLITE_NOTFOUND,  # gateway.c LOOKUP_DB / LOOKUP_STMT
        SQLITE_PROTOCOL,  # bad format version
        19,  # SQLITE_CONSTRAINT — CHECK on plain INSERT does NOT auto-rollback
        21,  # SQLITE_MISUSE
        DQLITE_NOTFOUND,
        DQLITE_PARSE,
    ],
)
def test_code_not_in_auto_rollback_set(code: int) -> None:
    """Pin: the listed codes are NOT in TX_AUTO_ROLLBACK_PRIMARY_CODES.
    A maintainer adding a code to the auto-rollback set must
    explicitly remove it from this matrix."""
    assert code not in TX_AUTO_ROLLBACK_PRIMARY_CODES
