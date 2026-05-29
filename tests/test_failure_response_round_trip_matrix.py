"""End-to-end round-trip matrix for FailureResponse: extended/boundary/
dqlite-namespace codes against empty/short/common messages."""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder
from dqlitewire.constants import (
    DQLITE_NOTFOUND,
    DQLITE_PARSE,
    SQLITE_IOERR_LEADERSHIP_LOST,
    SQLITE_IOERR_NOT_LEADER,
)
from dqlitewire.messages.responses import FailureResponse


@pytest.mark.parametrize(
    ("code", "message"),
    [
        # code=0 is genuinely emitted upstream (gateway.c:372 / :890).
        (0, "empty statement"),
        (1, ""),
        (1, "constraint violated"),
        (5, "checkpoint in progress"),
        (SQLITE_IOERR_NOT_LEADER, ""),
        (SQLITE_IOERR_NOT_LEADER, "not leader"),
        (SQLITE_IOERR_LEADERSHIP_LOST, ""),
        (SQLITE_IOERR_LEADERSHIP_LOST, "lost leadership mid-transaction"),
        (DQLITE_NOTFOUND, "no database opened"),
        (DQLITE_PARSE, "unknown request type"),
        # uint64 boundary.
        (2**63 - 1, ""),
        (2**63, ""),
        (2**64 - 1, "max code"),
    ],
)
def test_failure_response_round_trip(code: int, message: str) -> None:
    original = FailureResponse(code=code, message=message)
    encoded = original.encode()
    decoder = MessageDecoder(is_request=False)
    decoder.feed(encoded)
    decoded = decoder.decode()
    assert isinstance(decoded, FailureResponse)
    assert decoded.code == code
    assert decoded.message == message
    assert not decoder.is_poisoned
