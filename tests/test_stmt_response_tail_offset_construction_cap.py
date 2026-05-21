"""Pin: ``StmtResponse.__post_init__`` enforces ``_MAX_TAIL_OFFSET``
at construction.

Previously the cap fired only inside ``encode_body`` (for emission) and
``decode_body`` (for ingestion). A mock-server / proxy author building
a ``StmtResponse(schema=1, tail_offset=_MAX_TAIL_OFFSET+1)`` got a clean
dataclass; the cap surfaced as ``EncodeError`` only at first encode,
far from the construction site. The sibling ``num_params`` field on the
same dataclass is already construction-capped, and
``ResultResponse.__post_init__`` explicitly cites ``StmtResponse._MAX_
TAIL_OFFSET`` as its precedent for construction-time cap discipline.

This pin closes the asymmetry: the cap now fires at construction time,
in addition to the existing encode/decode caps (kept as defense-in-
depth).
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.responses import _MAX_TAIL_OFFSET, StmtResponse


def test_stmt_response_tail_offset_construction_over_cap_rejected() -> None:
    with pytest.raises(EncodeError, match="exceeds maximum"):
        StmtResponse(
            db_id=0,
            stmt_id=0,
            num_params=0,
            tail_offset=_MAX_TAIL_OFFSET + 1,
            schema=1,
        )


def test_stmt_response_tail_offset_at_cap_accepted() -> None:
    """The cap is exclusive — exactly at the cap must be accepted."""
    msg = StmtResponse(
        db_id=0,
        stmt_id=0,
        num_params=0,
        tail_offset=_MAX_TAIL_OFFSET,
        schema=1,
    )
    assert msg.tail_offset == _MAX_TAIL_OFFSET


def test_stmt_response_tail_offset_none_unaffected() -> None:
    """``tail_offset=None`` (the V0 default) is unaffected by the cap."""
    msg = StmtResponse(db_id=0, stmt_id=0, num_params=0)
    assert msg.tail_offset is None


def test_stmt_response_negative_tail_offset_still_rejected_by_uint64_validator() -> None:
    """Order matters: ``_validate_uint64`` runs FIRST, so a negative
    value triggers the range error rather than slipping into the cap
    check (which would compare a negative against the positive cap
    and silently succeed)."""
    with pytest.raises(EncodeError, match="out of range|must be int"):
        StmtResponse(
            db_id=0,
            stmt_id=0,
            num_params=0,
            tail_offset=-1,
            schema=1,
        )


def test_stmt_response_encode_body_cap_still_defense_in_depth() -> None:
    """The encode-time cap is kept as defense-in-depth. The simplest
    way to exercise it is to bypass construction (object.__setattr__)
    on a frozen dataclass — confirming the encode-time cap still
    fires when a tail_offset is mutated past construction."""
    msg = StmtResponse(db_id=0, stmt_id=0, num_params=0, tail_offset=0, schema=1)
    # Frozen dataclasses use object.__setattr__ to bypass the freeze.
    # The encode-time cap is the second line of defense for this case.
    object.__setattr__(msg, "tail_offset", _MAX_TAIL_OFFSET + 1)
    with pytest.raises(EncodeError, match="exceeds maximum"):
        msg.encode_body()
