"""StmtResponse.__post_init__ enforces _MAX_TAIL_OFFSET at construction,
in addition to the encode/decode caps (kept as defense-in-depth)."""

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
    """The cap is exclusive: exactly at the cap is accepted."""
    msg = StmtResponse(
        db_id=0,
        stmt_id=0,
        num_params=0,
        tail_offset=_MAX_TAIL_OFFSET,
        schema=1,
    )
    assert msg.tail_offset == _MAX_TAIL_OFFSET


def test_stmt_response_tail_offset_none_unaffected() -> None:
    """tail_offset=None (the V0 default) is unaffected by the cap."""
    msg = StmtResponse(db_id=0, stmt_id=0, num_params=0)
    assert msg.tail_offset is None


def test_stmt_response_negative_tail_offset_still_rejected_by_uint64_validator() -> None:
    """_validate_uint64 must run before the cap check, else a negative value
    would compare under the positive cap and silently succeed."""
    with pytest.raises(EncodeError, match="out of range|must be int"):
        StmtResponse(
            db_id=0,
            stmt_id=0,
            num_params=0,
            tail_offset=-1,
            schema=1,
        )


def test_stmt_response_encode_body_cap_still_defense_in_depth() -> None:
    """encode-time cap still fires when tail_offset is mutated past construction."""
    msg = StmtResponse(db_id=0, stmt_id=0, num_params=0, tail_offset=0, schema=1)
    object.__setattr__(msg, "tail_offset", _MAX_TAIL_OFFSET + 1)
    with pytest.raises(EncodeError, match="exceeds maximum"):
        msg.encode_body()
