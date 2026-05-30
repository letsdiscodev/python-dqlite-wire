"""StmtResponse.__post_init__ coerces a V1-implicit tail_offset to zero."""

from __future__ import annotations

from dqlitewire.messages.responses import StmtResponse


def test_stmt_response_v1_default_tail_offset_normalised_to_zero() -> None:
    """schema=1 with tail_offset=None normalises to tail_offset=0."""
    msg = StmtResponse(db_id=1, stmt_id=2, num_params=0, schema=1)
    assert msg.tail_offset == 0
