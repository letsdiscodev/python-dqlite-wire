"""StmtResponse.__post_init__ coerces via object.__setattr__ (not bare
assignment) so the V1-implicit-zero coercion survives a future frozen flip."""

from __future__ import annotations

import inspect

from dqlitewire.messages.responses import StmtResponse


def test_stmt_response_post_init_uses_object_setattr() -> None:
    src = inspect.getsource(StmtResponse.__post_init__)
    assert "object.__setattr__" in src, (
        "StmtResponse.__post_init__ must coerce via object.__setattr__ to "
        "match the AssignRequest / NodeInfo sibling pattern."
    )
    # Comments may mention the bare assignment; it must not be a real statement.
    code_lines = [line for line in src.splitlines() if not line.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    assert "self.tail_offset = 0" not in code


def test_stmt_response_v1_default_tail_offset_normalised_to_zero() -> None:
    """schema=1 with tail_offset=None normalises to tail_offset=0."""
    msg = StmtResponse(db_id=1, stmt_id=2, num_params=0, schema=1)
    assert msg.tail_offset == 0
