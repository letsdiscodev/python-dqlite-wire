"""Pin: ``StmtResponse.__post_init__`` uses ``object.__setattr__``
for the V1-implicit-zero coercion, matching the
``AssignRequest`` / ``NodeInfo`` sibling pattern.

A bare ``self.tail_offset = 0`` works today but breaks under a
future ``@dataclass(frozen=True)`` flip. The sibling classes that
share this coercion idiom use ``object.__setattr__`` defensively;
this test guards against the bare-assignment returning.
"""

from __future__ import annotations

import inspect

from dqlitewire.messages.responses import StmtResponse


def test_stmt_response_post_init_uses_object_setattr() -> None:
    src = inspect.getsource(StmtResponse.__post_init__)
    assert "object.__setattr__" in src, (
        "StmtResponse.__post_init__ must coerce via object.__setattr__ to "
        "match the AssignRequest / NodeInfo sibling pattern."
    )
    # The bare assignment must not appear as an actual statement; the
    # forward-compat comment is allowed to mention it as a string in
    # double-backticks.
    code_lines = [line for line in src.splitlines() if not line.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    assert "self.tail_offset = 0" not in code


def test_stmt_response_v1_default_tail_offset_normalised_to_zero() -> None:
    """Behaviour pin: the coercion still runs — ``schema=1``
    with ``tail_offset=None`` normalises to ``tail_offset=0``."""
    msg = StmtResponse(db_id=1, stmt_id=2, num_params=0, schema=1)
    assert msg.tail_offset == 0
