"""AssignRequest docstring records the deliberate divergence from C's
silent-fold-to-VOTER for unknown roles (we reject so future role codes
aren't masked in mixed-version rollouts), plus a frozen=True tripwire on
the post-init role coercion."""

from __future__ import annotations

from dqlitewire.constants import NodeRole
from dqlitewire.messages.requests import AssignRequest


def test_assign_request_raw_int_role_coerces_to_nodeRole_and_equates() -> None:
    """Tripwire for a frozen=True flip: post-init role coercion must succeed."""
    msg = AssignRequest(node_id=1, role=0)
    assert msg.role is NodeRole.VOTER
    assert msg == AssignRequest(node_id=1, role=NodeRole.VOTER)
