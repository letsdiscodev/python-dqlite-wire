"""AssignRequest.encode_body_legacy docstring must not claim parity with
LeaderResponse.encode_body_legacy: the sibling rejects information loss,
but this one silently drops role (legacy PROMOTE has no role field)."""

from __future__ import annotations

from dqlitewire.messages.requests import AssignRequest


def test_assignrequest_legacy_silently_drops_role_behaviour() -> None:
    """Legacy encoder is information-lossy: role=SPARE and role=None encode
    to the same 8-byte body."""
    from dqlitewire.constants import NodeRole

    req_with_role = AssignRequest(node_id=42, role=NodeRole.SPARE)
    req_no_role = AssignRequest(node_id=42, role=None, _legacy_intent=True)
    assert req_with_role.encode_body_legacy() == req_no_role.encode_body_legacy()
    assert len(req_with_role.encode_body_legacy()) == 8
