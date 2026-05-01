"""Pin: ``AssignRequest.encode_body_legacy`` docstring does NOT
falsely claim parity with ``LeaderResponse.encode_body_legacy``.

The two methods diverge intentionally:
- ``LeaderResponse.encode_body_legacy`` raises ``EncodeError`` if a
  non-zero ``node_id`` would be silently lost in the legacy shape.
- ``AssignRequest.encode_body_legacy`` silently drops ``role`` (the
  legacy PROMOTE body has no role field by design).

The previous docstring used the bare phrase "Mirror of
``LeaderResponse.encode_body_legacy``", which elsewhere in this
package signals a bidirectional information-loss-rejection guarantee.
Pin the corrected docstring so a future re-tightening of the legacy
encoders does not silently re-introduce the misleading phrasing.
"""

from __future__ import annotations

from dqlitewire.messages.requests import AssignRequest


def test_assignrequest_legacy_docstring_does_not_claim_mirror() -> None:
    doc = AssignRequest.encode_body_legacy.__doc__ or ""
    assert "Mirror of ``LeaderResponse.encode_body_legacy``" not in doc, (
        "Docstring must not claim mirror parity with LeaderResponse.encode_body_legacy "
        "— that sibling rejects information loss; this one silently drops role."
    )


def test_assignrequest_legacy_docstring_documents_role_drop() -> None:
    doc = AssignRequest.encode_body_legacy.__doc__ or ""
    assert "role" in doc.lower(), "Docstring must mention role"
    assert "drops" in doc.lower() or "drop" in doc.lower(), (
        "Docstring must explicitly call out that role is dropped"
    )


def test_assignrequest_legacy_silently_drops_role_behaviour() -> None:
    """Pin behaviour: legacy encoder is information-lossy by design.
    A request with role=NodeRole.SPARE encodes to the same 8-byte
    legacy body as a request with role=None — confirming the docstring's
    "silently drops role" claim is honest about behaviour."""
    from dqlitewire.constants import NodeRole

    req_with_role = AssignRequest(node_id=42, role=NodeRole.SPARE)
    req_no_role = AssignRequest(node_id=42, role=None)
    assert req_with_role.encode_body_legacy() == req_no_role.encode_body_legacy()
    assert len(req_with_role.encode_body_legacy()) == 8
