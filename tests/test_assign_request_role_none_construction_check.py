"""AssignRequest rejects bare role=None at construction (failing early, not
at encode time) unless the _legacy_intent=True sentinel is set."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.requests import AssignRequest
from dqlitewire.types import encode_uint64


def test_assign_request_bare_construction_rejects_role_none() -> None:
    """Bare AssignRequest(node_id=42) must fail at construction, not encode."""
    with pytest.raises(EncodeError, match="role"):
        AssignRequest(node_id=42)


def test_assign_request_role_none_with_legacy_intent_constructs() -> None:
    """_legacy_intent=True opts into the legacy PROMOTE body with role=None."""
    msg = AssignRequest(node_id=42, role=None, _legacy_intent=True)
    assert msg.role is None
    assert msg.encode_body_legacy() == encode_uint64(42)


def test_assign_request_legacy_intent_does_not_compare_or_repr() -> None:
    """The sentinel field must not appear in equality or repr."""
    bare = AssignRequest(node_id=42, role=None, _legacy_intent=True)
    other = AssignRequest(node_id=42, role=None, _legacy_intent=True)
    assert bare == other
    assert "_legacy_intent" not in repr(bare)
