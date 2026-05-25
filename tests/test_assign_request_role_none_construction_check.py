"""Pin: ``AssignRequest`` rejects bare ``role=None`` at construction
time unless the explicit ``_legacy_intent=True`` sentinel is set.

Without the construction-time check, ``AssignRequest(node_id=42)``
would admit an invalid state into the dataclass and fail only at
``encode_body()`` — possibly far from the construction site. The
sentinel mirrors the ``_decoded_schema`` pattern on ``ExecRequest`` so
the project keeps a uniform vocabulary for "this dataclass holds a
shape that requires explicit caller intent".
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.requests import AssignRequest
from dqlitewire.types import encode_uint64


def test_assign_request_bare_construction_rejects_role_none() -> None:
    """The bare-typo case — ``AssignRequest(node_id=42)`` without a
    role — must fail at construction, not at encode time."""
    with pytest.raises(EncodeError, match="role"):
        AssignRequest(node_id=42)


def test_assign_request_role_none_with_legacy_intent_constructs() -> None:
    """Callers that explicitly want to emit the legacy 1-word PROMOTE
    body opt in with ``_legacy_intent=True`` and may keep ``role=None``."""
    msg = AssignRequest(node_id=42, role=None, _legacy_intent=True)
    assert msg.role is None
    assert msg.encode_body_legacy() == encode_uint64(42)


def test_assign_request_legacy_intent_does_not_compare_or_repr() -> None:
    """The sentinel field must not appear in equality or repr, matching
    the ``_decoded_schema`` pattern."""
    bare = AssignRequest(node_id=42, role=None, _legacy_intent=True)
    other = AssignRequest(node_id=42, role=None, _legacy_intent=True)
    assert bare == other
    assert "_legacy_intent" not in repr(bare)
