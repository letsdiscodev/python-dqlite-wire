"""Pin: the ``AssignRequest`` docstring documents the deliberate
divergence from C's silent-fold semantics for unknown roles, and the
construction path does not depend on post-init attribute mutation in a
way that would silently break under ``@dataclass(frozen=True)``.

Upstream C (``dqlite-upstream/src/translate.c::translateDqliteRole``)
folds unknown role uint64 values to ``DQLITE_VOTER`` via its
``default:`` arm — commented as "For backward compat with clients
that don't set a role". The Python wire layer rejects with
``DecodeError`` instead so a future role code (e.g. ``OBSERVER = 3``)
cannot be silently masked during a mixed-version rollout. The
narrowing is deliberate; the docstring must record the divergence so
a future maintainer doesn't conclude it was an oversight.

The second test guards the frozen-fragility tripwire: if the
dataclass is ever flipped to ``@dataclass(frozen=True)`` without
folding the coercion into ``__init__``, raw-int construction will
raise ``FrozenInstanceError`` and this test fails.
"""

from __future__ import annotations

from dqlitewire.constants import NodeRole
from dqlitewire.messages.requests import AssignRequest


def test_assign_request_docstring_documents_c_role_narrowing_divergence() -> None:
    """The class docstring must record the deliberate divergence from
    C's silent-fold semantics so a future maintainer doesn't relax
    the narrowing to "match C"."""
    doc = AssignRequest.__doc__ or ""
    assert "translateDqliteRole" in doc, (
        "docstring must name the C reference symbol so maintainers can grep"
    )
    assert "Strict-decode role narrowing diverges from C" in doc


def test_assign_request_raw_int_role_coerces_to_nodeRole_and_equates() -> None:
    """Tripwire for a future ``frozen=True`` flip without ``__init__``
    fold: the post-init coercion must succeed and produce an instance
    equal to its enum-constructed sibling."""
    msg = AssignRequest(node_id=1, role=0)
    assert msg.role is NodeRole.VOTER
    assert msg == AssignRequest(node_id=1, role=NodeRole.VOTER)
