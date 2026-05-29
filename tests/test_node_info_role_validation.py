"""``NodeInfo.__post_init__`` rejects role values outside the canonical ``NodeRole`` enum,
so the diagnostic fires at construction rather than at the peer-side decoder.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import NodeRole
from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.responses import NodeInfo, ServersResponse


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (NodeRole.VOTER, NodeRole.VOTER),
        (NodeRole.STANDBY, NodeRole.STANDBY),
        (NodeRole.SPARE, NodeRole.SPARE),
        # Bare ints in the canonical 0/1/2 range are coerced to enum.
        (0, NodeRole.VOTER),
        (1, NodeRole.STANDBY),
        (2, NodeRole.SPARE),
    ],
)
def test_node_info_accepts_canonical_roles(role: NodeRole | int, expected: NodeRole) -> None:
    node = NodeInfo(node_id=1, address="leader:9001", role=role)  # type: ignore[arg-type]
    assert node.role == expected
    assert isinstance(node.role, NodeRole)


@pytest.mark.parametrize("bogus_role", [3, 4, 999])
def test_node_info_rejects_unknown_roles(bogus_role: int) -> None:
    """Bogus role values in uint64 range must be rejected at construction, not on the wire."""
    from dqlitewire.exceptions import EncodeError

    with pytest.raises(EncodeError, match="role"):
        NodeInfo(node_id=1, address="leader:9001", role=bogus_role)  # type: ignore[arg-type]


@pytest.mark.parametrize("out_of_range", [-1, -(2**31), 2**64])
def test_node_info_rejects_out_of_range_roles(out_of_range: int) -> None:
    """Out-of-uint64-range roles raise the uint64 diagnostic, distinct from "not a known role"."""
    with pytest.raises(EncodeError, match="role"):
        NodeInfo(node_id=1, address="leader:9001", role=out_of_range)  # type: ignore[arg-type]


def test_servers_response_round_trip_with_all_valid_roles() -> None:
    """End-to-end: a ServersResponse carrying one node per role
    encodes and decodes back to identical NodeInfos. Pins that the
    construction-time validation does not break the happy path."""
    nodes = [
        NodeInfo(node_id=1, address="voter:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="standby:9002", role=NodeRole.STANDBY),
        NodeInfo(node_id=3, address="spare:9003", role=NodeRole.SPARE),
    ]
    body = ServersResponse(nodes=nodes).encode_body()
    decoded = ServersResponse.decode_body(body)
    assert decoded.nodes == nodes
