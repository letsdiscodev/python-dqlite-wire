"""Pin: ``NodeInfo.__post_init__`` rejects the raft-config-invariant
violations ``(node_id=0, *)`` and ``(node_id != 0, address="")`` so a
caller-built malformed value can't reach ``ServersResponse.encode_body``
and emit bytes the same package's ``ServersResponse.decode_body`` will
reject.

Upstream ``dqlite-upstream/src/gateway.c::encodeServer`` populates each
entry from ``raft->configuration.servers``, which the Raft API
documents as requiring ``node_id >= 1`` and a non-empty address (the
upstream ``Add`` request rejects empty addresses). So the only
legitimate per-entry shape is ``(>=1, non-empty)``.

The decoder side has rejected these shapes since a prior cycle
(``ServersResponse.decode_body``'s raft-configuration invariant
check). The encoder side did not — a caller could:

    ServersResponse(nodes=[NodeInfo(node_id=0, address="evil", role=VOTER)]).encode_body()

producing wire bytes the same package's ``decode_body`` then refused.
Mock-server / proxy / golden-byte authors got values that round-
tripped in neither direction.

Symmetric with the LeaderResponse encode-side fix in
``test_leader_response_encode_side_atomicity.py``. The check lives on
``NodeInfo.__post_init__`` rather than on ``ServersResponse.encode_body``
because there is no legacy decoder that produces ``(0, addr)`` entries
(unlike ``LeaderResponse.decode_body_legacy``) — every existing
construction site uses the legitimate shape. Failing fast at the
``NodeInfo`` caller frame is the cleanest fail point.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import NodeRole
from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.responses import NodeInfo, ServersResponse


def test_zero_id_with_nonempty_address_rejected_at_construction() -> None:
    """``NodeInfo.__post_init__`` rejects ``node_id=0``; raft node ids
    are >= 1 by invariant."""
    with pytest.raises(EncodeError, match="node_id"):
        NodeInfo(node_id=0, address="evil:9001", role=NodeRole.VOTER)


def test_zero_id_with_empty_address_rejected_at_construction() -> None:
    """``(0, "")`` is meaningful in ``LeaderResponse`` ("no leader
    known") but inside a ServersResponse entry it represents a
    phantom no-op node. Rejected for the same reason."""
    with pytest.raises(EncodeError, match="node_id"):
        NodeInfo(node_id=0, address="", role=NodeRole.VOTER)


def test_nonzero_id_with_empty_address_rejected_at_construction() -> None:
    """A ``node_id != 0`` entry must carry a routable address — the
    upstream ``Add`` request requires non-empty address."""
    with pytest.raises(EncodeError, match="address"):
        NodeInfo(node_id=42, address="", role=NodeRole.VOTER)


def test_nonzero_id_with_nonempty_address_accepted() -> None:
    """The only legitimate shape ``(>=1, non-empty)``."""
    node = NodeInfo(node_id=5, address="host:9001", role=NodeRole.VOTER)
    assert node.node_id == 5
    assert node.address == "host:9001"


def test_construction_reject_message_includes_caller_value() -> None:
    """The construction-time reject diagnostic identifies which caller-
    supplied field was malformed."""
    with pytest.raises(EncodeError) as exc_info:
        NodeInfo(node_id=0, address="phantom:1234", role=NodeRole.VOTER)
    diag = str(exc_info.value)
    assert "node_id" in diag

    with pytest.raises(EncodeError) as exc_info:
        NodeInfo(node_id=99, address="", role=NodeRole.VOTER)
    diag = str(exc_info.value)
    assert "address" in diag


def test_servers_response_round_trip_unchanged_for_valid_shapes() -> None:
    """Regression: a normal multi-node ServersResponse still round-
    trips cleanly through the encoder + decoder pair."""
    from dqlitewire.codec import MessageDecoder, encode_message

    msg = ServersResponse(
        nodes=[
            NodeInfo(node_id=1, address="n1:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="n2:9002", role=NodeRole.STANDBY),
            NodeInfo(node_id=3, address="n3:9003", role=NodeRole.SPARE),
        ]
    )
    decoder = MessageDecoder(is_request=False)
    bytes_out = encode_message(msg)
    decoded = decoder.decode_bytes(bytes_out)
    assert isinstance(decoded, ServersResponse)
    assert len(decoded.nodes) == 3
    assert decoded.nodes[0].node_id == 1
    assert decoded.nodes[2].role == NodeRole.SPARE


def test_empty_servers_response_still_encodes_cleanly() -> None:
    """Regression: an empty ServersResponse (zero nodes) is the
    canonical "cluster topology unknown" shape and must encode
    cleanly. The atomicity check fires on each NodeInfo, not on the
    container, so an empty list bypasses it."""
    from dqlitewire.codec import MessageDecoder, encode_message

    msg = ServersResponse(nodes=[])
    decoder = MessageDecoder(is_request=False)
    bytes_out = encode_message(msg)
    decoded = decoder.decode_bytes(bytes_out)
    assert isinstance(decoded, ServersResponse)
    assert decoded.nodes == []


def test_pre_existing_decode_side_reject_still_works() -> None:
    """Regression: the decode-side reject for the same shape still
    fires (defense in depth against a malicious peer that bypasses
    Python construction via raw bytes)."""
    from dqlitewire.exceptions import DecodeError
    from dqlitewire.types import encode_text, encode_uint64

    # Hand-build a (0, non-empty, VOTER) entry frame WITHOUT going
    # through the Python constructor.
    body = (
        encode_uint64(1)  # count = 1
        + encode_uint64(0)  # node_id = 0 (invalid)
        + encode_text("evil:9001")
        + encode_uint64(0)  # role = VOTER
    )
    with pytest.raises(DecodeError, match="malformed"):
        ServersResponse.decode_body(body)
