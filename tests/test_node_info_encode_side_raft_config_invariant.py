"""``NodeInfo.__post_init__`` rejects raft-config violations ``(node_id=0, *)`` and
``(node_id != 0, address="")``; only ``(>=1, non-empty)`` is legitimate per upstream Raft.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import NodeRole
from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.responses import NodeInfo, ServersResponse


def test_zero_id_with_nonempty_address_rejected_at_construction() -> None:
    """Raft node ids are >= 1 by invariant."""
    with pytest.raises(EncodeError, match="node_id"):
        NodeInfo(node_id=0, address="evil:9001", role=NodeRole.VOTER)


def test_zero_id_with_empty_address_rejected_at_construction() -> None:
    """``(0, "")`` means "no leader" in LeaderResponse but is a phantom node here."""
    with pytest.raises(EncodeError, match="node_id"):
        NodeInfo(node_id=0, address="", role=NodeRole.VOTER)


def test_nonzero_id_with_empty_address_rejected_at_construction() -> None:
    """A ``node_id != 0`` entry must carry a routable address."""
    with pytest.raises(EncodeError, match="address"):
        NodeInfo(node_id=42, address="", role=NodeRole.VOTER)


def test_nonzero_id_with_nonempty_address_accepted() -> None:
    node = NodeInfo(node_id=5, address="host:9001", role=NodeRole.VOTER)
    assert node.node_id == 5
    assert node.address == "host:9001"


def test_construction_reject_message_includes_caller_value() -> None:
    """The reject diagnostic identifies which field was malformed."""
    with pytest.raises(EncodeError) as exc_info:
        NodeInfo(node_id=0, address="phantom:1234", role=NodeRole.VOTER)
    diag = str(exc_info.value)
    assert "node_id" in diag

    with pytest.raises(EncodeError) as exc_info:
        NodeInfo(node_id=99, address="", role=NodeRole.VOTER)
    diag = str(exc_info.value)
    assert "address" in diag


def test_servers_response_round_trip_unchanged_for_valid_shapes() -> None:
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
    """Empty ServersResponse ("topology unknown") encodes cleanly: the check fires per-NodeInfo."""
    from dqlitewire.codec import MessageDecoder, encode_message

    msg = ServersResponse(nodes=[])
    decoder = MessageDecoder(is_request=False)
    bytes_out = encode_message(msg)
    decoded = decoder.decode_bytes(bytes_out)
    assert isinstance(decoded, ServersResponse)
    assert decoded.nodes == []


def test_pre_existing_decode_side_reject_still_works() -> None:
    """Decode-side reject also fires: defense against a peer bypassing Python construction."""
    from dqlitewire.exceptions import DecodeError
    from dqlitewire.types import encode_text, encode_uint64

    # Hand-build a (0, non-empty, VOTER) frame, bypassing the constructor.
    body = (
        encode_uint64(1)  # count = 1
        + encode_uint64(0)  # node_id = 0 (invalid)
        + encode_text("evil:9001")
        + encode_uint64(0)  # role = VOTER
    )
    with pytest.raises(DecodeError, match="malformed"):
        ServersResponse.decode_body(body)
