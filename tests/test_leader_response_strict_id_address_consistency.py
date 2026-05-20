"""Pin: ``LeaderResponse.decode_body`` rejects malformed
``(node_id=0, address!="")`` and ``(node_id>0, address="")`` shapes
that a conforming upstream C server never emits.

Upstream ``raft_leader`` in ``dqlite-upstream/src/gateway.c::handle_leader``
is atomic — both fields are set together or neither. The wire-layer
rejection is defense-in-depth: the client-side ``_query_leader`` and
``leader_info`` already reject these shapes, but a non-client
consumer of ``LeaderResponse`` would not benefit from those guards.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import LeaderResponse
from dqlitewire.types import encode_text, encode_uint64


def _build_body(node_id: int, address: str) -> bytes:
    return encode_uint64(node_id) + encode_text(address, max_size=64, label="leader address")


def test_leader_response_legitimate_no_leader_known_accepted() -> None:
    """``(0, "")`` is the canonical "no leader known" reply."""
    body = _build_body(0, "")
    resp = LeaderResponse.decode_body(body)
    assert resp.node_id == 0
    assert resp.address == ""


def test_leader_response_legitimate_leader_accepted() -> None:
    """``(nonzero, nonempty)`` is the canonical "leader=X" reply."""
    body = _build_body(7, "leader:9001")
    resp = LeaderResponse.decode_body(body)
    assert resp.node_id == 7
    assert resp.address == "leader:9001"


def test_leader_response_zero_id_with_nonempty_address_rejected() -> None:
    """Hostile/buggy peer emitting ``(0, "evil:9001")`` is rejected."""
    body = _build_body(0, "evil:9001")
    with pytest.raises(DecodeError, match="malformed"):
        LeaderResponse.decode_body(body)


def test_leader_response_nonzero_id_with_empty_address_rejected() -> None:
    """Hostile/buggy peer emitting ``(42, "")`` is rejected."""
    body = _build_body(42, "")
    with pytest.raises(DecodeError, match="malformed"):
        LeaderResponse.decode_body(body)
