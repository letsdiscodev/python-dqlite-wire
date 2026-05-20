"""Pin: ``ServersResponse.decode_body`` enforces the
``(node_id, address)`` atomicity invariant on each entry, mirroring
``LeaderResponse.decode_body``'s discipline.

Raft's node table populates the ``SERVERS`` body from the same store
as the ``LEADER`` body — node ids are >= 1 by raft invariant (0 is
reserved for "no node"), and ``Add`` requests reject empty
addresses. The only legitimate per-entry shapes are
``(0, "")`` (which never appears inside a ServersResponse with
``count > 0``) and ``(>=1, non-empty)``. A peer (mock / proxy /
hostile fork) emitting a mixed shape would silently propagate to
``dqliteclient.cluster.NodeInfo`` and the SQLAlchemy slot cache;
rejecting at the wire boundary keeps the downstream layers'
invariants honest.

Symmetric with ``LeaderResponse`` per
``test_leader_response_strict_id_address_consistency.py``.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import ServersResponse
from dqlitewire.types import encode_text, encode_uint64


def test_servers_response_rejects_zero_id_with_nonempty_address() -> None:
    """A ``node_id=0`` entry with a non-empty address is malformed —
    raft node ids are >= 1 by invariant."""
    body = (
        encode_uint64(1)  # count = 1
        + encode_uint64(0)  # node_id = 0 (invalid)
        + encode_text("evil:9001")
        + encode_uint64(0)  # role = VOTER
    )
    with pytest.raises(DecodeError, match="malformed"):
        ServersResponse.decode_body(body)


def test_servers_response_rejects_nonzero_id_with_empty_address() -> None:
    """A ``node_id != 0`` entry must carry a routable address — the
    upstream ``Add`` request requires non-empty address."""
    body = (
        encode_uint64(1)
        + encode_uint64(42)  # node_id = 42 (valid)
        + encode_text("")  # address = "" (invalid)
        + encode_uint64(0)
    )
    with pytest.raises(DecodeError, match="malformed"):
        ServersResponse.decode_body(body)


def test_servers_response_accepts_legitimate_entry() -> None:
    """``(node_id=42, address="leader:9001", role=VOTER)`` is the
    canonical happy-path shape — no atomicity reject."""
    body = encode_uint64(1) + encode_uint64(42) + encode_text("leader:9001") + encode_uint64(0)
    resp = ServersResponse.decode_body(body)
    assert len(resp.nodes) == 1
    assert resp.nodes[0].node_id == 42
    assert resp.nodes[0].address == "leader:9001"


def test_servers_response_long_malformed_address_truncated_in_error_message() -> None:
    """Defense against an attacker flooding operator logs with multi-
    KiB junk at the rendering site, mirroring ``LeaderResponse``'s
    discipline. The wire decoder bounds the address at
    ``_MAX_ADDRESS_SIZE`` (256 bytes), but the diagnostic itself
    truncates to 64 chars + U+2026 ellipsis."""
    huge = "A" * 200
    body = (
        encode_uint64(1)
        + encode_uint64(0)  # invalid combination with the long address
        + encode_text(huge, max_size=256, label="address")
        + encode_uint64(0)
    )
    with pytest.raises(DecodeError) as excinfo:
        ServersResponse.decode_body(body)
    msg = str(excinfo.value)
    assert huge not in msg, "full address must not leak into diagnostic"
    assert "…" in msg


def test_servers_response_malformed_address_exactly_64_chars_not_truncated() -> None:
    """Boundary: ``len(address) == 64`` renders verbatim, no ellipsis.
    Symmetric with ``LeaderResponse`` sibling — a regression changing
    the predicate to ``< 64`` would silently truncate at the boundary."""
    addr = "B" * 64
    body = (
        encode_uint64(1)
        + encode_uint64(0)
        + encode_text(addr, max_size=256, label="address")
        + encode_uint64(0)
    )
    with pytest.raises(DecodeError) as excinfo:
        ServersResponse.decode_body(body)
    msg = str(excinfo.value)
    assert addr in msg
    assert "…" not in msg


def test_servers_response_malformed_address_65_chars_truncated() -> None:
    """Off-by-one boundary: ``len(address) == 65`` trips truncation.
    Symmetric with ``LeaderResponse`` sibling — a regression changing
    ``<= 64`` to ``<= 65`` would defeat the log-flood cap at the
    single-byte overflow point."""
    addr = "C" * 65
    body = (
        encode_uint64(1)
        + encode_uint64(0)
        + encode_text(addr, max_size=256, label="address")
        + encode_uint64(0)
    )
    with pytest.raises(DecodeError) as excinfo:
        ServersResponse.decode_body(body)
    msg = str(excinfo.value)
    assert "…" in msg
    assert addr not in msg
