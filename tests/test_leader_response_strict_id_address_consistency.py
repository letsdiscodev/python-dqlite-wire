"""Pin: ``LeaderResponse.decode_body`` rejects the malformed
``(node_id=0, address!="")`` shape and tolerates the
``(node_id>0, address="")`` shape that a real follower can transiently
emit after a ``RAFT_NOMEM`` from ``recvUpdateLeader``.

Upstream ``raft_leader`` in ``dqlite-upstream/src/gateway.c::handle_leader``
normally emits paired ``(0, "")`` or ``(id>0, address!="")``. The
``(node_id>0, address="")`` shape is reachable persistently when
``recvUpdateLeader`` (raft/recv.c) sets ``current_leader.id = id``
in step 1 and then fails to malloc the address buffer in step 4
(``RAFT_NOMEM``). ``handle_leader`` null-coerces the NULL address
to ``""`` on the wire. The Go client and the C client both treat
this as "leader unknown"; the Python decoder matches that behaviour
with a WARNING for forensic visibility.

The ``(node_id=0, address!="")`` shape remains malformed by
construction — a peer setting an address but reporting id=0 is
hostile or corrupt — and is still rejected.
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


def test_leader_response_nonzero_id_with_empty_address_accepted_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``(42, "")`` is the RAFT_NOMEM transient: ``recvUpdateLeader``
    sets ``current_leader.id`` first and may then fail to malloc the
    address. A WARNING records the transient; the message decodes to
    an unusable shape that the client layer treats as "leader unknown"
    (same as Go / C clients)."""
    body = _build_body(42, "")
    with caplog.at_level("WARNING", logger="dqlitewire.messages.responses"):
        resp = LeaderResponse.decode_body(body)
    assert resp.node_id == 42
    assert resp.address == ""
    assert any("RAFT_NOMEM" in rec.message for rec in caplog.records)


def _build_body_raw_text(node_id: int, address: str) -> bytes:
    """Build a wire body bypassing ``encode_text``'s ``max_size`` cap so
    we can exercise the decoder's diagnostic-bound truncation arm with
    arbitrarily long addresses. Shape: ``uint64 node_id +
    NUL-terminated UTF-8 + padding-to-word-boundary``."""
    payload = address.encode("utf-8") + b"\x00"
    pad = (-len(payload)) % 8
    return encode_uint64(node_id) + payload + b"\x00" * pad


def test_leader_response_long_malformed_address_truncated_in_error_message() -> None:
    """A hostile peer's oversized malformed address must render at most
    64 chars + ellipsis in the ``DecodeError`` args; the full payload
    must NOT appear. Defense against an attacker flooding operator
    logs with multi-KiB junk at the ``DecodeError`` rendering site.

    The ellipsis is the single character U+2026 (``"…"``), not three
    ASCII dots — match it literally."""
    huge = "A" * 200
    body = _build_body_raw_text(0, huge)
    with pytest.raises(DecodeError) as excinfo:
        LeaderResponse.decode_body(body)
    msg = str(excinfo.value)
    assert huge not in msg, "full address must not leak into diagnostic — defeats log-flood cap"
    assert "…" in msg, "ellipsis marker U+2026 must indicate truncation"
    assert msg.count("A") <= 64


def test_leader_response_address_exactly_64_chars_not_truncated() -> None:
    """Boundary: ``len(address) == 64`` renders verbatim, no ellipsis."""
    addr = "B" * 64
    body = _build_body_raw_text(0, addr)
    with pytest.raises(DecodeError) as excinfo:
        LeaderResponse.decode_body(body)
    msg = str(excinfo.value)
    assert addr in msg
    assert "…" not in msg


def test_leader_response_address_65_chars_truncated() -> None:
    """Off-by-one boundary: ``len(address) == 65`` trips the truncation."""
    addr = "C" * 65
    body = _build_body_raw_text(0, addr)
    with pytest.raises(DecodeError) as excinfo:
        LeaderResponse.decode_body(body)
    msg = str(excinfo.value)
    assert "…" in msg
    assert addr not in msg
