"""Tests for cluster responses: LeaderResponse, ServersResponse and NodeInfo invariants."""

from __future__ import annotations

import logging

import pytest

from dqlitewire.constants import NodeRole
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.messages.responses import LeaderResponse, NodeInfo, ServersResponse
from dqlitewire.types import encode_text, encode_uint64

# ---- merged from test_leader_response_encode_side_atomicity.py ----
# Pin: ``LeaderResponse.encode_body`` rejects the
# ``(node_id=0, address != "")`` shape so a caller-built malformed
# value can't emit bytes the same package's ``decode_body`` will
# reject.
#
# The raft-leader atomicity invariant (``dqlite-upstream/src/gateway.
# c::handle_leader``) emits two shapes only:
#
#   * ``(0, "")``  — canonical "no leader known" sentinel.
#   * ``(N, addr)`` for ``N >= 1`` — known leader.
#
# A ``RAFT_NOMEM`` transient on the follower's leader-update path can
# ALSO produce ``(N, "")`` — the decoder accepts that, logs a warning,
# and treats it as "leader unknown" (see ``LeaderResponse.decode_
# body``'s ``node_id != 0 and not address`` branch).
#
# The ``(0, non-empty)`` shape was rejected by ``decode_body`` only —
# ``encode_body`` had no symmetric guard, so:
#
#     LeaderResponse(node_id=0, address="evil:9001").encode_body()
#
# produced valid wire bytes that the same package's ``decode_message``
# then refused. Mock-server / proxy / golden-byte authors got values
# that round-tripped in neither direction.
#
# The check lives on ``encode_body`` (not ``__post_init__``) because
# ``decode_body_legacy`` legitimately constructs ``LeaderResponse(0,
# addr)`` from the legacy wire shape — the legacy body carries only
# the address, with ``node_id`` hard-coded 0 on decode. The legacy
# ``encode_body_legacy`` already has its own symmetric strict check
# (non-zero ``node_id`` rejected); the modern ``encode_body`` now
# mirrors that "encode-time strict" pattern.


def test_zero_id_with_nonempty_address_rejected_at_encode() -> None:
    """``encode_body`` raises ``EncodeError`` so the malformed value
    never reaches the wire."""
    msg = LeaderResponse(node_id=0, address="evil:9001")
    with pytest.raises(EncodeError, match="node_id=0"):
        msg.encode_body()


def test_zero_id_with_empty_address_encodes_cleanly() -> None:
    """Canonical "no leader known" sentinel — must encode cleanly."""
    msg = LeaderResponse(node_id=0, address="")
    body = msg.encode_body()
    assert isinstance(body, bytes)


def test_nonzero_id_with_nonempty_address_encodes_cleanly() -> None:
    """Normal "known leader" shape — must encode cleanly."""
    msg = LeaderResponse(node_id=5, address="host:9001")
    body = msg.encode_body()
    assert isinstance(body, bytes)


def test_nonzero_id_with_empty_address_encodes_cleanly() -> None:
    """``RAFT_NOMEM`` transient shape — accepted (decoder warns and
    treats as "leader unknown")."""
    msg = LeaderResponse(node_id=42, address="")
    body = msg.encode_body()
    assert isinstance(body, bytes)


def test_encode_reject_message_includes_address_for_diagnostics() -> None:
    """The encode-time reject diagnostic carries enough context for an
    operator reading a build / test log to identify the malformed
    caller frame."""
    msg = LeaderResponse(node_id=0, address="phantom:1234")
    with pytest.raises(EncodeError) as exc_info:
        msg.encode_body()
    diag = str(exc_info.value)
    assert "node_id=0" in diag
    assert "phantom:1234" in diag


def test_encode_reject_message_sanitises_control_codepoints_in_address() -> None:
    """The encode-time reject runs ``sanitize_server_text`` on the
    address before interpolation, matching the decode-side reject's
    discipline. Defends against U+2028 / bidi / ZWSP smuggling through
    ``!r``."""
    # U+2028 LINE SEPARATOR is a control codepoint that Python's ``!r``
    # does NOT escape (unlike LF / CR). ``sanitize_server_text`` strips
    # it. Pin that the raised diagnostic does not carry the literal
    # codepoint.
    msg = LeaderResponse(node_id=0, address="evil :9001")
    with pytest.raises(EncodeError) as exc_info:
        msg.encode_body()
    diag = str(exc_info.value)
    assert " " not in diag


def test_encode_reject_message_truncates_oversize_address() -> None:
    """The reject diagnostic truncates the address to 64 characters
    plus an ellipsis so a hostile / pathological caller can't blow up
    a log line by passing a giant string."""
    long_addr = "a" * 200
    msg = LeaderResponse(node_id=0, address=long_addr)
    with pytest.raises(EncodeError) as exc_info:
        msg.encode_body()
    diag = str(exc_info.value)
    # The diagnostic must not contain the full 200-char run.
    assert "a" * 200 not in diag
    # The ellipsis marker confirms truncation fired.
    assert "…" in diag


def test_pre_existing_round_trip_unchanged_for_valid_shapes() -> None:
    """Regression: the three legitimate shapes still round-trip cleanly
    via the modern encoder + decoder pair."""
    from dqlitewire.codec import MessageDecoder, encode_message

    decoder = MessageDecoder(is_request=False)
    for msg in [
        LeaderResponse(node_id=0, address=""),
        LeaderResponse(node_id=5, address="host:9001"),
        LeaderResponse(node_id=42, address=""),
    ]:
        bytes_out = encode_message(msg)
        decoded = decoder.decode_bytes(bytes_out)
        assert isinstance(decoded, LeaderResponse)
        assert decoded.node_id == msg.node_id
        assert decoded.address == msg.address


def test_pre_existing_decode_side_reject_still_works() -> None:
    """Regression: the decode-side reject for the same shape still
    fires (defense in depth against a malicious peer that bypasses
    Python construction via raw bytes)."""
    from dqlitewire.codec import MessageDecoder
    from dqlitewire.constants import ResponseType
    from dqlitewire.messages.base import Header
    from dqlitewire.types import encode_text, encode_uint64

    # Hand-build a (0, non-empty) frame WITHOUT going through the
    # Python constructor.
    body = encode_uint64(0) + encode_text("evil:9001")
    header = Header(size_words=len(body) // 8, msg_type=ResponseType.LEADER, schema=0)
    data = header.encode() + body
    decoder = MessageDecoder(is_request=False)
    with pytest.raises(DecodeError, match="malformed"):
        decoder.decode_bytes(data)


def test_decode_body_legacy_still_constructs_zero_id_with_nonempty_address() -> None:
    """Regression: ``decode_body_legacy`` legitimately produces
    ``LeaderResponse(0, addr)`` from the legacy wire shape (which
    carries only the address and hard-codes ``node_id=0`` on decode).
    The check on ``encode_body`` MUST NOT block this path."""
    from dqlitewire.types import encode_text

    body = encode_text("host:9001")
    msg = LeaderResponse.decode_body_legacy(body)
    assert msg.node_id == 0
    assert msg.address == "host:9001"


# ---- merged from test_leader_response_strict_id_address_consistency.py ----
# Pin: ``LeaderResponse.decode_body`` rejects the malformed
# ``(node_id=0, address!="")`` shape and tolerates the
# ``(node_id>0, address="")`` shape that a real follower can transiently
# emit after a ``RAFT_NOMEM`` from ``recvUpdateLeader``.
#
# Upstream ``raft_leader`` in ``dqlite-upstream/src/gateway.c::handle_leader``
# normally emits paired ``(0, "")`` or ``(id>0, address!="")``. The
# ``(node_id>0, address="")`` shape is reachable persistently when
# ``recvUpdateLeader`` (raft/recv.c) sets ``current_leader.id = id``
# in step 1 and then fails to malloc the address buffer in step 4
# (``RAFT_NOMEM``). ``handle_leader`` null-coerces the NULL address
# to ``""`` on the wire. The Go client and the C client both treat
# this as "leader unknown"; the Python decoder matches that behaviour
# with a WARNING for forensic visibility.
#
# The ``(node_id=0, address!="")`` shape remains malformed by
# construction — a peer setting an address but reporting id=0 is
# hostile or corrupt — and is still rejected.


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


# ---- merged from test_node_info_encode_side_raft_config_invariant.py ----
# ``NodeInfo.__post_init__`` rejects raft-config violations ``(node_id=0, *)`` and
# ``(node_id != 0, address="")``; only ``(>=1, non-empty)`` is legitimate per upstream Raft.


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


def test_node_info_pre_existing_decode_side_reject_still_works() -> None:
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


# ---- merged from test_node_info_role_validation.py ----
# ``NodeInfo.__post_init__`` rejects role values outside the canonical ``NodeRole`` enum,
# so the diagnostic fires at construction rather than at the peer-side decoder.


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


# ---- merged from test_servers_response_duplicate_node_id_rejected.py ----
# Pin: ``ServersResponse.decode_body`` rejects duplicate ``node_id``
# values mid-stream.
#
# The sibling ``FilesResponse.decode_body`` rejects duplicate filenames;
# ``ServersResponse`` was previously silent on duplicate node IDs.
# Conforming C/Go servers populate this response from the Raft log,
# which guarantees unique IDs by construction — a duplicate would only
# surface from a misframed or hostile peer. Detecting it here keeps the
# strictness uniform with the sibling decoder and prevents downstream
# consumers (a ``dict`` keyed by ``node_id``) from silently overwriting
# a prior entry.


def _build_body_with_duplicate_ids() -> bytes:
    """Two nodes with the same node_id but different addresses —
    illegal per the Raft log uniqueness invariant."""
    return (
        encode_uint64(2)  # node count
        + encode_uint64(1)  # node 1
        + encode_text("addr1:9001")
        + encode_uint64(0)  # voter
        + encode_uint64(1)  # node 1 (DUPLICATE id)
        + encode_text("addr2:9002")
        + encode_uint64(0)
    )


def test_duplicate_node_id_rejected() -> None:
    body = _build_body_with_duplicate_ids()
    with pytest.raises(DecodeError, match="Duplicate node_id 1"):
        ServersResponse.decode_body(body)


def test_unique_node_ids_accepted() -> None:
    """Negative pin: legitimate distinct IDs decode cleanly."""
    body = (
        encode_uint64(2)
        + encode_uint64(1)
        + encode_text("addr1:9001")
        + encode_uint64(0)
        + encode_uint64(2)
        + encode_text("addr2:9002")
        + encode_uint64(0)
    )
    response = ServersResponse.decode_body(body)
    assert len(response.nodes) == 2
    assert response.nodes[0].node_id == 1
    assert response.nodes[1].node_id == 2


# ---- merged from test_servers_response_id_address_atomicity.py ----
# Pin: ``ServersResponse.decode_body`` enforces the
# ``(node_id, address)`` atomicity invariant on each entry, mirroring
# ``LeaderResponse.decode_body``'s discipline.
#
# Raft's node table populates the ``SERVERS`` body from the same store
# as the ``LEADER`` body — node ids are >= 1 by raft invariant (0 is
# reserved for "no node"), and ``Add`` requests reject empty
# addresses. The only legitimate per-entry shapes are
# ``(0, "")`` (which never appears inside a ServersResponse with
# ``count > 0``) and ``(>=1, non-empty)``. A peer (mock / proxy /
# hostile fork) emitting a mixed shape would silently propagate to
# ``dqliteclient.cluster.NodeInfo`` and the SQLAlchemy slot cache;
# rejecting at the wire boundary keeps the downstream layers'
# invariants honest.
#
# Symmetric with ``LeaderResponse`` per
# ``test_leader_response_strict_id_address_consistency.py``.


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


def test_servers_response_rejects_zero_id_with_empty_address() -> None:
    """Symmetric ``(0, "")`` is also rejected inside a ServersResponse
    entry. The shape is meaningful only in ``LeaderResponse`` ("no
    leader known"); inside a ServersResponse it represents a phantom
    no-op node that the prior atomicity-only predicate silently
    accepted. Raft node ids are >= 1 by invariant — a configuration
    entry with id=0 is never legitimate."""
    body = (
        encode_uint64(1)  # count = 1
        + encode_uint64(0)  # node_id = 0 (invalid in ServersResponse)
        + encode_text("")  # address = "" (the symmetric pair)
        + encode_uint64(0)  # role = VOTER
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
    ``MAX_ADDRESS_SIZE`` (256 bytes), but the diagnostic itself
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


# ---- merged from test_servers_response_unknown_role_policy.py ----
# ``ServersResponse.decode_body`` exposes an ``unknown_role_policy``
# kwarg for forward-compat with future ``NodeRole`` values.
#
# Default ``"reject"`` keeps the strict-decode posture. ``"warn"``
# emits a ``logger.warning`` and substitutes ``SPARE`` (safest
# default — cannot serve writes, won't be probed for leadership).
# ``"accept"`` silently substitutes ``SPARE``.
#
# Go's ``getNodes`` accepts any uint64 silently and returns
# "unknown role" via ``NodeRole.String()``. Python's ``"warn"`` /
# ``"accept"`` approach that posture without breaking the
# ``NodeRole`` enum's strict typing.


def _build_unknown_role_body(node_id: int, address: str, raw_role: int) -> bytes:
    """Build a single-node ServersResponse body with a chosen raw
    role value."""
    return (
        encode_uint64(1)  # node count
        + encode_uint64(node_id)
        + encode_text(address)
        + encode_uint64(raw_role)
    )


def test_unknown_role_default_rejects() -> None:
    body = _build_unknown_role_body(1, "h:9001", raw_role=99)
    with pytest.raises(DecodeError, match="Invalid node role 99"):
        ServersResponse.decode_body(body)


def test_unknown_role_reject_explicit_matches_default() -> None:
    body = _build_unknown_role_body(1, "h:9001", raw_role=99)
    with pytest.raises(DecodeError, match="Invalid node role 99"):
        ServersResponse.decode_body(body, unknown_role_policy="reject")


def test_unknown_role_warn_substitutes_spare(caplog: pytest.LogCaptureFixture) -> None:
    body = _build_unknown_role_body(1, "h:9001", raw_role=99)
    with caplog.at_level(logging.WARNING):
        resp = ServersResponse.decode_body(body, unknown_role_policy="warn")
    assert len(resp.nodes) == 1
    assert resp.nodes[0].role is NodeRole.SPARE
    # Aggregated single WARNING per response: cardinality of unknown
    # roles is small; one record should mention 99.
    assert any("99" in r.message for r in caplog.records)
    assert any("substituted SPARE" in r.message for r in caplog.records)


def test_unknown_role_accept_substitutes_spare_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = _build_unknown_role_body(1, "h:9001", raw_role=99)
    with caplog.at_level(logging.WARNING):
        resp = ServersResponse.decode_body(body, unknown_role_policy="accept")
    assert resp.nodes[0].role is NodeRole.SPARE
    # No log emitted under "accept".
    assert not any("unknown NodeRole" in r.message for r in caplog.records)


def test_known_role_unaffected_by_policy() -> None:
    """All three known roles round-trip cleanly under every policy."""
    for raw_role, expected in (
        (0, NodeRole.VOTER),
        (1, NodeRole.STANDBY),
        (2, NodeRole.SPARE),
    ):
        body = _build_unknown_role_body(1, "h:9001", raw_role=raw_role)
        for policy in ("reject", "warn", "accept"):
            resp = ServersResponse.decode_body(body, unknown_role_policy=policy)
            assert resp.nodes[0].role is expected


def test_invalid_policy_raises_decode_error() -> None:
    """`decode_body` validates kwargs as part of the decode contract;
    every error from a `decode_*` path surfaces under `DecodeError`,
    not the encode-direction `EncodeError`. A caller writing
    `except DecodeError` must catch the bad-policy validation."""
    from dqlitewire.exceptions import DecodeError

    body = _build_unknown_role_body(1, "h:9001", raw_role=0)
    with pytest.raises(DecodeError, match="unknown_role_policy"):
        ServersResponse.decode_body(body, unknown_role_policy="bogus")


def test_unknown_role_warn_aggregates_one_log_per_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin: warn-mode emits ONE WARNING per response regardless of
    how many nodes carry unknown roles. A multi-node ``cluster_info()``
    poll over a cluster after a server upgrade would otherwise produce
    one WARNING per node per poll — continuous flow on operator
    dashboards."""
    # Build a body with 3 nodes carrying unknown role 99.
    body_parts = bytearray()
    import struct as _struct

    body_parts.extend(_struct.pack("<Q", 3))  # count
    for node_id in (1, 2, 3):
        body_parts.extend(_struct.pack("<Q", node_id))
        addr = f"h:{9000 + node_id}".encode()
        body_parts.extend(addr + b"\x00")
        # pad address to word boundary
        pad = (8 - ((len(addr) + 1) % 8)) % 8
        body_parts.extend(b"\x00" * pad)
        body_parts.extend(_struct.pack("<Q", 99))  # unknown role
    body = bytes(body_parts)

    with caplog.at_level(logging.WARNING):
        resp = ServersResponse.decode_body(body, unknown_role_policy="warn")

    # All three nodes substituted SPARE.
    assert all(n.role is NodeRole.SPARE for n in resp.nodes)
    # ONE warning record total — not one per node.
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, f"expected 1 WARNING, got {len(warnings)}"
    assert "3 node(s)" in warnings[0].message
    assert "99" in warnings[0].message


# ---- merged from test_decode_error_display_addr_sanitises_control_codepoints.py ----
# Pin: the malformed ``(node_id, address)`` ``DecodeError`` in both
# ``LeaderResponse`` and ``ServersResponse`` routes the address through
# ``sanitize_server_text`` before truncation and ``!r`` rendering. ``repr``
# escapes ``\n``/``\r``/``\t`` but not the broader line-separator/bidi/
# zero-width class, so a hostile single codepoint could otherwise inject a
# line-split into journald/SIEM ingest (truncation does not help — one
# codepoint fits).


def test_leader_response_malformed_decode_error_sanitises_u2028() -> None:
    """``str(DecodeError)`` must not contain a raw U+2028 line separator
    (repr does not escape it)."""
    # Malformed atomicity: node_id=0 paired with a non-empty address.
    forged = "leader FORGED:9001"
    body = encode_uint64(0) + encode_text(forged)
    with pytest.raises(DecodeError) as exc_info:
        LeaderResponse.decode_body(body)
    msg = str(exc_info.value)
    assert " " not in msg, (
        "U+2028 LINE SEPARATOR must be sanitised before reaching the "
        "diagnostic; ``repr`` does not escape it"
    )


def test_leader_response_malformed_decode_error_sanitises_bidi() -> None:
    """U+202E RIGHT-TO-LEFT OVERRIDE must be sanitised so an attacker can't
    reorder the diagnostic shown in operator log viewers."""
    forged = "leader‮FORGED:9001"
    body = encode_uint64(0) + encode_text(forged)
    with pytest.raises(DecodeError) as exc_info:
        LeaderResponse.decode_body(body)
    msg = str(exc_info.value)
    assert "‮" not in msg


def test_servers_response_malformed_decode_error_sanitises_u2028() -> None:
    """Sibling pin for ``ServersResponse``'s per-entry atomicity check."""
    forged = "node 1:9001"
    body = (
        encode_uint64(1)  # count
        + encode_uint64(0)  # malformed: node_id=0 with non-empty address
        + encode_text(forged)
        + encode_uint64(0)
    )
    with pytest.raises(DecodeError) as exc_info:
        ServersResponse.decode_body(body)
    msg = str(exc_info.value)
    assert " " not in msg


def test_leader_response_malformed_decode_error_truncation_still_applied() -> None:
    """An address just over the cap still gets the truncation suffix:
    sanitising first must not bypass it. Length is pinned at ``cap + 1`` so a
    future cap raise surfaces as a deliberate edit here, not a misread
    regression."""
    forged = "A" * 65
    body = encode_uint64(0) + encode_text(forged)
    with pytest.raises(DecodeError) as exc_info:
        LeaderResponse.decode_body(body)
    msg = str(exc_info.value)
    assert "…" in msg, (
        "the cap's U+2026 ellipsis suffix must still apply after sanitisation "
        "when the address exceeds the truncation cap"
    )
    assert forged not in msg, (
        "the rendered address must be shorter than the original forged "
        "input — truncation must fire after sanitisation"
    )


def test_servers_response_malformed_decode_error_truncation_still_applied() -> None:
    """Sibling pin for ``ServersResponse`` (see above for the ``cap + 1``
    rationale)."""
    forged = "B" * 65
    body = encode_uint64(1) + encode_uint64(0) + encode_text(forged) + encode_uint64(0)
    with pytest.raises(DecodeError) as exc_info:
        ServersResponse.decode_body(body)
    msg = str(exc_info.value)
    assert "…" in msg
    assert forged not in msg
