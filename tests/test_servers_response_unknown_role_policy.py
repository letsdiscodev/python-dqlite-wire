"""``ServersResponse.decode_body`` exposes an ``unknown_role_policy``
kwarg for forward-compat with future ``NodeRole`` values.

Default ``"reject"`` keeps the strict-decode posture. ``"warn"``
emits a ``logger.warning`` and substitutes ``SPARE`` (safest
default — cannot serve writes, won't be probed for leadership).
``"accept"`` silently substitutes ``SPARE``.

Go's ``getNodes`` accepts any uint64 silently and returns
"unknown role" via ``NodeRole.String()``. Python's ``"warn"`` /
``"accept"`` approach that posture without breaking the
``NodeRole`` enum's strict typing.
"""

import logging

import pytest

from dqlitewire.constants import NodeRole
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import ServersResponse
from dqlitewire.types import encode_text, encode_uint64


def _build_body(node_id: int, address: str, raw_role: int) -> bytes:
    """Build a single-node ServersResponse body with a chosen raw
    role value."""
    return (
        encode_uint64(1)  # node count
        + encode_uint64(node_id)
        + encode_text(address)
        + encode_uint64(raw_role)
    )


def test_unknown_role_default_rejects() -> None:
    body = _build_body(1, "h:9001", raw_role=99)
    with pytest.raises(DecodeError, match="Invalid node role 99"):
        ServersResponse.decode_body(body)


def test_unknown_role_reject_explicit_matches_default() -> None:
    body = _build_body(1, "h:9001", raw_role=99)
    with pytest.raises(DecodeError, match="Invalid node role 99"):
        ServersResponse.decode_body(body, unknown_role_policy="reject")


def test_unknown_role_warn_substitutes_spare(caplog: pytest.LogCaptureFixture) -> None:
    body = _build_body(1, "h:9001", raw_role=99)
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
    body = _build_body(1, "h:9001", raw_role=99)
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
        body = _build_body(1, "h:9001", raw_role=raw_role)
        for policy in ("reject", "warn", "accept"):
            resp = ServersResponse.decode_body(body, unknown_role_policy=policy)
            assert resp.nodes[0].role is expected


def test_invalid_policy_raises_encode_error() -> None:
    from dqlitewire.exceptions import EncodeError

    body = _build_body(1, "h:9001", raw_role=0)
    with pytest.raises(EncodeError, match="unknown_role_policy"):
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
