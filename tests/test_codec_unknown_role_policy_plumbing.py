"""Pin: ``MessageDecoder`` plumbs ``unknown_role_policy`` through to
``ServersResponse.decode_body`` so production cluster-info consumers
can opt into forward-compat tolerance without bypassing the streaming
decoder.

Threat model: a future C server emits a new role byte (e.g., role=4).
Default ``"reject"`` mode raises ``DecodeError``. Operators wanting
graceful degradation set ``unknown_role_policy="warn"`` (substitute
``NodeRole.SPARE`` and emit ``logger.warning``) or ``"accept"``
(substitute silently). Without the plumbing the knob was reachable
only by callers who skipped ``MessageDecoder`` and called
``ServersResponse.decode_body`` directly — i.e., dead code through
production paths.
"""

from __future__ import annotations

import logging
import struct

import pytest

from dqlitewire import (
    MessageDecoder,
    NodeRole,
    decode_message,
)
from dqlitewire.constants import ResponseType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.messages.base import Header
from dqlitewire.messages.responses import NodeInfo, ServersResponse


def _build_servers_frame_with_unknown_role(role: int) -> bytes:
    """Construct a SERVERS response frame whose single node has the
    given role byte. Used to drive the policy paths."""
    # Body shape: count uint64 LE | for each node: id uint64 | text(address) | role uint64.
    address = "10.0.0.1:9001"
    addr_bytes = address.encode("utf-8") + b"\x00"
    # Pad address to 8-byte boundary.
    pad = (-len(addr_bytes)) % 8
    addr_padded = addr_bytes + b"\x00" * pad
    body = struct.pack("<Q", 1)  # node count
    body += struct.pack("<Q", 42)  # node_id
    body += addr_padded
    body += struct.pack("<Q", role)
    # Header: size_words + msg_type + schema=0 + reserved=0
    assert len(body) % 8 == 0
    size_words = len(body) // 8
    header_bytes = Header(size_words, ResponseType.SERVERS, schema=0).encode()
    return header_bytes + body


def test_default_decoder_rejects_unknown_role() -> None:
    decoder = MessageDecoder()
    frame = _build_servers_frame_with_unknown_role(role=99)
    decoder.feed(frame)
    with pytest.raises(DecodeError, match="role"):
        decoder.decode()


def test_warn_policy_decoder_substitutes_spare_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    decoder = MessageDecoder(unknown_role_policy="warn")
    frame = _build_servers_frame_with_unknown_role(role=99)
    decoder.feed(frame)
    with caplog.at_level(logging.WARNING):
        result = decoder.decode()
    assert isinstance(result, ServersResponse)
    assert result.nodes == [NodeInfo(node_id=42, address="10.0.0.1:9001", role=NodeRole.SPARE)]
    # warn mode: a single WARNING record was emitted naming the unknown role.
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warning_records) == 1
    assert "99" in warning_records[0].getMessage()


def test_accept_policy_decoder_substitutes_spare_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    decoder = MessageDecoder(unknown_role_policy="accept")
    frame = _build_servers_frame_with_unknown_role(role=99)
    decoder.feed(frame)
    with caplog.at_level(logging.WARNING):
        result = decoder.decode()
    assert isinstance(result, ServersResponse)
    assert result.nodes == [NodeInfo(node_id=42, address="10.0.0.1:9001", role=NodeRole.SPARE)]
    # accept mode: NO warning emitted.
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_records == []


def test_invalid_policy_value_raises_at_construction() -> None:
    # Aligned with the deeper ServersResponse.decode_body validator,
    # which raises EncodeError for the same input (round-29 taxonomy
    # migration). Callers can use a single `except EncodeError` for
    # both layers.
    with pytest.raises(EncodeError, match="unknown_role_policy must be one of"):
        MessageDecoder(unknown_role_policy="bogus")


def test_decode_message_helper_forwards_unknown_role_policy() -> None:
    """Pin: ``decode_message(..., unknown_role_policy="warn")`` parity
    with ``MessageDecoder``. The convenience helper must forward the
    kwarg so stateless one-off decode (e.g., from a packet trace) can
    opt into forward-compat tolerance."""
    frame = _build_servers_frame_with_unknown_role(role=99)
    # Default reject mode: raises.
    with pytest.raises(DecodeError, match="role"):
        decode_message(frame)
    # warn mode: substitute SPARE silently here — log assertion is
    # covered by the streaming-decoder pin above.
    result = decode_message(frame, unknown_role_policy="warn")
    assert isinstance(result, ServersResponse)
    assert result.nodes == [NodeInfo(node_id=42, address="10.0.0.1:9001", role=NodeRole.SPARE)]


def test_known_role_unchanged_under_all_policies() -> None:
    """Regression guard: known roles {0,1,2} are unaffected by the
    policy knob. Default-mode decoder must continue producing the
    same NodeInfo shape it always has."""
    for policy in ("reject", "warn", "accept"):
        decoder = MessageDecoder(unknown_role_policy=policy)
        frame = _build_servers_frame_with_unknown_role(role=0)  # VOTER
        decoder.feed(frame)
        result = decoder.decode()
        assert isinstance(result, ServersResponse)
        assert result.nodes == [NodeInfo(node_id=42, address="10.0.0.1:9001", role=NodeRole.VOTER)]
