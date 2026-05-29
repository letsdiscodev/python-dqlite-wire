"""MessageDecoder must plumb ``unknown_role_policy`` through to
``ServersResponse.decode_body`` so consumers can opt into forward-compat tolerance
("reject"/"warn"/"accept" for an unknown role byte) without bypassing the decoder.
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
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.base import Header
from dqlitewire.messages.responses import NodeInfo, ServersResponse


def _build_servers_frame_with_unknown_role(role: int) -> bytes:
    """A SERVERS response frame whose single node has the given role byte."""
    # Body: count uint64 LE | per node: id uint64 | text(address) | role uint64.
    address = "10.0.0.1:9001"
    addr_bytes = address.encode("utf-8") + b"\x00"
    pad = (-len(addr_bytes)) % 8
    addr_padded = addr_bytes + b"\x00" * pad
    body = struct.pack("<Q", 1)  # node count
    body += struct.pack("<Q", 42)  # node_id
    body += addr_padded
    body += struct.pack("<Q", role)
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
    # DecodeError mirrors the ServersResponse.decode_body validator, so callers
    # can use one ``except DecodeError`` for both construction- and decode-time.
    with pytest.raises(DecodeError, match="unknown_role_policy must be one of"):
        MessageDecoder(unknown_role_policy="bogus")


def test_decode_message_helper_forwards_unknown_role_policy() -> None:
    """decode_message must forward unknown_role_policy too, so stateless one-off decode
    (e.g. from a packet trace) gets the same forward-compat tolerance as MessageDecoder."""
    frame = _build_servers_frame_with_unknown_role(role=99)
    with pytest.raises(DecodeError, match="role"):
        decode_message(frame)
    # warn substitutes SPARE; the log assertion lives in the streaming-decoder test above.
    result = decode_message(frame, unknown_role_policy="warn")
    assert isinstance(result, ServersResponse)
    assert result.nodes == [NodeInfo(node_id=42, address="10.0.0.1:9001", role=NodeRole.SPARE)]


def test_known_role_unchanged_under_all_policies() -> None:
    """Known roles {0,1,2} are unaffected by the policy knob under every mode."""
    for policy in ("reject", "warn", "accept"):
        decoder = MessageDecoder(unknown_role_policy=policy)
        frame = _build_servers_frame_with_unknown_role(role=0)  # VOTER
        decoder.feed(frame)
        result = decoder.decode()
        assert isinstance(result, ServersResponse)
        assert result.nodes == [NodeInfo(node_id=42, address="10.0.0.1:9001", role=NodeRole.VOTER)]
