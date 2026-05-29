"""The legacy LeaderResponse dispatch must gate on the direction guard
``not self._is_request``, not the coincidental ``msg_class is LeaderResponse``
check: RequestType.CLIENT and ResponseType.LEADER both equal 1, and a future
RESPONSE_TYPES alias would break the identity coupling.
"""

from __future__ import annotations

import inspect

import pytest

from dqlitewire import PROTOCOL_VERSION_LEGACY
from dqlitewire.codec import (
    MessageDecoder,
    decode_message,
)
from dqlitewire.constants import RequestType, ResponseType
from dqlitewire.messages.base import Header
from dqlitewire.messages.requests import ClientRequest
from dqlitewire.messages.responses import LeaderResponse


def test_legacy_leader_dispatch_gates_on_direction_not_identity() -> None:
    src = inspect.getsource(MessageDecoder.decode_bytes)
    assert "not self._is_request" in src, (
        "Legacy LeaderResponse dispatch must gate on the direction "
        "guard ``not self._is_request``, not solely on the identity "
        "coincidence ``msg_class is LeaderResponse``."
    )


def _build_legacy_leader_response_bytes(address: str) -> bytes:
    """Legacy LEADER response bytes: header + padded NUL-terminated address, with no
    node_id prefix (that field was added in V1)."""
    payload = address.encode("utf-8") + b"\x00"
    pad = (-len(payload)) % 8
    body = payload + b"\x00" * pad
    header = Header(
        size_words=len(body) // 8,
        msg_type=ResponseType.LEADER,
        schema=0,
    )
    return header.encode() + body


def _build_client_request_bytes(client_id: int) -> bytes:
    """A ClientRequest has msg_type 1 (== ResponseType.LEADER); a request-side decoder
    must route it through ClientRequest.decode_body, not decode_body_legacy."""
    return ClientRequest(client_id=client_id).encode()


def test_legacy_dispatch_fires_on_response_side_legacy_version() -> None:
    """Positive twin: response-side + LEGACY + LEADER routes through decode_body_legacy,
    which synthesises node_id=0 (the legacy body has no node_id prefix)."""
    addr = "leader.example.com:9001"
    wire = _build_legacy_leader_response_bytes(addr)

    result = decode_message(wire, is_request=False, version=PROTOCOL_VERSION_LEGACY)

    assert isinstance(result, LeaderResponse)
    assert result.address == addr
    # node_id=0 confirms legacy decode ran; non-zero would mean V1 decode_body
    # mis-parsed the address's first 8 bytes as a node_id.
    assert result.node_id == 0


def test_legacy_dispatch_does_not_fire_on_request_side_client_msg_type() -> None:
    """Negative twin: a request-side decoder on msg_type=1 must NOT route through
    decode_body_legacy, even with version=LEGACY — the direction guard prevents
    mis-parsing the uint64 client_id as a NUL-terminated legacy address."""
    wire = _build_client_request_bytes(client_id=0x1234567812345678)

    result = decode_message(wire, is_request=True, version=PROTOCOL_VERSION_LEGACY)

    assert isinstance(result, ClientRequest), (
        f"request-side decoder must route msg_type=1 to ClientRequest, "
        f"not legacy LeaderResponse — got {type(result).__name__}"
    )
    assert not isinstance(result, LeaderResponse)
    assert result.client_id == 0x1234567812345678


def test_v1_leader_response_does_not_route_through_legacy_decode() -> None:
    """Version-guard negative: a V1 LEADER response routes through V1 decode_body, not
    decode_body_legacy. Dropping the version predicate would mis-parse the node_id
    prefix as the start of the address string."""
    from dqlitewire.constants import PROTOCOL_VERSION

    node_id = 0x4242
    address = "leader.example.com:9001"
    payload = address.encode("utf-8") + b"\x00"
    pad = (-len(payload)) % 8
    body = node_id.to_bytes(8, "little") + payload + b"\x00" * pad
    header = Header(
        size_words=len(body) // 8,
        msg_type=ResponseType.LEADER,
        schema=0,
    )
    wire = header.encode() + body

    result = decode_message(wire, is_request=False, version=PROTOCOL_VERSION)

    assert isinstance(result, LeaderResponse)
    assert result.node_id == node_id, (
        f"V1 decoder must read node_id from the 8-byte prefix; got "
        f"{result.node_id}. node_id=0 would indicate legacy dispatch "
        f"ran on V1 bytes."
    )
    assert result.address == address


def test_client_request_and_leader_response_share_msg_type_code() -> None:
    """The direction guard exists for this numeric collision; if either type is
    renumbered the guard becomes redundant and this assertion flags it."""
    assert RequestType.CLIENT == ResponseType.LEADER == 1


def test_request_side_with_aliased_response_type_at_collision_still_skips_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future-proof pin: even with RESPONSE_TYPES[1] rebound to ClientRequest (defeating
    the identity check), the direction guard must still keep request-side msg_type=1 off
    the legacy arm — a guard-dropping "simplification" passes the other twins but not this."""
    import dqlitewire.codec as codec_module

    # setitem restores the original on teardown, keeping test isolation.
    monkeypatch.setitem(codec_module.RESPONSE_TYPES, 1, ClientRequest)
    wire = _build_client_request_bytes(client_id=0xDEADBEEFCAFEBABE)

    # version=LEGACY would gate the legacy arm on its own; the direction must override.
    result = decode_message(wire, is_request=True, version=PROTOCOL_VERSION_LEGACY)

    assert isinstance(result, ClientRequest), (
        f"request-side decoder must route msg_type=1 to ClientRequest "
        f"via the direction guard, not LeaderResponse.decode_body_legacy "
        f"via the identity check; got {type(result).__name__}"
    )
    assert result.client_id == 0xDEADBEEFCAFEBABE
