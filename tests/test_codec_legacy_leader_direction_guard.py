"""Pin: the legacy LeaderResponse dispatch in ``MessageDecoder``
gates on ``not self._is_request`` (the direction guard) rather than
on the structural-coincidence ``msg_class is LeaderResponse`` check.

``RequestType.CLIENT == 1`` collides numerically with
``ResponseType.LEADER == 1``. The identity check on ``msg_class``
incidentally provides the direction discrimination today, but a
future alias/subclass registration in ``RESPONSE_TYPES`` would
break the implicit coupling. The direction guard is the load-
bearing predicate.
"""

from __future__ import annotations

import inspect

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
    """Construct the legacy LEADER response wire bytes: 8-byte header
    + NUL-terminated address padded to 8-byte boundary. The legacy
    body has no ``node_id`` prefix (that field was added in V1)."""
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
    """A ``ClientRequest`` carries a uint64 ``client_id`` body and
    msg_type=``RequestType.CLIENT`` which equals 1 — the same numeric
    code as ``ResponseType.LEADER``. A request-side decoder must
    route this through ``ClientRequest.decode_body``, NOT through
    ``LeaderResponse.decode_body_legacy``."""
    return ClientRequest(client_id=client_id).encode()


def test_legacy_dispatch_fires_on_response_side_legacy_version() -> None:
    """Positive twin: response-side decoder + LEGACY version + LEADER
    msg_type routes through ``decode_body_legacy``. The legacy body
    has no ``node_id`` prefix; the decoder synthesises ``node_id=0``."""
    addr = "leader.example.com:9001"
    wire = _build_legacy_leader_response_bytes(addr)

    result = decode_message(wire, is_request=False, version=PROTOCOL_VERSION_LEGACY)

    assert isinstance(result, LeaderResponse)
    assert result.address == addr
    # ``decode_body_legacy`` stamps node_id=0 (the field was added in
    # the V1 schema). A non-zero value would mean the V1 ``decode_body``
    # accidentally ran on legacy bytes, mis-parsing the uint64 of the
    # address as a node_id.
    assert result.node_id == 0


def test_legacy_dispatch_does_not_fire_on_request_side_client_msg_type() -> None:
    """Negative twin: a request-side decoder seeing ``msg_type=1`` (the
    numeric collision: ``RequestType.CLIENT == 1 == ResponseType.LEADER``)
    must NOT route through ``LeaderResponse.decode_body_legacy``. The
    direction guard ``not self._is_request`` is what enforces this —
    without it, a request-side decoder would mis-parse a ``ClientRequest``
    body (uint64 client_id) as a legacy LEADER address (NUL-terminated
    string), producing a malformed LeaderResponse with random-looking
    address bytes from the client_id integer.

    Even passing ``version=PROTOCOL_VERSION_LEGACY`` to a request-side
    decoder must not change the dispatch — request-side traffic is
    always V1 in practice but the runtime guard must protect against
    the misuse regardless of the version argument."""
    wire = _build_client_request_bytes(client_id=0x1234567812345678)

    result = decode_message(wire, is_request=True, version=PROTOCOL_VERSION_LEGACY)

    assert isinstance(result, ClientRequest), (
        f"request-side decoder must route msg_type=1 to ClientRequest, "
        f"not legacy LeaderResponse — got {type(result).__name__}"
    )
    assert not isinstance(result, LeaderResponse)
    assert result.client_id == 0x1234567812345678


def test_v1_leader_response_does_not_route_through_legacy_decode() -> None:
    """Negative version-guard pin: a V1 LEADER response (carrying the
    ``node_id`` uint64 prefix) must route through V1 ``decode_body``,
    NOT ``decode_body_legacy``. The version predicate
    ``self._version == PROTOCOL_VERSION_LEGACY`` in the legacy
    dispatch arm is load-bearing for V1 traffic; a regression that
    drops it would mis-parse the first 8 bytes (node_id) as the start
    of the NUL-terminated address string. Third leg of the dispatch
    test triangle (the other two cover the positive LEGACY arm and
    the direction-negative ClientRequest arm)."""
    from dqlitewire.codec import PROTOCOL_VERSION

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
    """Documentation pin: the entire point of the direction guard is
    the numeric collision. If a future refactor renumbers either type
    to disambiguate, the guard becomes redundant — this assertion
    would fail and the dev would re-evaluate whether the guard still
    serves its purpose."""
    assert RequestType.CLIENT == ResponseType.LEADER == 1
