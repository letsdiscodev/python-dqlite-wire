"""``decode_message``'s stateless handshake bypass re-validates the version
and leaves a normal request round-trip unchanged.
"""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder, decode_message, encode_message
from dqlitewire.constants import PROTOCOL_VERSION
from dqlitewire.exceptions import HandshakeError
from dqlitewire.messages.requests import LeaderRequest


def test_force_handshake_for_stateless_rejects_unsupported_version() -> None:
    """The bypass helper re-validates the version even if the constructor's
    check is later dropped."""
    decoder = MessageDecoder(is_request=True)
    with pytest.raises(HandshakeError, match="Unsupported protocol version"):
        decoder._force_handshake_for_stateless(0xDEADBEEF)


def test_decode_message_round_trip_request_unchanged() -> None:
    msg = LeaderRequest()
    wire = encode_message(msg)
    decoded = decode_message(wire, is_request=True, version=PROTOCOL_VERSION)
    assert decoded == msg
