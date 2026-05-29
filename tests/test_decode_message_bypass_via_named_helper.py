"""``decode_message`` routes its handshake bypass through the named
``MessageDecoder._force_handshake_for_stateless`` rather than writing
``_handshake_done`` / ``_version`` directly, so a future property setter
or observability hook isn't silently bypassed.
"""

from __future__ import annotations

import inspect

import pytest

from dqlitewire.codec import MessageDecoder, decode_message, encode_message
from dqlitewire.constants import PROTOCOL_VERSION
from dqlitewire.exceptions import HandshakeError
from dqlitewire.messages.requests import LeaderRequest


def test_decode_message_does_not_write_handshake_done_directly() -> None:
    src = inspect.getsource(decode_message)
    assert "decoder._handshake_done = True" not in src
    assert "decoder._version = version" not in src
    assert "_force_handshake_for_stateless" in src


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
