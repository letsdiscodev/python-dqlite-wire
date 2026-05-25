"""Pin: the stateless ``decode_message`` helper routes its handshake
bypass through the named ``MessageDecoder._force_handshake_for_stateless``
method rather than writing ``_handshake_done`` / ``_version`` directly.

The direct private-attr write is mechanically equivalent today but
would silently bypass any future property setter or observability
hook added to the handshake state machine. Centralising the bypass
on a single named method gives a forward-compat anchor.
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
    """Defense-in-depth re-validation: a future refactor that drops the
    constructor's version check must not silently smuggle bogus values
    past the bypass helper."""
    decoder = MessageDecoder(is_request=True)
    with pytest.raises(HandshakeError, match="Unsupported protocol version"):
        decoder._force_handshake_for_stateless(0xDEADBEEF)


def test_decode_message_round_trip_request_unchanged() -> None:
    """Behaviour-preservation: a round-tripped request decodes equal to
    its encoded form (the bypass is mechanically equivalent)."""
    msg = LeaderRequest()
    wire = encode_message(msg)
    decoded = decode_message(wire, is_request=True, version=PROTOCOL_VERSION)
    assert decoded == msg
