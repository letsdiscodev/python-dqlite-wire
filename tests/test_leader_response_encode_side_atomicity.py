"""Pin: ``LeaderResponse.encode_body`` rejects the
``(node_id=0, address != "")`` shape so a caller-built malformed
value can't emit bytes the same package's ``decode_body`` will
reject.

The raft-leader atomicity invariant (``dqlite-upstream/src/gateway.
c::handle_leader``) emits two shapes only:

  * ``(0, "")``  — canonical "no leader known" sentinel.
  * ``(N, addr)`` for ``N >= 1`` — known leader.

A ``RAFT_NOMEM`` transient on the follower's leader-update path can
ALSO produce ``(N, "")`` — the decoder accepts that, logs a warning,
and treats it as "leader unknown" (see ``LeaderResponse.decode_
body``'s ``node_id != 0 and not address`` branch).

The ``(0, non-empty)`` shape was rejected by ``decode_body`` only —
``encode_body`` had no symmetric guard, so:

    LeaderResponse(node_id=0, address="evil:9001").encode_body()

produced valid wire bytes that the same package's ``decode_message``
then refused. Mock-server / proxy / golden-byte authors got values
that round-tripped in neither direction.

The check lives on ``encode_body`` (not ``__post_init__``) because
``decode_body_legacy`` legitimately constructs ``LeaderResponse(0,
addr)`` from the legacy wire shape — the legacy body carries only
the address, with ``node_id`` hard-coded 0 on decode. The legacy
``encode_body_legacy`` already has its own symmetric strict check
(non-zero ``node_id`` rejected); the modern ``encode_body`` now
mirrors that "encode-time strict" pattern.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.messages.responses import LeaderResponse


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
