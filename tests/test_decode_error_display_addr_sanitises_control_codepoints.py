"""Pin: the malformed ``(node_id, address)`` ``DecodeError`` in both
``LeaderResponse`` and ``ServersResponse`` routes the address through
``sanitize_server_text`` before truncation and ``!r`` rendering. ``repr``
escapes ``\\n``/``\\r``/``\\t`` but not the broader line-separator/bidi/
zero-width class, so a hostile single codepoint could otherwise inject a
line-split into journald/SIEM ingest (truncation does not help — one
codepoint fits)."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import LeaderResponse, ServersResponse
from dqlitewire.types import encode_text, encode_uint64


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
