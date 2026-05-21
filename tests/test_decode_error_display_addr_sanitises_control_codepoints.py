"""Pin: the malformed ``(node_id, address)`` ``DecodeError`` in
both ``LeaderResponse.decode_body`` and ``ServersResponse.decode_body``
routes the address through ``sanitize_server_text`` BEFORE the
64-char truncation and the ``!r`` rendering.

``repr`` escapes ``\\n`` / ``\\r`` / ``\\t`` but does NOT escape the
broader line-separator class (U+2028 LINE SEPARATOR, U+2029 PARAGRAPH
SEPARATOR), bidi marks (U+202A..E), zero-width characters
(U+200B..D, U+FEFF). A hostile peer crafting a malformed atomicity
entry with one of those codepoints in ``address`` would otherwise
land a line-split in ``journald`` / SIEM ingest via the
``DecodeError`` text — the 64-char truncation does not help because
a single codepoint fits.

``sanitize_server_text`` replaces the broader class with ``?`` so
the ``DecodeError`` text stays in a single forensically-recoverable
line.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import LeaderResponse, ServersResponse
from dqlitewire.types import encode_text, encode_uint64


def test_leader_response_malformed_decode_error_sanitises_u2028() -> None:
    """U+2028 in the address survives the 64-char cap but must NOT
    survive the sanitiser. ``str(DecodeError)`` cannot contain a
    raw line-separator."""
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
    """U+202E RIGHT-TO-LEFT OVERRIDE in the address must be sanitised
    so an attacker cannot reorder the diagnostic text shown in
    operator log viewers."""
    forged = "leader‮FORGED:9001"
    body = encode_uint64(0) + encode_text(forged)
    with pytest.raises(DecodeError) as exc_info:
        LeaderResponse.decode_body(body)
    msg = str(exc_info.value)
    assert "‮" not in msg


def test_servers_response_malformed_decode_error_sanitises_u2028() -> None:
    """Sibling pin for ``ServersResponse``'s per-entry atomicity
    check. The two decoders share the diagnostic shape; both must
    route through ``sanitize_server_text``."""
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
    """Belt-and-suspenders: a malformed address just over the 64-char
    cap (no sanitisable codepoints) still gets the display truncation
    suffix. Sanitising before truncation must not bypass the cap.

    The input length is pinned at ``cap + 1`` (65 chars) rather than an
    arbitrary 200: the assertion below verifies the rendered address is
    shorter than the input, which is the truncation invariant. A future
    cap raise would surface as a deliberate update to this single
    length literal, not as a misread "sanitisation regression" against
    an over-long forged input.
    """
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
    """Sibling pin for ``ServersResponse``. See sibling docstring above
    for the ``cap + 1`` rationale."""
    forged = "B" * 65
    body = encode_uint64(1) + encode_uint64(0) + encode_text(forged) + encode_uint64(0)
    with pytest.raises(DecodeError) as exc_info:
        ServersResponse.decode_body(body)
    msg = str(exc_info.value)
    assert "…" in msg
    assert forged not in msg
