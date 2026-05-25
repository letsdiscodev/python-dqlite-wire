"""Pin: ``WelcomeResponse.decode_body`` emits a ``logger.warning``
when ``heartbeat_timeout == 0``.

The docstring explicitly calls a zero heartbeat "semantically
ambiguous" / "misconfigured peer or non-conforming server" — that
diagnostic content used to live only in source comments, so operators
running a dqlite cluster with a misconfigured peer got no log signal.

The wire layer keeps its permissive-accept contract (the decoder
still returns the response; no DecodeError raised) but emits a
single warning that surfaces the docstring's diagnostic content into
the log stream. Aligns with the in-tree ``ServersResponse.decode_body``
``unknown_role_policy="warn"`` precedent.
"""

from __future__ import annotations

import logging

import pytest

from dqlitewire.messages.responses import WelcomeResponse
from dqlitewire.types import encode_uint64


def test_decode_body_zero_heartbeat_emits_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A zero heartbeat is accepted but produces a single
    logger.warning at decode time."""
    body = encode_uint64(0)
    with caplog.at_level(logging.WARNING, logger="dqlitewire.messages.responses"):
        resp = WelcomeResponse.decode_body(body)
    assert resp.heartbeat_timeout == 0
    # Single warning emitted.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].message
    # Diagnostic content from the docstring surfaces in the log line.
    assert "heartbeat_timeout=0" in msg or "heartbeat" in msg.lower()
    assert "15000" in msg or "non-conforming" in msg.lower() or "misconfig" in msg.lower()


def test_decode_body_default_heartbeat_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A legitimate heartbeat (e.g. 15000ms upstream default) does
    NOT trigger the warning."""
    body = encode_uint64(15000)
    with caplog.at_level(logging.WARNING, logger="dqlitewire.messages.responses"):
        resp = WelcomeResponse.decode_body(body)
    assert resp.heartbeat_timeout == 15000
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_decode_body_zero_heartbeat_is_still_accepted() -> None:
    """The warning is observability-only — the decoder still returns
    a valid WelcomeResponse with heartbeat_timeout=0 (preserves the
    documented permissive-accept contract)."""
    body = encode_uint64(0)
    resp = WelcomeResponse.decode_body(body)
    assert resp.heartbeat_timeout == 0
    assert resp.heartbeat_timeout_seconds == 0.0
