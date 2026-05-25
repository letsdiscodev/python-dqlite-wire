"""Pin: every response-side ``decode_body`` validates the ``schema``
kwarg at the per-class layer, mirroring the request-side discipline.

The production dispatcher (``MessageDecoder.decode_bytes``) caps
``schema`` at ``_RESPONSE_MAX_SCHEMA[type]`` before calling per-class
decoders, but direct callers (tests, proxies, fuzzers,
golden-byte harnesses) bypass the dispatcher. Until this fix the
per-class layer silently accepted bogus schemas — request decoders
have always rejected. Pin the symmetric defense-in-depth.
"""

from __future__ import annotations

import struct

import pytest

from dqlitewire.codec import RESPONSE_TYPES
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import LeaderResponse, StmtResponse


# Minimally-valid body per response class so the bogus-schema reject
# fires before any body-shape decode would raise a different error.
def _body_for(cls: type) -> bytes:
    if cls is LeaderResponse:
        # node_id=0, address="" + padding = 8 + 8 bytes.
        return struct.pack("<Q", 0) + b"\x00" * 8
    # Most decoders accept a string of NULs as a degenerate body.
    return b"\x00" * 64


@pytest.mark.parametrize("msg_cls", list(RESPONSE_TYPES.values()))
def test_response_decode_body_rejects_unknown_schema(msg_cls: type) -> None:
    body = _body_for(msg_cls)
    with pytest.raises(DecodeError, match="(?i)schema"):
        msg_cls.decode_body(body, schema=99)  # type: ignore[attr-defined]


def test_stmt_response_v1_schema_still_accepted() -> None:
    """Regression: ``StmtResponse`` already accepted schema=1 (V1)
    and continues to do so — the broad reject is "unknown schema",
    not "non-zero schema"."""
    body = struct.pack("<II", 0, 1) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    msg = StmtResponse.decode_body(body, schema=1)
    assert msg.tail_offset == 0
