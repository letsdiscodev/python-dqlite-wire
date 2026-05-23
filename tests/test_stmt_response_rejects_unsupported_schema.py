"""``StmtResponse.decode_body`` should reject schema values outside
the documented set ``{0, 1}`` — matching the upstream protocol
constants ``DQLITE_PREPARE_STMT_SCHEMA_V0=0`` and
``DQLITE_PREPARE_STMT_SCHEMA_V1=1``.

The dispatch table in ``codec.py::_RESPONSE_MAX_SCHEMA`` already caps
inbound headers at ``1`` for ``STMT``; this is the defense-in-depth
companion at the message layer so direct callers of ``decode_body``
(tests, future refactors) see the same strict-decode posture as the
sibling decoders (``ExecRequest``, ``QueryRequest``, ``PrepareRequest``,
``AssignRequest``).
"""

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import StmtResponse


def _body_schema1(tail_offset: int = 0) -> bytes:
    """Build a syntactically valid V1 body (24 bytes)."""
    db_id = b"\x01\x00\x00\x00"
    stmt_id = b"\x02\x00\x00\x00"
    num_params = b"\x00" * 8
    tail = tail_offset.to_bytes(8, "little")
    return db_id + stmt_id + num_params + tail


def test_decode_body_rejects_schema_two() -> None:
    """``schema=2`` is not a defined STMT schema — upstream
    ``protocol.h`` defines only V0=0 and V1=1, and
    ``gateway.c::handle_prepare`` itself rejects anything else. The
    message-layer decoder should match that posture."""
    body = _body_schema1()
    with pytest.raises(DecodeError, match="unsupported schema"):
        StmtResponse.decode_body(body, schema=2)


def test_decode_body_rejects_negative_schema() -> None:
    """Defence against a misused caller passing a negative int."""
    body = _body_schema1()
    with pytest.raises(DecodeError, match="unsupported schema"):
        StmtResponse.decode_body(body, schema=-1)


def test_decode_body_accepts_schema_zero() -> None:
    """Sanity: V0 still decodes."""
    body = b"\x01\x00\x00\x00" + b"\x02\x00\x00\x00" + b"\x00" * 8
    msg = StmtResponse.decode_body(body, schema=0)
    assert msg.db_id == 1
    assert msg.stmt_id == 2


def test_decode_body_accepts_schema_one() -> None:
    """Sanity: V1 still decodes."""
    body = _body_schema1(tail_offset=42)
    msg = StmtResponse.decode_body(body, schema=1)
    assert msg.db_id == 1
    assert msg.stmt_id == 2
    assert msg.tail_offset == 42
