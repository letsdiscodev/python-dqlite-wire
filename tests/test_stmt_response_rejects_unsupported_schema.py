"""StmtResponse.decode_body rejects schema outside {0, 1} (defense-in-depth
companion to the codec dispatch-table cap, for direct decode_body callers)."""

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
    """schema=2 is undefined upstream (only V0/V1 exist), so reject it."""
    body = _body_schema1()
    with pytest.raises(DecodeError, match="unsupported schema"):
        StmtResponse.decode_body(body, schema=2)


def test_decode_body_rejects_negative_schema() -> None:
    body = _body_schema1()
    with pytest.raises(DecodeError, match="unsupported schema"):
        StmtResponse.decode_body(body, schema=-1)


def test_decode_body_accepts_schema_zero() -> None:
    body = b"\x01\x00\x00\x00" + b"\x02\x00\x00\x00" + b"\x00" * 8
    msg = StmtResponse.decode_body(body, schema=0)
    assert msg.db_id == 1
    assert msg.stmt_id == 2


def test_decode_body_accepts_schema_one() -> None:
    body = _body_schema1(tail_offset=42)
    msg = StmtResponse.decode_body(body, schema=1)
    assert msg.db_id == 1
    assert msg.stmt_id == 2
    assert msg.tail_offset == 42
