"""Pin: a foreign-encoded request with EMPTY ``params`` round-trips
byte-identically through Python's decoder/encoder for BOTH valid
wire shapes:

1. **Go-style** (no params tuple bytes at all) — Go's
   ``putNamedValues`` writes ZERO bytes for empty params. An
   ``ExecRequest`` body is then 8 bytes (db_id + stmt_id).
2. **C-style** (explicit 8-byte zero header) — C's
   ``tuple_encoder__init`` writes the count byte (V0) or count
   word (V1), then pads to the next word boundary. Body is 16
   bytes.

The previous fix (ISSUE-842) made the C-style → C-style direction
byte-identical by adding ``emit_empty_header=True`` when
``_decoded_schema is not None``. That fix inverted the asymmetry:
Go-encoded inputs then re-emitted as 16 bytes (the C shape). The
``_decoded_empty_header`` companion field distinguishes the two
decoded states so the encoder picks the shape the wire input
actually carried.

Mock-server harnesses, proxies, and golden-byte cross-language
conformance tests are the audience for the ``_decoded_schema`` /
``_decoded_empty_header`` round-trip hint.
"""

from __future__ import annotations

import pytest

from dqlitewire.messages.requests import (
    ExecRequest,
    ExecSqlRequest,
    QueryRequest,
    QuerySqlRequest,
)


@pytest.mark.parametrize("schema", [0, 1])
@pytest.mark.parametrize("RequestCls", [ExecRequest, QueryRequest])
def test_empty_params_go_style_roundtrip_byte_identical(
    schema: int, RequestCls: type[ExecRequest] | type[QueryRequest]
) -> None:
    """A Go-encoded EXEC/QUERY with empty params is 8 bytes
    (db_id + stmt_id, no tuple bytes); decoding then re-encoding must
    NOT add a fake 8-byte empty-params header."""
    data = (1).to_bytes(4, "little") + (2).to_bytes(4, "little")
    req = RequestCls.decode_body(data, schema=schema)
    assert list(req.params) == []
    assert req.encode_body() == data, (
        f"Go-style empty-params {RequestCls.__name__} (schema={schema}) "
        f"round-trip is not byte-identical: got {len(req.encode_body())} bytes, "
        f"expected {len(data)}"
    )


@pytest.mark.parametrize("schema", [0, 1])
@pytest.mark.parametrize("RequestCls", [ExecRequest, QueryRequest])
def test_empty_params_c_style_roundtrip_byte_identical(
    schema: int, RequestCls: type[ExecRequest] | type[QueryRequest]
) -> None:
    """A C-encoded EXEC/QUERY with empty params is 16 bytes
    (db_id + stmt_id + 8-byte zero header); the existing
    ``_decoded_schema`` round-trip discipline keeps the explicit
    header on re-encode."""
    data = (1).to_bytes(4, "little") + (2).to_bytes(4, "little") + b"\x00" * 8
    req = RequestCls.decode_body(data, schema=schema)
    assert list(req.params) == []
    assert req.encode_body() == data


@pytest.mark.parametrize("schema", [0, 1])
@pytest.mark.parametrize("RequestCls", [ExecSqlRequest, QuerySqlRequest])
def test_sql_empty_params_go_style_roundtrip_byte_identical(
    schema: int,
    RequestCls: type[ExecSqlRequest] | type[QuerySqlRequest],
) -> None:
    """SQL variant Go-style: db_id (8) + sql_text (padded) +
    0 bytes for empty params."""
    # SQL "x" + NUL terminator = 2 bytes; padded to word = 8 bytes.
    sql_bytes = b"x\x00" + b"\x00" * 6
    data = (1).to_bytes(8, "little") + sql_bytes
    req = RequestCls.decode_body(data, schema=schema)
    assert list(req.params) == []
    assert req.sql == "x"
    assert req.encode_body() == data


@pytest.mark.parametrize("schema", [0, 1])
@pytest.mark.parametrize("RequestCls", [ExecSqlRequest, QuerySqlRequest])
def test_sql_empty_params_c_style_roundtrip_byte_identical(
    schema: int,
    RequestCls: type[ExecSqlRequest] | type[QuerySqlRequest],
) -> None:
    """SQL variant C-style: db_id (8) + sql_text (padded) +
    8-byte zero header for empty params."""
    sql_bytes = b"x\x00" + b"\x00" * 6
    data = (1).to_bytes(8, "little") + sql_bytes + b"\x00" * 8
    req = RequestCls.decode_body(data, schema=schema)
    assert list(req.params) == []
    assert req.sql == "x"
    assert req.encode_body() == data


def test_caller_originated_empty_params_emits_go_style() -> None:
    """When a caller constructs ``ExecRequest(db_id, stmt_id)`` (no
    decode), the default ``_decoded_empty_header=None`` keeps the
    Go-style omission — the wire-friendly default for new requests."""
    req = ExecRequest(db_id=1, stmt_id=2, params=[])
    body = req.encode_body()
    # 8 bytes: db_id (4) + stmt_id (4), no tuple bytes.
    assert len(body) == 8
    assert body == (1).to_bytes(4, "little") + (2).to_bytes(4, "little")
