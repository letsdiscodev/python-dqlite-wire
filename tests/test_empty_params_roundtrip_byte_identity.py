"""A foreign-encoded request with EMPTY ``params`` must round-trip
byte-identically for both valid wire shapes: Go-style (no tuple bytes, body 8
bytes) and C-style (explicit 8-byte zero header, body 16 bytes). The
``_decoded_empty_header`` field records which shape the input carried so the
encoder re-emits the same one."""

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
    """A Go-encoded EXEC/QUERY with empty params (8 bytes) must not gain a fake
    8-byte empty-params header on re-encode."""
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
    """A C-encoded EXEC/QUERY with empty params (16 bytes) must keep the
    explicit zero header on re-encode."""
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
    # SQL "x" + NUL = 2 bytes, padded to an 8-byte word; no params bytes.
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
    sql_bytes = b"x\x00" + b"\x00" * 6
    data = (1).to_bytes(8, "little") + sql_bytes + b"\x00" * 8
    req = RequestCls.decode_body(data, schema=schema)
    assert list(req.params) == []
    assert req.sql == "x"
    assert req.encode_body() == data


def test_caller_originated_empty_params_emits_go_style() -> None:
    """A caller-constructed request (no decode) defaults to the Go-style
    omission via ``_decoded_empty_header=None``."""
    req = ExecRequest(db_id=1, stmt_id=2, params=[])
    body = req.encode_body()
    assert len(body) == 8
    assert body == (1).to_bytes(4, "little") + (2).to_bytes(4, "little")
