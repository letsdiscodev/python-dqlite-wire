"""_decoded_schema hint gives byte-identical re-emission of foreign-encoded empty-params bodies."""

from __future__ import annotations

import pytest

from dqlitewire.messages.requests import (
    ExecRequest,
    ExecSqlRequest,
    QueryRequest,
    QuerySqlRequest,
)


@pytest.mark.parametrize("schema", [0, 1])
def test_exec_request_empty_params_byte_identical(schema: int) -> None:
    db_id, stmt_id = 1, 2
    body = db_id.to_bytes(4, "little") + stmt_id.to_bytes(4, "little") + b"\x00" * 8
    req = ExecRequest.decode_body(body, schema=schema)
    assert req.params == []
    assert req._decoded_schema == schema
    assert req.encode_body() == body, "round-trip not byte-identical"


@pytest.mark.parametrize("schema", [0, 1])
def test_query_request_empty_params_byte_identical(schema: int) -> None:
    db_id, stmt_id = 1, 2
    body = db_id.to_bytes(4, "little") + stmt_id.to_bytes(4, "little") + b"\x00" * 8
    req = QueryRequest.decode_body(body, schema=schema)
    assert req.params == []
    assert req._decoded_schema == schema
    assert req.encode_body() == body


@pytest.mark.parametrize("schema", [0, 1])
def test_exec_sql_request_empty_params_byte_identical(schema: int) -> None:
    """ExecSqlRequest body: db_id (8) + sql (text + padding) + params."""
    db_id = 1
    from dqlitewire.types import encode_text

    body = db_id.to_bytes(8, "little") + encode_text("SELECT 1") + b"\x00" * 8
    req = ExecSqlRequest.decode_body(body, schema=schema)
    assert req.params == []
    assert req._decoded_schema == schema
    assert req.encode_body() == body


@pytest.mark.parametrize("schema", [0, 1])
def test_query_sql_request_empty_params_byte_identical(schema: int) -> None:
    from dqlitewire.types import encode_text

    db_id = 1
    body = db_id.to_bytes(8, "little") + encode_text("SELECT 1") + b"\x00" * 8
    req = QuerySqlRequest.decode_body(body, schema=schema)
    assert req.params == []
    assert req._decoded_schema == schema
    assert req.encode_body() == body


def test_self_originated_empty_params_still_zero_bytes() -> None:
    """Self-originated requests (no _decoded_schema hint) keep the Go-style 0-byte empty-params."""
    req = ExecRequest(db_id=1, stmt_id=2, params=[])
    assert req._decoded_schema is None
    encoded = req.encode_body()
    assert len(encoded) == 8
