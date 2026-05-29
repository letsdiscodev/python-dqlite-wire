"""_decoded_empty_header is validated at construction to match its wire-byte semantics."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.requests import (
    ExecRequest,
    ExecSqlRequest,
    QueryRequest,
    QuerySqlRequest,
)


@pytest.mark.parametrize(
    "RequestCls,kwargs",
    [
        (ExecRequest, {"db_id": 1, "stmt_id": 2}),
        (QueryRequest, {"db_id": 1, "stmt_id": 2}),
        (ExecSqlRequest, {"db_id": 1, "sql": "x"}),
        (QuerySqlRequest, {"db_id": 1, "sql": "x"}),
    ],
)
def test_decoded_empty_header_true_with_non_empty_params_rejected(
    RequestCls: type, kwargs: dict[str, object]
) -> None:
    """_decoded_empty_header=True only makes sense with empty params; bad combos rejected."""
    with pytest.raises(EncodeError, match="_decoded_empty_header"):
        RequestCls(**kwargs, params=[42], _decoded_empty_header=True)


@pytest.mark.parametrize(
    "RequestCls,kwargs",
    [
        (ExecRequest, {"db_id": 1, "stmt_id": 2}),
        (QueryRequest, {"db_id": 1, "stmt_id": 2}),
        (ExecSqlRequest, {"db_id": 1, "sql": "x"}),
        (QuerySqlRequest, {"db_id": 1, "sql": "x"}),
    ],
)
def test_decoded_empty_header_true_with_empty_params_accepted(
    RequestCls: type, kwargs: dict[str, object]
) -> None:
    """Legitimate use: params=[] AND _decoded_empty_header=True (round-trip a C-style frame)."""
    req = RequestCls(**kwargs, params=[], _decoded_empty_header=True)
    assert req._decoded_empty_header is True


@pytest.mark.parametrize(
    "RequestCls,kwargs",
    [
        (ExecRequest, {"db_id": 1, "stmt_id": 2}),
        (QueryRequest, {"db_id": 1, "stmt_id": 2}),
    ],
)
def test_decoded_empty_header_false_with_non_empty_params_accepted(
    RequestCls: type, kwargs: dict[str, object]
) -> None:
    """False ("decoded a Go-style frame") is admissible with any param count."""
    req = RequestCls(**kwargs, params=[42, 43], _decoded_empty_header=False)
    assert req._decoded_empty_header is False


def test_caller_originated_default_is_none() -> None:
    """None is the caller-originated default; False would lose the decoded-vs-never bit."""
    req = ExecRequest(db_id=1, stmt_id=2)
    assert req._decoded_empty_header is None


@pytest.mark.parametrize("bad_value", [1, 0, "yes", "", object(), [], (1,)])
def test_decoded_empty_header_non_bool_rejected(bad_value: object) -> None:
    """Reject non-bool/non-None inputs: else bool() at encode time promotes truthy values."""
    with pytest.raises(EncodeError, match="_decoded_empty_header must be None or bool"):
        ExecRequest(db_id=1, stmt_id=2, params=[], _decoded_empty_header=bad_value)  # type: ignore[arg-type]


def test_decoded_empty_header_none_accepted_explicitly() -> None:
    req = ExecRequest(db_id=1, stmt_id=2, params=[], _decoded_empty_header=None)
    assert req._decoded_empty_header is None
