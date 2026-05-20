"""Pin: ``_decoded_empty_header`` is validated at construction so the
field stays in sync with the wire-byte semantics it represents.

The documented contract says the flag is "Set by ``decode_body`` only
when ``params`` is empty AND the inbound frame had bytes for the
tuple". A caller constructing a request with
``_decoded_empty_header=True`` and non-empty params produces an
internally-inconsistent state — the encoder's
``emit_empty_header=True`` arm is currently a no-op when params is
non-empty, but a future refactor that broadens the arm to "always
emit the 8-byte header" would produce malformed wire bytes only on
this misuse pattern.

Mirrors the sibling ``_decoded_schema`` validator's discipline.
"""

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
    """``_decoded_empty_header=True`` only makes sense with empty
    params (C-style 8-byte zero header). The validator must reject
    the inconsistent combination at construction."""
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
    """The legitimate use of the flag: ``params=[]`` AND
    ``_decoded_empty_header=True`` (round-trip a C-style frame)."""
    req = RequestCls(**kwargs, params=[], _decoded_empty_header=True)
    # Field is excluded from compare/repr; access directly.
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
    """``False`` is the documented "decoded a Go-style frame" state
    and is admissible with any param count — the field only
    discriminates between the two valid empty-params wire shapes."""
    req = RequestCls(**kwargs, params=[42, 43], _decoded_empty_header=False)
    assert req._decoded_empty_header is False


def test_caller_originated_default_is_none() -> None:
    """``None`` is the caller-originated default → Go-style omission.
    A future refactor changing this to ``False`` would lose the
    discriminator between "decoded a Go-style frame" and "never
    decoded"."""
    req = ExecRequest(db_id=1, stmt_id=2)
    assert req._decoded_empty_header is None
