"""Pin: ``PrepareRequest`` / ``ExecSqlRequest`` / ``QuerySqlRequest``
reject non-``str`` ``sql`` at construction (``EncodeError``) rather
than at ``encode_body()``. Plus: the encoded label diagnostics name
the field ("SQL") rather than the generic "Text".
"""

from __future__ import annotations

import pytest

from dqlitewire import EncodeError
from dqlitewire.messages.requests import (
    ExecSqlRequest,
    PrepareRequest,
    QuerySqlRequest,
)


@pytest.mark.parametrize(
    "cls",
    [PrepareRequest, ExecSqlRequest, QuerySqlRequest],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        b"SELECT 1",  # bytes
        123,  # int
        None,  # None
        memoryview(b"SELECT 1"),  # memoryview
    ],
)
def test_sql_field_must_be_str_at_construction(cls: type, bad_value: object) -> None:
    with pytest.raises(EncodeError, match="sql must be str"):
        cls(db_id=0, sql=bad_value)


@pytest.mark.parametrize(
    "cls",
    [PrepareRequest, ExecSqlRequest, QuerySqlRequest],
)
def test_sql_encode_oversize_error_names_field(cls: type) -> None:
    """Encode-side cap diagnostics carry the ``SQL`` label so an
    operator triaging a wire capture knows which field overflowed
    without walking the traceback."""
    from dqlitewire.types import _MAX_TEXT_VALUE_SIZE

    huge_sql = "X" * (_MAX_TEXT_VALUE_SIZE + 1)
    req = cls(db_id=0, sql=huge_sql)
    with pytest.raises(EncodeError, match="SQL"):
        req.encode_body()
