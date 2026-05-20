"""Pin: ``PrepareRequest`` / ``ExecSqlRequest`` / ``QuerySqlRequest``
reject non-``str`` ``sql`` at construction (``EncodeError``) rather
than at ``encode_body()``. Plus: the encoded label diagnostics name
the field ("SQL") rather than the generic "Text".
"""

from __future__ import annotations

import pytest

from dqlitewire import DecodeError, EncodeError
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


@pytest.mark.parametrize(
    "cls",
    [PrepareRequest, ExecSqlRequest, QuerySqlRequest],
)
def test_sql_decode_oversize_error_names_field(cls: type) -> None:
    """Decode-side cap diagnostics carry the ``SQL`` label, symmetric
    with the encode side, so an operator reading a wire capture's
    ``DecodeError`` sees which field overflowed without walking the
    traceback. Regression-resistant against a refactor that drops
    ``label="SQL"`` (or default-restores ``max_size``) on the decode
    side — that change would be silent today (only encode-side has
    a pin)."""
    from dqlitewire.types import _MAX_TEXT_VALUE_SIZE

    # Body shape: uint64 db_id + text payload (NUL-terminated, padded
    # to 8-byte boundary). Build a payload longer than the cap that
    # still contains a NUL at the end so the decoder hits the
    # size-cap branch (not the unterminated branch).
    db_id_bytes = (0).to_bytes(8, "little")
    huge_text = b"X" * (_MAX_TEXT_VALUE_SIZE + 8)
    payload = huge_text + b"\x00"
    pad = (-len(payload)) % 8
    body = db_id_bytes + payload + b"\x00" * pad

    with pytest.raises(DecodeError, match="SQL"):
        cls.decode_body(body)


@pytest.mark.parametrize(
    "cls",
    [PrepareRequest, ExecSqlRequest, QuerySqlRequest],
)
def test_sql_decode_unterminated_error_names_field(cls: type) -> None:
    """Decode-side null-termination diagnostic also carries the SQL
    label so a truncated payload's error message is field-specific
    rather than the generic ``"Text not null-terminated"``."""
    db_id_bytes = (0).to_bytes(8, "little")
    sql_bytes = b"SELECT 1 FROM t"
    pad = (-len(db_id_bytes + sql_bytes)) % 8
    body = db_id_bytes + sql_bytes + b"Y" * pad  # no NUL anywhere

    with pytest.raises(DecodeError, match="SQL"):
        cls.decode_body(body)
