"""``encode_value(.., ISO8601)`` validates via datetime/time.fromisoformat before emitting bytes."""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


def test_iso8601_rejects_non_iso_string() -> None:
    with pytest.raises(EncodeError) as exc_info:
        encode_value("not-an-iso-date", ValueType.ISO8601)
    msg = str(exc_info.value)
    assert "ISO8601" in msg
    assert "TEXT" in msg or ".isoformat()" in msg


def test_iso8601_accepts_full_datetime() -> None:
    encoded, vtype = encode_value("2024-01-15T12:30:45", ValueType.ISO8601)
    assert vtype == ValueType.ISO8601
    assert len(encoded) > 0


def test_iso8601_accepts_date_only() -> None:
    encoded, vtype = encode_value("2024-01-15", ValueType.ISO8601)
    assert vtype == ValueType.ISO8601


def test_iso8601_accepts_bare_time_via_fallback() -> None:
    """A bare time fails datetime.fromisoformat but passes time.fromisoformat."""
    encoded, vtype = encode_value("12:30:45", ValueType.ISO8601)
    assert vtype == ValueType.ISO8601


def test_iso8601_accepts_aware_datetime() -> None:
    encoded, vtype = encode_value("2024-01-15T12:30:45+00:00", ValueType.ISO8601)
    assert vtype == ValueType.ISO8601


def test_text_with_non_iso_string_still_succeeds() -> None:
    """Validation applies only to ISO8601, not plain TEXT."""
    encoded, vtype = encode_value("not-an-iso-date", ValueType.TEXT)
    assert vtype == ValueType.TEXT
    assert len(encoded) > 0
