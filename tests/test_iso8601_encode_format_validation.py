"""Pin: ``encode_value(value, ValueType.ISO8601)`` validates the
string via ``datetime.fromisoformat`` / ``time.fromisoformat`` BEFORE
emitting wire bytes.

Without this, ``encode_value("not-an-iso-date", ValueType.ISO8601)``
silently produces a wire cell byte-identical to a TEXT cell of the
same string, and the failure surfaces hours later on the consumer
side (e.g. ``dqlitedbapi`` calling ``datetime.fromisoformat``).
Surface the failure at the bind site instead.

The probe ladder mirrors the consumer's parser pair: try
``datetime.fromisoformat`` first, fall through to
``time.fromisoformat`` so bare-time strings (``"12:30:45"``) that the
consumer's time arm accepts still pass.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


def test_iso8601_rejects_non_iso_string() -> None:
    """Encoding a non-ISO string as ISO8601 must raise EncodeError
    with the actionable hint suggesting ValueType.TEXT or .isoformat()
    coercion."""
    with pytest.raises(EncodeError) as exc_info:
        encode_value("not-an-iso-date", ValueType.ISO8601)
    msg = str(exc_info.value)
    assert "ISO8601" in msg
    # Actionable hint: suggest TEXT or .isoformat().
    assert "TEXT" in msg or ".isoformat()" in msg


def test_iso8601_accepts_full_datetime() -> None:
    """A full ISO 8601 datetime string passes."""
    encoded, vtype = encode_value("2024-01-15T12:30:45", ValueType.ISO8601)
    assert vtype == ValueType.ISO8601
    assert len(encoded) > 0


def test_iso8601_accepts_date_only() -> None:
    """A date-only string is parseable by datetime.fromisoformat."""
    encoded, vtype = encode_value("2024-01-15", ValueType.ISO8601)
    assert vtype == ValueType.ISO8601


def test_iso8601_accepts_bare_time_via_fallback() -> None:
    """A bare time string (``"12:30:45"``) fails
    ``datetime.fromisoformat`` but passes ``time.fromisoformat`` — the
    probe ladder must mirror the consumer's parser pair."""
    encoded, vtype = encode_value("12:30:45", ValueType.ISO8601)
    assert vtype == ValueType.ISO8601


def test_iso8601_accepts_aware_datetime() -> None:
    """ISO 8601 with timezone offset must be accepted."""
    encoded, vtype = encode_value("2024-01-15T12:30:45+00:00", ValueType.ISO8601)
    assert vtype == ValueType.ISO8601


def test_text_with_non_iso_string_still_succeeds() -> None:
    """The validation applies ONLY to ISO8601, not plain TEXT."""
    encoded, vtype = encode_value("not-an-iso-date", ValueType.TEXT)
    assert vtype == ValueType.TEXT
    assert len(encoded) > 0
