"""decode_value's TEXT/ISO8601 branch forwards ValueType.name as the decode_text label."""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError
from dqlitewire.types import decode_value


def test_decode_value_iso8601_truncated_names_iso8601() -> None:
    """No-NUL buffer must surface "ISO8601" in the DecodeError, not the generic "Text" prefix."""
    truncated = b"20240101"
    with pytest.raises(DecodeError, match="ISO8601"):
        decode_value(truncated, ValueType.ISO8601)


def test_decode_value_text_truncated_names_text() -> None:
    """The TEXT branch must surface "TEXT" (the ValueType name) in the DecodeError."""
    truncated = b"abcdefgh"
    with pytest.raises(DecodeError, match="TEXT"):
        decode_value(truncated, ValueType.TEXT)
