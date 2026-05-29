"""encode_value rejection branches must bound the quoted value repr at
_MAX_VALUE_REPR so a hostile caller can't bake kilobytes into the error."""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


def test_boolean_rejection_with_huge_int_is_bounded() -> None:
    huge = 10**500
    with pytest.raises(EncodeError) as exc_info:
        encode_value(huge, ValueType.BOOLEAN)
    msg = str(exc_info.value)
    assert len(msg) < 400, f"BOOLEAN error message len {len(msg)} > 400"
    assert "chars" in msg or "digits" in msg


def test_null_rejection_with_huge_int_is_bounded() -> None:
    huge = 10**500
    with pytest.raises(EncodeError) as exc_info:
        encode_value(huge, ValueType.NULL)
    msg = str(exc_info.value)
    assert len(msg) < 400, f"NULL error message len {len(msg)} > 400"
    assert "chars" in msg or "digits" in msg


def test_null_rejection_with_large_bytes_is_bounded() -> None:
    payload = b"x" * 100_000
    with pytest.raises(EncodeError) as exc_info:
        encode_value(payload, ValueType.NULL)
    msg = str(exc_info.value)
    assert len(msg) < 400, f"NULL bytes error message len {len(msg)} > 400"


def test_small_value_rejection_message_unchanged() -> None:
    with pytest.raises(EncodeError) as exc_info:
        encode_value(5, ValueType.BOOLEAN)
    msg = str(exc_info.value)
    assert "5" in msg
    assert " chars]" not in msg
