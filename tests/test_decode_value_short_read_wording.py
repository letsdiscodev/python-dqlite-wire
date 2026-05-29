"""decode_value short-read diagnostics name the wire type asked for, not the primitive."""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError
from dqlitewire.types import decode_value


@pytest.mark.parametrize(
    "value_type,expected_label",
    [
        (ValueType.INTEGER, "INTEGER cell"),
        (ValueType.UNIXTIME, "UNIXTIME cell"),
        (ValueType.FLOAT, "FLOAT cell"),
        (ValueType.BOOLEAN, "BOOLEAN cell"),
        (ValueType.NULL, "NULL cell"),
    ],
)
def test_short_read_diagnostic_names_value_type(value_type: ValueType, expected_label: str) -> None:
    with pytest.raises(DecodeError, match=f"Need 8 bytes for {expected_label}"):
        decode_value(b"\x00\x00\x00\x00", value_type)
