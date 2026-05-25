"""Pin: ``decode_value`` short-read diagnostics name the wire type the
caller asked for, not the underlying primitive.

Previously a 4-byte buffer fed to ``decode_value`` produced five
different error messages (``"Need 8 bytes for int64"``, ``"... for
uint64"``, ``"... for double"``, ``"... for NULL value"``,
``"... for int64"`` for UNIXTIME), of which two named the wrong type
for the caller's actual question. Threading ``label=`` through the
int64 / uint64 / double primitives lets each decode arm forward a
per-cell label.
"""

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
