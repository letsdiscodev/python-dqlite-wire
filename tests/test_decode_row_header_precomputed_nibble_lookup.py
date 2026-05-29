"""``decode_row_header`` resolves nibble -> ValueType via a module-level
precomputed ``_NIBBLE_TO_VALUETYPE`` tuple rather than calling the IntEnum
constructor per cell (a per-row hot loop).
"""

from __future__ import annotations

from unittest import mock

import pytest

from dqlitewire import tuples as tuples_mod
from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError


def test_nibble_to_valuetype_lookup_table_exists_and_is_well_formed() -> None:
    """16 entries: a ValueType at each known type code, None elsewhere."""
    table = tuples_mod._NIBBLE_TO_VALUETYPE
    assert len(table) == 16, "table must cover all 4-bit nibble values"

    valid_codes = {int(v) for v in ValueType}
    for nibble in range(16):
        if nibble in valid_codes:
            assert table[nibble] is not None, f"nibble {nibble} should map to a ValueType"
            assert int(table[nibble]) == nibble  # type: ignore[arg-type]
            assert isinstance(table[nibble], ValueType)
        else:
            assert table[nibble] is None, (
                f"nibble {nibble} is not a known ValueType; should map to None"
            )


def test_decode_row_header_does_not_call_valuetype_constructor_per_cell() -> None:
    """Decoding a header must not invoke the ValueType constructor; the
    precomputed table is consulted instead."""
    # 16-column header: each byte packs two nibbles (low, high), 8 bytes total.
    valid_codes = sorted(int(v) for v in ValueType)
    nibbles = [valid_codes[i % len(valid_codes)] for i in range(16)]
    header_bytes = bytearray(8)
    for i in range(0, 16, 2):
        low = nibbles[i]
        high = nibbles[i + 1]
        header_bytes[i // 2] = (high << 4) | low

    call_count = 0
    original_call = type(ValueType).__call__

    def counting_call(cls, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_call(cls, *args, **kwargs)

    with mock.patch.object(type(ValueType), "__call__", counting_call):
        types, consumed = tuples_mod.decode_row_header(bytes(header_bytes), 16)

    assert consumed == 8
    assert isinstance(types, list)
    assert len(types) == 16
    for t, expected_nibble in zip(types, nibbles, strict=True):
        assert int(t) == expected_nibble

    # The table is built at import time; only per-cell calls reach the patch.
    assert call_count == 0, (
        f"decode_row_header called ValueType() {call_count} times; "
        "expected zero (precomputed _NIBBLE_TO_VALUETYPE table should "
        "replace the per-cell constructor)"
    )


def test_decode_row_header_invalid_nibble_preserves_existing_error_phrasing() -> None:
    """An invalid nibble must still raise with the "Invalid value type code"
    phrasing pinned by test_tuples.py."""
    valid_codes = {int(v) for v in ValueType}
    invalid_nibbles = [n for n in range(16) if n not in valid_codes]
    assert invalid_nibbles, "test setup requires at least one invalid nibble"
    invalid = invalid_nibbles[0]

    header_bytes = bytearray(8)
    header_bytes[0] = invalid

    with pytest.raises(DecodeError, match="Invalid value type code"):
        tuples_mod.decode_row_header(bytes(header_bytes), 1)
