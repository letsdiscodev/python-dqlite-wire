"""Pin: ``decode_row_header`` uses a module-level precomputed
``_NIBBLE_TO_VALUETYPE`` tuple to resolve nibble → ValueType
rather than calling the IntEnum constructor per cell.

Each ``ValueType(nibble)`` call goes through ``IntEnum.__call__`` →
``__new__`` → ``_value2member_map_`` dict lookup → IntEnum value-
coercion check. Measured ~200 ns per call in CPython 3.14. Called
once per cell of every decoded row; for a 100k-row × 16-col result
the dispatch alone is ~320 ms of pure-Python loop CPU.

The precomputed tuple replaces the constructor with a single C-
level index operation (~30 ns), 6-7× faster on the dispatch with
no behavioural change. Mirrors the in-tree precedent of
``_VALID_TYPE_CODES`` (a precomputed frozenset for encode-side
validation) at tuples.py:28.

The existing test (test_tuples.py:451-454) pins the error message
``"Invalid value type code"`` on malformed nibbles; the fix must
preserve that exact phrasing.
"""

from __future__ import annotations

from unittest import mock

import pytest

from dqlitewire import tuples as tuples_mod
from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError


def test_nibble_to_valuetype_lookup_table_exists_and_is_well_formed() -> None:
    """The precomputed table has exactly 16 entries (one per
    possible nibble), with valid ValueType members at the cells
    corresponding to known type codes and None elsewhere.
    """
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
    """Per-cell ValueType(nibble) is the IntEnum-constructor hot
    loop the fix removes. Patch ValueType to raise if called and
    decode a header — the precomputed-lookup fix must not call
    the constructor at runtime.
    """
    # Build a 16-column header with valid nibbles.
    # Each byte packs two nibbles (low, high). With column_count=16
    # we need 8 bytes + 0 padding.
    valid_codes = sorted(int(v) for v in ValueType)
    # Cycle through valid codes for both nibbles.
    nibbles = [valid_codes[i % len(valid_codes)] for i in range(16)]
    header_bytes = bytearray(8)
    for i in range(0, 16, 2):
        low = nibbles[i]
        high = nibbles[i + 1]
        header_bytes[i // 2] = (high << 4) | low

    # Patch ValueType.__call__ to raise — proves the precomputed
    # table is consulted instead.
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
    # Every type should be a valid ValueType (already verified).
    for t, expected_nibble in zip(types, nibbles, strict=True):
        assert int(t) == expected_nibble

    # The precomputed-table fix must NOT call ValueType() at runtime.
    # The table is built at module import time (16 calls there);
    # those are not counted by our patch because the table is
    # already built. Per-cell calls during decode_row_header are
    # the only calls this test would see.
    assert call_count == 0, (
        f"decode_row_header called ValueType() {call_count} times; "
        "expected zero (precomputed _NIBBLE_TO_VALUETYPE table should "
        "replace the per-cell constructor)"
    )


def test_decode_row_header_invalid_nibble_preserves_existing_error_phrasing() -> None:
    """The existing test (test_tuples.py:451-454) pins the error
    message ``"Invalid value type code"``. The fix must preserve
    that exact phrasing.
    """
    # Build a header with an invalid nibble (e.g. 0 — verify it's
    # not a valid ValueType code).
    valid_codes = {int(v) for v in ValueType}
    invalid_nibbles = [n for n in range(16) if n not in valid_codes]
    assert invalid_nibbles, "test setup requires at least one invalid nibble"
    invalid = invalid_nibbles[0]

    # Single-column header: 1 byte, padded to 8.
    header_bytes = bytearray(8)
    header_bytes[0] = invalid

    with pytest.raises(DecodeError, match="Invalid value type code"):
        tuples_mod.decode_row_header(bytes(header_bytes), 1)
