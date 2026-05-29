"""Intentional asymmetry vs. C ``bind.c::DQLITE_BOOLEAN``: encode rejects
non-``{0,1}`` ints (catching caller bugs), but decode reads any uint64 as
truthy. Decode->re-encode of an out-of-range raw BOOLEAN is lossy by design."""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import decode_value, encode_uint64, encode_value


def test_encode_boolean_rejects_arbitrary_int() -> None:
    with pytest.raises(EncodeError, match=r"BOOLEAN requires"):
        encode_value(5, ValueType.BOOLEAN)


def test_decode_boolean_accepts_raw_5_as_truthy() -> None:
    """A raw=5 BOOLEAN cell decodes to ``True`` (permissive, matching C)."""
    cell = encode_uint64(5)
    value, consumed = decode_value(cell, ValueType.BOOLEAN)
    assert value is True
    assert consumed == 8


def test_round_trip_true_does_not_recover_raw_5() -> None:
    """Re-encoding the decoded True writes raw 1, not the original 5 (lossy by design)."""
    cell_raw5 = encode_uint64(5)
    value, _ = decode_value(cell_raw5, ValueType.BOOLEAN)
    assert value is True

    re_encoded, _ = encode_value(value, ValueType.BOOLEAN)
    assert re_encoded == encode_uint64(1)
    assert re_encoded != cell_raw5
