"""Pin: ``encode_value`` BOOLEAN arm rejects non-``{0, 1}`` ints
(Python-side strictness), while ``decode_value`` BOOLEAN arm reads
any uint64 as truthiness — the documented asymmetry vs.
``dqlite-upstream/src/bind.c::DQLITE_BOOLEAN`` (which binds ``value->
boolean == 0 ? 0 : 1``).

A peer-emitted BOOLEAN cell with raw=5 decodes to ``True`` here; the
asymmetry is intentional and a round-trip is lossy by design (the
raw 5 collapses to the bool True, which re-encodes as raw 1, not 5).
The C strictness footgun ``encode_value(5, BOOLEAN)`` -> silent
truthy bind would silently lose the caller's value; Python catches
it at the encode site.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import decode_value, encode_uint64, encode_value


def test_encode_boolean_rejects_arbitrary_int() -> None:
    """``encode_value(5, BOOLEAN)`` raises; the strictness exists to
    surface caller bugs that the C path silently rewrites."""
    with pytest.raises(EncodeError, match=r"BOOLEAN requires"):
        encode_value(5, ValueType.BOOLEAN)


def test_decode_boolean_accepts_raw_5_as_truthy() -> None:
    """A peer-emitted BOOLEAN cell with raw=5 (e.g. from a row inserted
    as ``INSERT INTO t(b BOOLEAN) VALUES (5)``) decodes to ``True`` —
    permissive truthy decode, matching C's truthiness coercion."""
    cell = encode_uint64(5)
    value, consumed = decode_value(cell, ValueType.BOOLEAN)
    assert value is True
    assert consumed == 8


def test_round_trip_true_does_not_recover_raw_5() -> None:
    """Round-trip identity caveat: ``decode_value(encode_uint64(5),
    BOOLEAN) == True``, but ``encode_value(True, BOOLEAN)`` writes
    bytes for 1 (not 5). The decode -> re-encode of an out-of-{0, 1}
    raw BOOLEAN cell is lossy by design — the documented asymmetry
    vs. C bind."""
    cell_raw5 = encode_uint64(5)
    value, _ = decode_value(cell_raw5, ValueType.BOOLEAN)
    assert value is True

    re_encoded, _ = encode_value(value, ValueType.BOOLEAN)
    # Re-encoding True writes raw 1, not raw 5.
    assert re_encoded == encode_uint64(1)
    assert re_encoded != cell_raw5
