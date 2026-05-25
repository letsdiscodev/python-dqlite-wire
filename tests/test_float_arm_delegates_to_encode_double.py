"""Pin: ``encode_value`` FLOAT arm's third guard (``if not isinstance
(value, float)``) used to duplicate ``encode_double``'s float-subclass
check with strictly poorer wording. Remove the redundant guard and
let ``encode_double``'s richer message ("encode_double requires float
... cast with float(x) explicitly if int / numeric-proxy semantics
are intended") reach the caller.

The bool / int guards stay in the FLOAT arm — they produce richer /
more specific wording than ``encode_double``'s generic rejections.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


def test_float_arm_rejects_decimal_with_canonical_message() -> None:
    """``Decimal`` is a numeric proxy, not a float subclass. The FLOAT
    arm must delegate to ``encode_double`` whose error message names
    the helper and the canonical "cast with float(x)" hint."""
    with pytest.raises(EncodeError) as exc_info:
        encode_value(Decimal("1.0"), ValueType.FLOAT)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "encode_double requires float" in msg
    assert "cast with float(x) explicitly" in msg


def test_float_arm_rejects_fraction_with_canonical_message() -> None:
    """``Fraction`` is also a numeric proxy."""
    with pytest.raises(EncodeError) as exc_info:
        encode_value(Fraction(3, 2), ValueType.FLOAT)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "encode_double requires float" in msg
    assert "cast with float(x) explicitly" in msg


def test_float_arm_still_rejects_bool_with_specific_message() -> None:
    """The bool guard stays in the FLOAT arm — it produces specific
    wording the encode_double generic reject would lose."""
    with pytest.raises(EncodeError, match=r"Expected float for FLOAT, got bool"):
        encode_value(True, ValueType.FLOAT)


def test_float_arm_still_rejects_int_with_precision_hint() -> None:
    """The int guard stays — its message names the |x| >= 2**53
    precision-loss boundary that encode_double's generic reject would
    not."""
    with pytest.raises(EncodeError) as exc_info:
        encode_value(42, ValueType.FLOAT)
    msg = str(exc_info.value)
    assert "Cannot encode int as FLOAT" in msg
    assert "2**53" in msg


def test_float_arm_accepts_float() -> None:
    """Negative twin: real floats encode normally."""
    encoded, vtype = encode_value(1.5, ValueType.FLOAT)
    assert vtype == ValueType.FLOAT
    assert len(encoded) == 8
