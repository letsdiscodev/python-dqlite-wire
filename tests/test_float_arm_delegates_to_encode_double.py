"""Pin: encode_value's FLOAT arm delegates float-subclass rejection to
encode_double (richer message), but keeps its own bool/int guards whose
wording is more specific than encode_double's generic reject."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


def test_float_arm_rejects_decimal_with_canonical_message() -> None:
    """Decimal is a numeric proxy, not a float subclass: the FLOAT arm
    delegates to encode_double's "cast with float(x)" message."""
    with pytest.raises(EncodeError) as exc_info:
        encode_value(Decimal("1.0"), ValueType.FLOAT)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "encode_double requires float" in msg
    assert "cast with float(x) explicitly" in msg


def test_float_arm_rejects_fraction_with_canonical_message() -> None:
    """Fraction is also a numeric proxy."""
    with pytest.raises(EncodeError) as exc_info:
        encode_value(Fraction(3, 2), ValueType.FLOAT)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "encode_double requires float" in msg
    assert "cast with float(x) explicitly" in msg


def test_float_arm_still_rejects_bool_with_specific_message() -> None:
    """The bool guard stays in the FLOAT arm for its specific wording."""
    with pytest.raises(EncodeError, match=r"Expected float for FLOAT, got bool"):
        encode_value(True, ValueType.FLOAT)


def test_float_arm_still_rejects_int_with_precision_hint() -> None:
    """The int guard stays — its message names the |x| >= 2**53
    precision-loss boundary that encode_double's generic reject omits."""
    with pytest.raises(EncodeError) as exc_info:
        encode_value(42, ValueType.FLOAT)
    msg = str(exc_info.value)
    assert "Cannot encode int as FLOAT" in msg
    assert "2**53" in msg


def test_float_arm_accepts_float() -> None:
    """Real floats encode normally."""
    encoded, vtype = encode_value(1.5, ValueType.FLOAT)
    assert vtype == ValueType.FLOAT
    assert len(encoded) == 8
