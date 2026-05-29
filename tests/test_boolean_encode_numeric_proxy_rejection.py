"""``encode_value(..., BOOLEAN)`` rejects numeric proxies (numpy.int64,
Decimal, Fraction) that expose ``__int__``/``__index__`` but don't subclass
``int``; accepting them would collapse the value through ``True``. Int
subclasses (IntEnum) are still admitted."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


class _NumericProxy:
    """Stand-in for ``numpy.int64``: numeric dunders but not an ``int`` subclass."""

    def __init__(self, value: int) -> None:
        self._value = value

    def __int__(self) -> int:
        return self._value

    def __index__(self) -> int:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return f"numpy.int64({self._value})"


def test_boolean_rejects_numeric_proxy_zero() -> None:
    """A proxy holding 0 passes ``value in (0, 1)`` via ``__eq__`` but is rejected."""
    proxy = _NumericProxy(0)
    with pytest.raises(EncodeError) as exc_info:
        encode_value(proxy, ValueType.BOOLEAN)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "Numeric proxies" in msg or "numeric prox" in msg.lower()
    assert "int(x)" in msg or "bool(x)" in msg


def test_boolean_rejects_numeric_proxy_one() -> None:
    proxy = _NumericProxy(1)
    with pytest.raises(EncodeError, match=r"BOOLEAN requires"):
        encode_value(proxy, ValueType.BOOLEAN)  # type: ignore[arg-type]


def test_boolean_rejects_decimal() -> None:
    with pytest.raises(EncodeError, match=r"BOOLEAN requires"):
        encode_value(Decimal(0), ValueType.BOOLEAN)  # type: ignore[arg-type]


def test_boolean_rejects_fraction() -> None:
    with pytest.raises(EncodeError, match=r"BOOLEAN requires"):
        encode_value(Fraction(0, 1), ValueType.BOOLEAN)  # type: ignore[arg-type]


def test_boolean_accepts_explicit_int_coercion_of_proxy() -> None:
    """Coercing the proxy via ``int(x)`` yields a real int the escape hatch admits."""
    proxy = _NumericProxy(0)
    encoded, vtype = encode_value(int(proxy), ValueType.BOOLEAN)
    assert vtype == ValueType.BOOLEAN
    assert encoded == b"\x00" * 8


def test_boolean_accepts_int_subclass_via_intenum() -> None:
    """Int subclasses (``enum.IntEnum``) are admitted; rejection targets only proxies."""
    import enum

    class Flag(enum.IntEnum):
        OFF = 0
        ON = 1

    encoded, vtype = encode_value(Flag.ON, ValueType.BOOLEAN)
    assert vtype == ValueType.BOOLEAN
