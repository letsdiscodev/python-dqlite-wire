"""Pin: ``encode_value(..., ValueType.BOOLEAN)`` rejects numeric
proxies (e.g. ``numpy.int64(0)``, ``Decimal(0)``, ``Fraction(0)``)
that expose ``__int__`` / ``__index__`` but do NOT subclass ``int``.

The BOOLEAN arm's ``isinstance(value, int)`` escape hatch admits int
SUBCLASSES (e.g. ``enum.IntEnum``) but not numeric proxies — same
discipline as ``encode_double``'s float-subclass check (types.py:
``encode_double`` docstring on numpy.float64 / Decimal / Fraction
rejection). Silently accepting them would round-trip ``numpy.int64(5)``
through ``True`` and lose the caller's original value.

The error message must surface the actionable hint: numeric proxies
require explicit ``int(x)`` / ``bool(x)`` coercion at the bind site.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


class _NumericProxy:
    """Stand-in for ``numpy.int64`` / ``numpy.bool_``: exposes
    ``__int__`` / ``__index__`` / ``__bool__`` but does NOT subclass
    ``int`` or ``bool``."""

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
    """A numeric proxy holding 0 (which would otherwise pass the
    ``value in (0, 1)`` predicate via ``__eq__``) must be rejected
    because it is not an int subclass."""
    proxy = _NumericProxy(0)
    with pytest.raises(EncodeError) as exc_info:
        encode_value(proxy, ValueType.BOOLEAN)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    # Error must surface the actionable coercion hint.
    assert "Numeric proxies" in msg or "numeric prox" in msg.lower()
    assert "int(x)" in msg or "bool(x)" in msg


def test_boolean_rejects_numeric_proxy_one() -> None:
    """Same rejection for a proxy holding 1."""
    proxy = _NumericProxy(1)
    with pytest.raises(EncodeError, match=r"BOOLEAN requires"):
        encode_value(proxy, ValueType.BOOLEAN)  # type: ignore[arg-type]


def test_boolean_rejects_decimal() -> None:
    """``decimal.Decimal`` is not an int subclass."""
    with pytest.raises(EncodeError, match=r"BOOLEAN requires"):
        encode_value(Decimal(0), ValueType.BOOLEAN)  # type: ignore[arg-type]


def test_boolean_rejects_fraction() -> None:
    """``fractions.Fraction`` is not an int subclass."""
    with pytest.raises(EncodeError, match=r"BOOLEAN requires"):
        encode_value(Fraction(0, 1), ValueType.BOOLEAN)  # type: ignore[arg-type]


def test_boolean_accepts_explicit_int_coercion_of_proxy() -> None:
    """The actionable hint: coercing the proxy via ``int(x)`` produces
    a real Python int, which the escape hatch then admits."""
    proxy = _NumericProxy(0)
    encoded, vtype = encode_value(int(proxy), ValueType.BOOLEAN)
    assert vtype == ValueType.BOOLEAN
    assert encoded == b"\x00" * 8


def test_boolean_accepts_int_subclass_via_intenum() -> None:
    """Real int subclasses (``enum.IntEnum``) are admitted by the
    escape hatch — the rejection targets numeric proxies, not all
    non-bool inputs."""
    import enum

    class Flag(enum.IntEnum):
        OFF = 0
        ON = 1

    encoded, vtype = encode_value(Flag.ON, ValueType.BOOLEAN)
    assert vtype == ValueType.BOOLEAN
    # IntEnum members satisfy isinstance(int) and value in (0, 1).
