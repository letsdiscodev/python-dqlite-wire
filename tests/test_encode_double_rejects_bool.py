"""``encode_double`` rejects ``bool`` rather than coercing ``True``/``False``
to ``1.0``/``0.0``, matching the other primitives' explicit bool guards."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_double


def test_encode_double_rejects_true() -> None:
    with pytest.raises(EncodeError, match="bool"):
        encode_double(True)


def test_encode_double_rejects_false() -> None:
    with pytest.raises(EncodeError, match="bool"):
        encode_double(False)


def test_encode_double_accepts_zero_and_finite_floats() -> None:
    assert encode_double(0.0) == b"\x00\x00\x00\x00\x00\x00\x00\x00"
    encoded = encode_double(1.5)
    assert len(encoded) == 8


def test_encode_double_rejects_bare_int() -> None:
    """Bare ``int`` is rejected: ``struct.pack`` coercion drops bits past 2**53
    and raises OverflowError past 2**1024. Callers must ``float(x)`` themselves."""
    with pytest.raises(EncodeError, match="requires float"):
        encode_double(5)


def test_encode_double_rejects_numpy_bool_proxy() -> None:
    """``numpy.bool_`` is not a ``bool`` subclass, so the bool guard misses it
    and the float-subclass check is what rejects it. Proxy mirrors that shape."""

    class FakeNpBool:
        def __init__(self, v: bool) -> None:
            self._v = v

        def __float__(self) -> float:
            return float(self._v)

    with pytest.raises(EncodeError, match="requires float"):
        encode_double(FakeNpBool(True))  # type: ignore[arg-type]


def test_encode_double_accepts_nan_and_inf() -> None:
    encode_double(float("nan"))
    encode_double(float("inf"))
    encode_double(float("-inf"))
