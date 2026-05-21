"""Pin: ``encode_double`` rejects ``bool`` rather than silently
coercing ``True`` / ``False`` to ``1.0`` / ``0.0``.

``isinstance(True, int)`` is True; ``float(True) == 1.0``. The
sibling primitives (``_validate_uint64`` at ``types.py``,
``encode_int64``, the ``encode_value`` FLOAT arm) all reject
``bool`` explicitly. ``encode_double`` was the discipline gap.
"""

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
    """Bare ``int`` (not ``bool``) is rejected too.

    ``struct.pack("<d", 42)`` would silently coerce via C-level float
    promotion, but the same coercion drops bits for ``|x| >= 2**53``
    and raises ``OverflowError`` (outside ``EncodeError``) for
    ``|x| >= 2**1024``. Callers wanting int → FLOAT must call
    ``float(x)`` themselves.
    """
    with pytest.raises(EncodeError, match="requires float"):
        encode_double(5)


def test_encode_double_rejects_numpy_bool_proxy() -> None:
    """``numpy.bool_`` is NOT a Python ``bool`` subclass (NumPy reparented
    it long ago). The bool guard alone leaves it slip through; the
    float-subclass check is what rejects it. A minimal proxy mirrors
    the NumPy shape: not a bool subclass, exposes ``__float__``."""

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
