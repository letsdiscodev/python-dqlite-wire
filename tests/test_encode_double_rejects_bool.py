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


def test_encode_double_accepts_int_passes_through() -> None:
    """Bare int (not bool) is accepted and coerced to float by struct."""
    encoded = encode_double(5)
    assert len(encoded) == 8


def test_encode_double_accepts_nan_and_inf() -> None:
    encode_double(float("nan"))
    encode_double(float("inf"))
    encode_double(float("-inf"))
