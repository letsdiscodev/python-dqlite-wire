"""Pin: ``decode_value`` TEXT / ISO8601 branch forwards the cell's
``ValueType.name`` as the ``label`` argument to ``decode_text`` so
mid-frame diagnostics name the actual wire-cell type, symmetric with
``encode_value``.

Without this, a truncated ISO8601 cell surfaced as ``"Text not
null-terminated"`` — obscuring the cell type at the moment an
operator reading a corrupted frame most needs it.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError
from dqlitewire.types import decode_value


def test_decode_value_iso8601_truncated_names_iso8601() -> None:
    """A buffer with no NUL terminator must surface ``"ISO8601"`` in
    the DecodeError, not the generic ``"Text"`` prefix."""
    # 8 bytes, no NUL — decode_text walks the whole buffer and raises.
    truncated = b"20240101"
    with pytest.raises(DecodeError, match="ISO8601"):
        decode_value(truncated, ValueType.ISO8601)


def test_decode_value_text_truncated_names_text() -> None:
    """The TEXT branch must surface ``"TEXT"`` (the ValueType name) in
    the DecodeError. Keeps the encoder/decoder label discipline
    symmetric."""
    truncated = b"abcdefgh"
    with pytest.raises(DecodeError, match="TEXT"):
        decode_value(truncated, ValueType.TEXT)
