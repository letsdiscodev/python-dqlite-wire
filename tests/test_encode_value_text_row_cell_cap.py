"""Pin: ``encode_value(str, ValueType.TEXT)`` enforces the same
``_MAX_TEXT_VALUE_SIZE`` (16 MiB) cap as ``decode_text`` so a
same-process encode → decode round-trip cannot ship bytes that the
decoder rejects.

The BLOB branch of ``encode_value`` already caps via
``_MAX_BLOB_SIZE``; the TEXT / ISO8601 branch was the asymmetric
gap. A 16 MiB+1 string encoded successfully under the previous code
and the resulting bytes failed ``decode_value(..., TEXT)`` with
``DecodeError("Text length exceeds maximum (16777216)")`` —
internally inconsistent codec.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import _MAX_TEXT_VALUE_SIZE, decode_value, encode_value


def test_encode_value_text_at_cap_round_trips() -> None:
    """A string at exactly the cap must encode and decode cleanly —
    pinning that the cap is inclusive of the boundary."""
    value = "x" * _MAX_TEXT_VALUE_SIZE
    encoded, _ = encode_value(value, ValueType.TEXT)
    decoded, _ = decode_value(encoded, ValueType.TEXT)
    assert decoded == value


def test_encode_value_text_one_over_cap_rejected_at_encode() -> None:
    """Cap+1 must raise EncodeError at the encode step rather than
    succeed and produce bytes the decoder will reject."""
    value = "x" * (_MAX_TEXT_VALUE_SIZE + 1)
    with pytest.raises(EncodeError):
        encode_value(value, ValueType.TEXT)


def test_encode_value_iso8601_at_cap_round_trips() -> None:
    """ISO8601 shares the same cap as TEXT."""
    # An ISO8601 cell payload at the cap is unrealistic but the cap
    # discipline must apply uniformly across the text-shaped value
    # types.
    value = "0" * _MAX_TEXT_VALUE_SIZE
    encoded, _ = encode_value(value, ValueType.ISO8601)
    decoded, _ = decode_value(encoded, ValueType.ISO8601)
    assert decoded == value


def test_encode_value_iso8601_one_over_cap_rejected() -> None:
    value = "0" * (_MAX_TEXT_VALUE_SIZE + 1)
    with pytest.raises(EncodeError):
        encode_value(value, ValueType.ISO8601)


def test_encode_value_text_error_message_names_the_value_type() -> None:
    """The cap diagnostic must name the ValueType (TEXT or ISO8601)
    so a caller sees which row-cell type tripped — not a generic
    "Text" prefix from the helper."""
    value = "x" * (_MAX_TEXT_VALUE_SIZE + 1)
    with pytest.raises(EncodeError, match="TEXT"):
        encode_value(value, ValueType.TEXT)
    with pytest.raises(EncodeError, match="ISO8601"):
        encode_value(value, ValueType.ISO8601)
