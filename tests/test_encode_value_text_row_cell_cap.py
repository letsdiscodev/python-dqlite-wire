"""encode_value TEXT/ISO8601 must enforce the same _MAX_TEXT_VALUE_SIZE cap
as decode_text, else a same-process encode then decode ships bytes the
decoder rejects."""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import _MAX_TEXT_VALUE_SIZE, decode_value, encode_value


def test_encode_value_text_at_cap_round_trips() -> None:
    value = "x" * _MAX_TEXT_VALUE_SIZE
    encoded, _ = encode_value(value, ValueType.TEXT)
    decoded, _ = decode_value(encoded, ValueType.TEXT)
    assert decoded == value


def test_encode_value_text_one_over_cap_rejected_at_encode() -> None:
    value = "x" * (_MAX_TEXT_VALUE_SIZE + 1)
    with pytest.raises(EncodeError):
        encode_value(value, ValueType.TEXT)


def test_encode_value_iso8601_at_cap_round_trips() -> None:
    value = "0" * _MAX_TEXT_VALUE_SIZE
    encoded, _ = encode_value(value, ValueType.ISO8601)
    decoded, _ = decode_value(encoded, ValueType.ISO8601)
    assert decoded == value


def test_encode_value_iso8601_one_over_cap_rejected() -> None:
    value = "0" * (_MAX_TEXT_VALUE_SIZE + 1)
    with pytest.raises(EncodeError):
        encode_value(value, ValueType.ISO8601)


def test_encode_value_text_error_message_names_the_value_type() -> None:
    value = "x" * (_MAX_TEXT_VALUE_SIZE + 1)
    with pytest.raises(EncodeError, match="TEXT"):
        encode_value(value, ValueType.TEXT)
    with pytest.raises(EncodeError, match="ISO8601"):
        encode_value(value, ValueType.ISO8601)
