"""decode_value dispatches each ValueType via an O(1) lookup keyed by type code, not if/elif."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from dqlitewire import types as types_mod
from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError


def _decode_value_source() -> str:
    return textwrap.dedent(inspect.getsource(types_mod.decode_value))


def test_decode_value_does_not_use_elif_chain_against_value_type() -> None:
    src = _decode_value_source()
    tree = ast.parse(src)

    elif_compare_chain_arms = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        # Look for ``value_type == ValueType.X`` shape.
        if (
            isinstance(node.left, ast.Name)
            and node.left.id == "value_type"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Attribute)
            and isinstance(node.comparators[0].value, ast.Name)
            and node.comparators[0].value.id == "ValueType"
        ):
            elif_compare_chain_arms += 1

    assert elif_compare_chain_arms <= 1, (
        f"decode_value still contains {elif_compare_chain_arms} "
        "``value_type == ValueType.X`` comparisons; the dispatch table "
        "rewrite should collapse all of them into a single lookup. "
        "Up to 1 comparison is tolerated for an explicit fallback arm "
        "(e.g. fast-path for the most common type)."
    )


@pytest.mark.parametrize(
    "value_type, payload, expected_value, expected_consumed",
    [
        (ValueType.NULL, b"\x00" * 8, None, 8),
        (ValueType.INTEGER, (42).to_bytes(8, "little", signed=True), 42, 8),
        (ValueType.INTEGER, (-1).to_bytes(8, "little", signed=True), -1, 8),
        (ValueType.UNIXTIME, (1_700_000_000).to_bytes(8, "little", signed=True), 1_700_000_000, 8),
        (ValueType.BOOLEAN, (1).to_bytes(8, "little", signed=False), True, 8),
        (ValueType.BOOLEAN, (0).to_bytes(8, "little", signed=False), False, 8),
    ],
)
def test_decode_value_primitives_round_trip(
    value_type: ValueType,
    payload: bytes,
    expected_value: object,
    expected_consumed: int,
) -> None:
    value, consumed = types_mod.decode_value(payload, value_type)
    assert value == expected_value
    assert consumed == expected_consumed


def test_decode_value_text_arm_uses_value_type_name_in_label() -> None:
    """TEXT and ISO8601 share the arm forwarding value_type.name as the decode_text label."""
    text = "hello"
    encoded = types_mod.encode_text(text, label="TEXT", max_size=64)
    value, _ = types_mod.decode_value(encoded, ValueType.TEXT)
    assert value == text

    iso = "2026-05-27T10:00:00"
    encoded = types_mod.encode_text(iso, label="ISO8601", max_size=64)
    value, _ = types_mod.decode_value(encoded, ValueType.ISO8601)
    assert value == iso


def test_decode_value_blob_arm_round_trips() -> None:
    payload = b"\x01\x02\x03\x04"
    encoded = types_mod.encode_blob(payload)
    value, _ = types_mod.decode_value(encoded, ValueType.BLOB)
    assert value == payload


def test_decode_value_float_arm_round_trips() -> None:
    encoded = types_mod.encode_double(1.5)
    value, _ = types_mod.decode_value(encoded, ValueType.FLOAT)
    assert value == 1.5


def test_decode_value_rejects_unknown_type_code() -> None:
    """A value_type code with no decoder entry surfaces as DecodeError("Unknown value type")."""

    class _FakeType(int):
        pass

    # Type code 0 is unused in the enum.
    fake = _FakeType(0)
    with pytest.raises(DecodeError, match="Unknown value type"):
        types_mod.decode_value(b"\x00" * 8, fake)  # type: ignore[arg-type]


def test_decode_value_handles_text_errors_kwarg() -> None:
    """The text_errors kwarg flows through to decode_text for both TEXT and ISO8601 arms."""
    invalid_utf8 = b"\xff\x00" + b"\x00" * 6
    with pytest.raises(DecodeError):
        types_mod.decode_value(invalid_utf8, ValueType.TEXT, text_errors="strict")
    value, _ = types_mod.decode_value(invalid_utf8, ValueType.TEXT, text_errors="replace")
    assert isinstance(value, str)
