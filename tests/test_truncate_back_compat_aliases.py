"""The underscore-prefixed _DEFAULT_MAX_RAW_MESSAGE/_cap_raw_message remain as
back-compat aliases for the now-public names until downstream consumers migrate."""

from __future__ import annotations

from dqlitewire import DEFAULT_MAX_RAW_MESSAGE, cap_raw_message
from dqlitewire._truncate import _DEFAULT_MAX_RAW_MESSAGE, _cap_raw_message


def test_value_alias_points_at_public_constant() -> None:
    assert _DEFAULT_MAX_RAW_MESSAGE == DEFAULT_MAX_RAW_MESSAGE


def test_function_alias_is_public_function() -> None:
    assert _cap_raw_message is cap_raw_message
