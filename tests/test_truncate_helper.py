"""Pin: ``_cap_raw_message`` truncation helper produces deterministic
output for both client and dbapi callers (single source of truth).
"""

from __future__ import annotations

import pytest

from dqlitewire._truncate import _cap_raw_message


def test_returns_none_unchanged() -> None:
    assert _cap_raw_message(None, 100) is None


def test_returns_input_unchanged_when_within_cap() -> None:
    assert _cap_raw_message("short", 100) == "short"


def test_returns_input_unchanged_at_exact_cap() -> None:
    text = "x" * 100
    assert _cap_raw_message(text, 100) == text


def test_truncates_with_suffix_when_over_cap() -> None:
    text = "x" * 150
    result = _cap_raw_message(text, 100)
    assert result is not None
    assert result.startswith("x" * 100)
    assert "[raw_message truncated, 50 codepoints]" in result


def test_overflow_count_matches_dropped_chars() -> None:
    text = "y" * 4096
    result = _cap_raw_message(text, 1024)
    assert result is not None
    assert "3072 codepoints" in result


def test_zero_cap_truncates_everything() -> None:
    result = _cap_raw_message("hello", 0)
    assert result is not None
    assert "5 codepoints" in result
    assert result.startswith("...")


@pytest.mark.parametrize("max_chars", [1, 10, 100, 1024, 4096])
def test_round_trip_under_cap_no_suffix(max_chars: int) -> None:
    text = "x" * (max_chars - 1) if max_chars > 0 else ""
    result = _cap_raw_message(text, max_chars)
    assert result == text
    if result:
        assert "truncated" not in result
