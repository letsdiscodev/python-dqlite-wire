"""Pin: ``cap_raw_message`` truncation is deterministic across callers."""

from __future__ import annotations

import pytest

from dqlitewire import DEFAULT_MAX_RAW_MESSAGE, cap_raw_message


def test_returns_none_unchanged() -> None:
    assert cap_raw_message(None, 100) is None


def test_returns_input_unchanged_when_within_cap() -> None:
    assert cap_raw_message("short", 100) == "short"


def test_returns_input_unchanged_at_exact_cap() -> None:
    text = "x" * 100
    assert cap_raw_message(text, 100) == text


def test_truncates_with_suffix_when_over_cap() -> None:
    text = "x" * 150
    result = cap_raw_message(text, 100)
    assert result is not None
    assert result.startswith("x" * 100)
    assert "[raw_message truncated, 50 codepoints]" in result


def test_overflow_count_matches_dropped_chars() -> None:
    text = "y" * 4096
    result = cap_raw_message(text, 1024)
    assert result is not None
    assert "3072 codepoints" in result


def test_zero_cap_truncates_everything() -> None:
    result = cap_raw_message("hello", 0)
    assert result is not None
    assert "5 codepoints" in result
    assert result.startswith("...")


@pytest.mark.parametrize("max_chars", [1, 10, 100, 1024, 4096])
def test_round_trip_under_cap_no_suffix(max_chars: int) -> None:
    text = "x" * (max_chars - 1) if max_chars > 0 else ""
    result = cap_raw_message(text, max_chars)
    assert result == text
    if result:
        assert "truncated" not in result


@pytest.mark.parametrize("bad_max", [-1, -100, -(2**31)])
def test_negative_max_chars_rejected(bad_max: int) -> None:
    """Reject negative budget so a typo surfaces instead of a misleading overflow count."""
    with pytest.raises(ValueError, match="max_chars must be >= 0"):
        cap_raw_message("hello", bad_max)


def test_negative_max_chars_rejects_even_for_none_input() -> None:
    """Validation fires before the ``raw_message is None`` short-circuit."""
    with pytest.raises(ValueError, match="max_chars must be >= 0"):
        cap_raw_message(None, -1)


def test_lf_passes_through_under_cap() -> None:
    """SECURITY: LF survives (this is a length cap, not a sanitiser); log sites must sanitise."""
    text = "line1\nline2"
    assert cap_raw_message(text, 100) == text


def test_tab_passes_through_under_cap() -> None:
    """SECURITY: Tab survives so the helper stays a pure length cap, not a sanitiser."""
    assert cap_raw_message("col1\tcol2", 100) == "col1\tcol2"


def test_crlf_passes_through_under_cap() -> None:
    """SECURITY: CR/CRLF survives the cap; sanitising is a documented non-goal."""
    assert cap_raw_message("a\r\nb", 100) == "a\r\nb"


def test_lf_in_truncated_prefix_survives() -> None:
    """The kept prefix preserves LF; only the overflow tail is dropped."""
    text = "a\nb" + ("x" * 200)
    result = cap_raw_message(text, 10)
    assert result is not None
    assert result.startswith("a\nb")


def test_default_max_raw_message_is_4_kib() -> None:
    """SSOT pin: the canonical raw-message cap is 4 KiB; downstream packages import it."""
    assert DEFAULT_MAX_RAW_MESSAGE == 4 * 1024 == 4096
