"""Pin: ``encode_value`` rejection branches that quote
caller-controlled values must bound the rendered repr at
``_MAX_VALUE_REPR`` characters — same discipline applied by
``_validate_uint*`` / ``_validate_int64`` via ``_bounded_repr``.

Without this, a hostile or buggy caller passing ``10 ** 500`` to the
BOOLEAN arm (or a 100 KiB ``bytes`` to the NULL arm) bakes a
kilobyte of digits / bytes into the EncodeError message — and into
every log line and traceback that quotes the exception. The
discipline is documented at ``_MAX_VALUE_REPR``'s definition; this
test pins it at the at-risk encode_value sites.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


def test_boolean_rejection_with_huge_int_is_bounded() -> None:
    """``encode_value(10 ** 500, ValueType.BOOLEAN)`` produces a
    ~500-digit ``repr(value)``; the error message must cap it via
    ``_bounded_repr_any``."""
    huge = 10**500
    with pytest.raises(EncodeError) as exc_info:
        encode_value(huge, ValueType.BOOLEAN)
    msg = str(exc_info.value)
    # The full repr would be ~500 chars; the bounded message must
    # stay well under that even with surrounding boilerplate.
    assert len(msg) < 400, f"BOOLEAN error message len {len(msg)} > 400"
    # Truncation marker present.
    assert "chars" in msg or "digits" in msg


def test_null_rejection_with_huge_int_is_bounded() -> None:
    """``encode_value(10 ** 500, ValueType.NULL)`` (non-None value
    with explicit NULL type) — the rejection quotes the value via
    ``{value!r}`` and must be bounded."""
    huge = 10**500
    with pytest.raises(EncodeError) as exc_info:
        encode_value(huge, ValueType.NULL)
    msg = str(exc_info.value)
    assert len(msg) < 400, f"NULL error message len {len(msg)} > 400"
    assert "chars" in msg or "digits" in msg


def test_null_rejection_with_large_bytes_is_bounded() -> None:
    """``encode_value(b"x" * 100_000, ValueType.NULL)`` —
    ``repr(b"x" * 100_000)`` is huge; the bound must catch it."""
    payload = b"x" * 100_000
    with pytest.raises(EncodeError) as exc_info:
        encode_value(payload, ValueType.NULL)
    msg = str(exc_info.value)
    assert len(msg) < 400, f"NULL bytes error message len {len(msg)} > 400"


def test_small_value_rejection_message_unchanged() -> None:
    """For small values (well under the cap), the message is the
    full repr — no truncation marker added."""
    with pytest.raises(EncodeError) as exc_info:
        encode_value(5, ValueType.BOOLEAN)
    msg = str(exc_info.value)
    assert "5" in msg
    # Small repr stays under cap; no "[NNN chars]" suffix.
    assert " chars]" not in msg
