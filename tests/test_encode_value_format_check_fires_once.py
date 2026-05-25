"""Pin: ``_reject_non_byte_format_memoryview`` must fire exactly once
per ``encode_value`` call on the inferred BLOB path.

Background: prior to the cleanup, the check ran in two places —
inside ``_infer_value_type`` (when value_type was None) AND inside the
explicit BLOB arm (after inference filled in value_type=BLOB). The
duplication paid an extra ``isinstance`` + format-membership test per
inferred BLOB cell, defeating the O(1)-per-cell promise of
``_infer_value_type``.

Hoisted form fires the check exactly once at the top of
``encode_value``, before any branching. Both the inferred BLOB path
and the explicit BLOB path pay the check exactly once.
"""

from __future__ import annotations

from unittest.mock import patch

from dqlitewire.constants import ValueType
from dqlitewire.types import encode_value


def test_format_check_fires_once_on_inferred_blob_path() -> None:
    """A ``memoryview(b'...')`` inferred to BLOB must trigger the
    format reject helper EXACTLY ONCE (not twice — once in
    _infer_value_type and once in the explicit-BLOB arm)."""
    mv = memoryview(b"\x01\x02\x03")
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value(mv)
    assert mock_reject.call_count == 1


def test_format_check_fires_once_on_explicit_blob_path() -> None:
    """An explicit-BLOB call must also trigger the helper exactly
    once."""
    mv = memoryview(b"\x01\x02\x03")
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value(mv, ValueType.BLOB)
    assert mock_reject.call_count == 1


def test_format_check_does_not_fire_for_non_memoryview_blob() -> None:
    """A ``bytes`` BLOB cell must NOT call the helper at all — the
    check is memoryview-specific."""
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value(b"abc")
        encode_value(b"abc", ValueType.BLOB)
    assert mock_reject.call_count == 0


def test_format_check_does_not_fire_for_non_blob_paths() -> None:
    """For non-BLOB values (str/int/float/bool/None), the helper must
    never be invoked."""
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value("hello")
        encode_value(42)
        encode_value(1.5)
        encode_value(True)
        encode_value(None)
    assert mock_reject.call_count == 0
