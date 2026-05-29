"""_reject_non_byte_format_memoryview must fire exactly once per
encode_value call (it was previously run in both _infer_value_type and the
explicit BLOB arm, doubling the per-cell cost)."""

from __future__ import annotations

from unittest.mock import patch

from dqlitewire.constants import ValueType
from dqlitewire.types import encode_value


def test_format_check_fires_once_on_inferred_blob_path() -> None:
    mv = memoryview(b"\x01\x02\x03")
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value(mv)
    assert mock_reject.call_count == 1


def test_format_check_fires_once_on_explicit_blob_path() -> None:
    mv = memoryview(b"\x01\x02\x03")
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value(mv, ValueType.BLOB)
    assert mock_reject.call_count == 1


def test_format_check_does_not_fire_for_non_memoryview_blob() -> None:
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value(b"abc")
        encode_value(b"abc", ValueType.BLOB)
    assert mock_reject.call_count == 0


def test_format_check_does_not_fire_for_non_blob_paths() -> None:
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value("hello")
        encode_value(42)
        encode_value(1.5)
        encode_value(True)
        encode_value(None)
    assert mock_reject.call_count == 0
