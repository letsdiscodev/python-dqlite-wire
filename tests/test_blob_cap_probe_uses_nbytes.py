"""encode_value's BLOB cap-before-materialise probe must use memoryview.nbytes
(byte count), not len(memoryview) (element count for multi-byte formats), so an
oversize multi-byte view is rejected before the materialise allocation
regardless of _reject_non_byte_format_memoryview's ordering."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import _MAX_BLOB_SIZE, encode_value


def test_blob_cap_probe_rejects_oversize_before_materialise() -> None:
    """A multi-byte memoryview whose element count fits under the cap but
    whose nbytes exceeds it must be rejected by the outer probe before
    bytes(value) materialises 8x the bytes."""
    import array as _array

    element_count = _MAX_BLOB_SIZE // 8 + 1
    mv = memoryview(_array.array("Q", [0] * element_count))
    assert mv.itemsize == 8
    assert len(mv) == element_count
    assert mv.nbytes == element_count * 8
    assert mv.nbytes > _MAX_BLOB_SIZE
    assert len(mv) < _MAX_BLOB_SIZE  # len-based probe would pass; nbytes-based catches it

    # Patch the format check to no-op (simulate a refactor that drops it) and
    # make encode_blob fail if reached — the outer probe must fire first.
    def _should_not_be_reached(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Outer cap probe failed to reject before materialise — "
            "encode_blob was reached, indicating the probe used "
            "len(mv) (element count) instead of mv.nbytes (byte count)."
        )

    with (
        patch("dqlitewire.types._reject_non_byte_format_memoryview"),
        patch("dqlitewire.types.encode_blob", side_effect=_should_not_be_reached),
        pytest.raises(EncodeError, match=r"Blob length \d+ exceeds maximum"),
    ):
        encode_value(mv, ValueType.BLOB)


def test_blob_cap_probe_reports_byte_count_not_element_count() -> None:
    """The EncodeError message must name the byte count (mv.nbytes), not the
    element count (len(mv)), to match the cap's units."""
    import array as _array

    element_count = _MAX_BLOB_SIZE // 8 + 1
    mv = memoryview(_array.array("Q", [0] * element_count))
    expected_nbytes = element_count * 8

    def _stub(*args: object, **kwargs: object) -> bytes:
        return b""

    with (
        patch("dqlitewire.types._reject_non_byte_format_memoryview"),
        patch("dqlitewire.types.encode_blob", side_effect=_stub),
        pytest.raises(EncodeError) as exc_info,
    ):
        encode_value(mv, ValueType.BLOB)

    assert str(expected_nbytes) in str(exc_info.value)
    assert f"Blob length {len(mv)} " not in str(exc_info.value)


def test_blob_cap_probe_bytes_still_uses_len() -> None:
    """For bytes/bytearray/mmap (no .nbytes), the probe falls back to
    len(value), which is byte-correct by type."""
    payload = b"\x00" * (_MAX_BLOB_SIZE + 1)
    with pytest.raises(EncodeError, match=r"Blob length \d+ exceeds maximum"):
        encode_value(payload, ValueType.BLOB)
