"""Pin: ``encode_value`` BLOB cap-before-materialise probe must use
``memoryview.nbytes`` (always bytes, regardless of ``itemsize``)
rather than ``len(memoryview)`` (element count for multi-byte
formats).

Without this, the cap correctness is silently coupled to the ordering
of ``_reject_non_byte_format_memoryview``: a future refactor that
moves or removes the format check would let a caller-supplied
``memoryview(array.array('Q', [0] * (_MAX_BLOB_SIZE // 8 + 1)))``
slip past the outer probe (``len(mv) == element_count``, under the
cap) and only get caught inside ``encode_blob`` AFTER the 8x
materialise allocation — exactly the wasted-allocation the outer
cap-before-materialise exists to prevent.

``.nbytes`` is invariant to ``itemsize`` and to any change in the
format-check ordering — robust by construction rather than by
convention.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import _MAX_BLOB_SIZE, encode_value


def test_blob_cap_probe_rejects_oversize_before_materialise() -> None:
    """With the format reject monkey-patched to no-op, a multi-byte-
    format memoryview whose element count fits under the cap but whose
    nbytes exceeds it must still be rejected by the OUTER probe
    (BEFORE ``bytes(value)`` materialises 8x the bytes).

    ``array.array('Q', ...)`` is 8 bytes per element. Pick an element
    count where ``len(mv)`` is well under the cap but ``mv.nbytes``
    exceeds it. Patch ``encode_blob`` to a no-op sentinel — if the
    outer probe relies on ``len(mv)``, the cap check passes and the
    test reaches the patched ``encode_blob`` (failing the assertion).
    With ``mv.nbytes``, the outer probe fires the EncodeError before
    reaching ``encode_blob``.
    """
    import array as _array

    # 8.4M elements of uint64 = ~67 MiB nbytes, exceeds the cap;
    # but len(mv) ~= 8.4M, well under the byte-cap if read with len().
    element_count = _MAX_BLOB_SIZE // 8 + 1
    mv = memoryview(_array.array("Q", [0] * element_count))
    assert mv.itemsize == 8
    assert len(mv) == element_count
    assert mv.nbytes == element_count * 8
    assert mv.nbytes > _MAX_BLOB_SIZE
    assert len(mv) < _MAX_BLOB_SIZE  # len-based probe would pass; nbytes-based catches it

    # Patch both:
    #   1. _reject_non_byte_format_memoryview to no-op — simulate the
    #      future refactor that removes / re-orders the format check.
    #   2. encode_blob to raise an AssertionError if reached — the
    #      outer cap-before-materialise probe must fire first.
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
    """The outer EncodeError message must name the byte count
    (mv.nbytes), not the element count (len(mv)). A caller debugging
    the cap with a multi-byte memoryview needs the byte number to
    match the cap's units."""
    import array as _array

    element_count = _MAX_BLOB_SIZE // 8 + 1
    mv = memoryview(_array.array("Q", [0] * element_count))
    expected_nbytes = element_count * 8

    # Patch encode_blob so only the outer probe's message can fire.
    def _stub(*args: object, **kwargs: object) -> bytes:
        return b""

    with (
        patch("dqlitewire.types._reject_non_byte_format_memoryview"),
        patch("dqlitewire.types.encode_blob", side_effect=_stub),
        pytest.raises(EncodeError) as exc_info,
    ):
        encode_value(mv, ValueType.BLOB)

    # The outer probe must cite the byte count, not the element count.
    assert str(expected_nbytes) in str(exc_info.value)
    assert f"Blob length {len(mv)} " not in str(exc_info.value)


def test_blob_cap_probe_bytes_still_uses_len() -> None:
    """Negative twin: for ``bytes`` / ``bytearray`` / ``mmap`` (no
    .nbytes attribute), the probe falls back to ``len(value)`` which
    is byte-correct by type."""
    payload = b"\x00" * (_MAX_BLOB_SIZE + 1)
    with pytest.raises(EncodeError, match=r"Blob length \d+ exceeds maximum"):
        encode_value(payload, ValueType.BLOB)
