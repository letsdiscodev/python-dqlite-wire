"""Pin: ``_reject_non_byte_format_memoryview`` falls through silently
when ``.format`` raises EITHER ``ValueError`` or ``BufferError``, so the
helper does not hinge on which exception a given CPython raises for a
released buffer (downstream ``bytes(value)`` then surfaces EncodeError)."""

from __future__ import annotations

import array
from typing import Any

from dqlitewire.types import _reject_non_byte_format_memoryview


def test_released_memoryview_falls_through_silently() -> None:
    """A released ``memoryview`` (non-byte format) must fall through
    silently so the downstream materialise step surfaces EncodeError."""
    mv = memoryview(array.array("i", [1, 2, 3]))
    mv.release()
    _reject_non_byte_format_memoryview(mv)  # no exception


def test_format_raising_buffer_error_falls_through_silently() -> None:
    """``.format`` raising ``BufferError`` must still fall through silently."""

    class _FakeMv:
        @property
        def format(self) -> Any:
            raise BufferError("simulated future-CPython narrowing")

        @property
        def itemsize(self) -> int:
            return 1

    # Runtime check is duck-typed on .format / .itemsize, so the fake stands in.
    _reject_non_byte_format_memoryview(_FakeMv())  # type: ignore[arg-type]


def test_format_raising_value_error_falls_through_silently() -> None:
    """``.format`` raising ``ValueError`` also falls through silently."""

    class _FakeMv:
        @property
        def format(self) -> Any:
            raise ValueError("operation forbidden on released memoryview object")

        @property
        def itemsize(self) -> int:
            return 1

    _reject_non_byte_format_memoryview(_FakeMv())  # type: ignore[arg-type]
