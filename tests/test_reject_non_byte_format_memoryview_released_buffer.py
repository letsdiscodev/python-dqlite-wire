"""Pin: ``_reject_non_byte_format_memoryview`` falls through silently
when ``.format`` access raises EITHER ``ValueError`` or
``BufferError``.

The current CPython implementation raises ``ValueError`` from
``memoryview.format`` on a released buffer (per ``Objects/
memoryobject.c::mbuf_check_view``). A future CPython that narrows
the released-buffer error class to ``BufferError`` (for symmetry with
``bytes(released_mv)`` which already raises ``BufferError``) would
propagate an unwrapped exception through the helper, breaking the
"all encode failures surface as EncodeError" convention and bypassing
the downstream ``bytes(value)`` materialise step that produces the
canonical ``"Cannot materialise BLOB"`` diagnostic.

Broadening the catch to ``(ValueError, BufferError)`` makes the
helper exception-class-agnostic — the helper's correctness no longer
hinges on a CPython implementation detail the package does not own.
"""

from __future__ import annotations

import array
from typing import Any

from dqlitewire.types import _reject_non_byte_format_memoryview


def test_released_memoryview_falls_through_silently() -> None:
    """A released ``memoryview`` whose ``.format`` raises (current
    CPython: ValueError) must fall through silently so the downstream
    materialise step surfaces the canonical EncodeError."""
    mv = memoryview(array.array("i", [1, 2, 3]))
    mv.release()
    # Pre-release the format would have raised EncodeError for non-byte
    # format; post-release the helper must fall through silently
    # regardless.
    _reject_non_byte_format_memoryview(mv)  # no exception


def test_format_raising_buffer_error_falls_through_silently() -> None:
    """Simulate a future CPython narrowing where ``.format`` raises
    ``BufferError`` (instead of ``ValueError``) on a released buffer.
    The helper must still fall through silently."""

    class _FakeMv:
        @property
        def format(self) -> Any:
            raise BufferError("simulated future-CPython narrowing")

        @property
        def itemsize(self) -> int:
            return 1

    # The helper takes a memoryview by static type; the runtime check
    # is duck-typed on .format / .itemsize so the fake stands in.
    _reject_non_byte_format_memoryview(_FakeMv())  # type: ignore[arg-type]


def test_format_raising_value_error_falls_through_silently() -> None:
    """Sanity: an object whose ``.format`` raises ``ValueError``
    (current CPython contract) also falls through silently."""

    class _FakeMv:
        @property
        def format(self) -> Any:
            raise ValueError("operation forbidden on released memoryview object")

        @property
        def itemsize(self) -> int:
            return 1

    _reject_non_byte_format_memoryview(_FakeMv())  # type: ignore[arg-type]
