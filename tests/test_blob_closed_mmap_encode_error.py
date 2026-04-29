"""Pin: a closed ``mmap.mmap`` (or released ``memoryview``) on the BLOB
encode path must surface as ``EncodeError``, not raw ``ValueError``
or ``BufferError``.

The wire-layer convention: every encode failure raises
``EncodeError``. ``bytes(closed_mmap)`` raises
``ValueError("mmap closed or invalid")`` from CPython's
``mmap_buffer_getbuf``; ``bytes(released_memoryview)`` raises
``BufferError``. Both must be wrapped at the wire boundary so callers
using ``except EncodeError`` don't silently miss the failure.
"""

from __future__ import annotations

import mmap

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


def test_encode_value_explicit_blob_closed_mmap_raises_encode_error() -> None:
    payload = b"hello"
    mm = mmap.mmap(-1, len(payload))
    mm.write(payload)
    mm.close()
    with pytest.raises(EncodeError):
        encode_value(mm, ValueType.BLOB)


def test_encode_value_inferred_blob_closed_mmap_raises_encode_error() -> None:
    payload = b"hello"
    mm = mmap.mmap(-1, len(payload))
    mm.write(payload)
    mm.close()
    with pytest.raises(EncodeError):
        encode_value(mm)


def test_encode_value_explicit_blob_released_memoryview_raises_encode_error() -> None:
    """A ``memoryview`` released via ``.release()`` raises
    ``BufferError`` from ``bytes(view)`` — same convention."""
    buf = bytearray(b"hello")
    view = memoryview(buf)
    view.release()
    with pytest.raises(EncodeError):
        encode_value(view, ValueType.BLOB)


def test_encode_value_explicit_blob_preserves_original_exception_via_cause() -> None:
    """The wrap should preserve the underlying error via ``__cause__``
    so debugging from the EncodeError still reaches the original."""
    payload = b"hello"
    mm = mmap.mmap(-1, len(payload))
    mm.write(payload)
    mm.close()
    try:
        encode_value(mm, ValueType.BLOB)
    except EncodeError as e:
        assert isinstance(e.__cause__, (ValueError, BufferError))
    else:  # pragma: no cover
        pytest.fail("expected EncodeError")
