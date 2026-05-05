"""``encode_value`` for BLOB probes ``len(value)`` BEFORE materialising
via ``bytes(value)`` so a hostile-large memoryview / mmap that knows
its length without copying is rejected up front.

Without the cap-before-materialise probe, a 100 MiB ``memoryview``
would be copied via ``bytes(value)`` (full 100 MiB allocation) only
to be rejected by the encoder's cap check. The probe at
``types.py:546-550`` shortcuts that copy.

The materialise-then-cap branch IS covered by ``test_types.py`` and
``test_blob_max_size_kwarg.py``; this is the orthogonal arm.
"""

from unittest.mock import patch

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


def test_encode_value_blob_len_probe_rejects_via_memoryview() -> None:
    """A real memoryview backed by a buffer just over the
    test-overridden cap exercises the ``len()`` probe path: the
    rejection cites the ``len()``-reported length; the
    ``bytes(value)`` materialise step is never reached."""
    # Patch the module-level cap to a tiny value so the test stays
    # cheap: a 16-byte buffer with a 8-byte cap exercises the same
    # code path as a 64 MiB buffer with the production cap.
    with patch("dqlitewire.types._MAX_BLOB_SIZE", 8):
        buf = memoryview(bytearray(16))
        with pytest.raises(EncodeError, match="exceeds maximum"):
            encode_value(buf, ValueType.BLOB)


def test_encode_value_blob_len_probe_rejects_via_bytearray() -> None:
    """``bytearray`` also has cheap ``len()`` and exercises the same
    cap-before-materialise branch."""
    with patch("dqlitewire.types._MAX_BLOB_SIZE", 8):
        buf = bytearray(16)
        with pytest.raises(EncodeError, match="exceeds maximum"):
            encode_value(buf, ValueType.BLOB)


def test_encode_value_blob_at_cap_accepts() -> None:
    """Boundary: a buffer at exactly the cap encodes fine."""
    with patch("dqlitewire.types._MAX_BLOB_SIZE", 16):
        buf = bytearray(16)
        encoded, vt = encode_value(buf, ValueType.BLOB)
        assert vt == ValueType.BLOB
        assert isinstance(encoded, bytes)


def test_encode_value_blob_materialise_failure_wrapped() -> None:
    """If ``len()`` reports OK but ``bytes(value)`` raises (closed
    mmap / released memoryview), the materialise-side wrap surfaces
    ``EncodeError`` not the bare ValueError/BufferError.

    Released memoryview is the simplest to reproduce."""
    buf = bytearray(16)
    view = memoryview(buf)
    view.release()
    with pytest.raises(EncodeError, match="Cannot materialise BLOB"):
        encode_value(view, ValueType.BLOB)
