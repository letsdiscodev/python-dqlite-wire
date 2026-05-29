"""``encode_value`` for BLOB checks ``len(value)`` against the cap before
materialising via ``bytes(value)``, so a hostile-large memoryview/mmap is
rejected without first copying its whole contents."""

from unittest.mock import patch

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


def test_encode_value_blob_len_probe_rejects_via_memoryview() -> None:
    """A memoryview just over the cap is rejected on the ``len()`` probe before
    ``bytes(value)`` runs."""
    # Tiny patched cap exercises the same path as the production cap cheaply.
    with patch("dqlitewire.types._MAX_BLOB_SIZE", 8):
        buf = memoryview(bytearray(16))
        with pytest.raises(EncodeError, match="exceeds maximum"):
            encode_value(buf, ValueType.BLOB)


def test_encode_value_blob_len_probe_rejects_via_bytearray() -> None:
    """``bytearray`` exercises the same cap-before-materialise branch."""
    with patch("dqlitewire.types._MAX_BLOB_SIZE", 8):
        buf = bytearray(16)
        with pytest.raises(EncodeError, match="exceeds maximum"):
            encode_value(buf, ValueType.BLOB)


def test_encode_value_blob_at_cap_accepts() -> None:
    """A buffer at exactly the cap encodes fine."""
    with patch("dqlitewire.types._MAX_BLOB_SIZE", 16):
        buf = bytearray(16)
        encoded, vt = encode_value(buf, ValueType.BLOB)
        assert vt == ValueType.BLOB
        assert isinstance(encoded, bytes)


def test_encode_value_blob_materialise_failure_wrapped() -> None:
    """If ``len()`` passes but ``bytes(value)`` raises (released memoryview /
    closed mmap), the wrap surfaces ``EncodeError`` not bare ValueError."""
    buf = bytearray(16)
    view = memoryview(buf)
    view.release()
    with pytest.raises(EncodeError, match="Cannot materialise BLOB"):
        encode_value(view, ValueType.BLOB)
