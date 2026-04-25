"""Coverage for ``dqlitewire.messages.base`` invariants — ``Header``
decode boundaries and ``Message.encode()``'s body-alignment check.

The wire layer's tests historically exercised these surfaces only
transitively through concrete Request / Response subclasses. Pin
the load-bearing invariants directly:

- ``Message.encode()`` must reject a developer-introduced
  ``encode_body()`` that returns a non-word-aligned body
  (mirrors the C server's strict ``dqlite_assert(_n % 8 == 0)``).
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.base import Message


class TestMessageEncodeAlignment:
    def test_encode_rejects_body_with_unaligned_length(self) -> None:
        """A custom ``Message`` subclass whose ``encode_body()``
        returns a length not divisible by ``WORD_SIZE`` (8 bytes)
        must raise ``EncodeError`` — silently emitting a misaligned
        frame would be rejected by the C peer's strict-decode and
        is exactly the kind of regression a custom subclass might
        introduce."""

        class _BadMessage(Message):
            MSG_TYPE: ClassVar[int] = 99

            def encode_body(self) -> bytes:
                return b"abc"  # 3 bytes — not 8-aligned

            @classmethod
            def decode_body(cls, data: bytes, schema: int = 0) -> _BadMessage:  # pragma: no cover
                raise NotImplementedError

        with pytest.raises(EncodeError, match=r"must be \d+-aligned"):
            _BadMessage().encode()

    def test_encode_accepts_body_with_aligned_length(self) -> None:
        """Sanity: an 8-byte body is accepted (negative test for the
        guard above — confirms the alignment check is the only thing
        that would have rejected the bad case)."""

        class _GoodMessage(Message):
            MSG_TYPE: ClassVar[int] = 99

            def encode_body(self) -> bytes:
                return b"\x00" * 8

            @classmethod
            def decode_body(cls, data: bytes, schema: int = 0) -> _GoodMessage:  # pragma: no cover
                raise NotImplementedError

        encoded = _GoodMessage().encode()
        assert isinstance(encoded, bytes)
        assert len(encoded) > 8  # header + body
