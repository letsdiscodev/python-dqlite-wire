"""Coverage for ``dqlitewire.messages.base`` invariants — ``Header``
decode boundaries and ``Message.encode()``'s body-alignment check.

The wire layer's tests historically exercised these surfaces only
transitively through concrete Request / Response subclasses. Pin
the load-bearing invariants directly:

- ``Message.encode()`` must reject a developer-introduced
  ``encode_body()`` that returns a non-word-aligned body
  (mirrors the C server's strict ``dqlite_assert(_n % 8 == 0)``).
- ``Header.encode()`` must wrap ``struct.error`` as
  ``EncodeError`` with the underlying exception preserved as
  ``__cause__`` (belt-and-braces guard behind the ``__post_init__``
  range checks).
"""

from __future__ import annotations

import struct
from typing import ClassVar

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.base import Header, Message


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


class TestHeaderEncodeStructErrorWrap:
    """Pin ``Header.encode()``'s ``struct.error`` → ``EncodeError``
    wrap at ``messages/base.py:80-81``.

    Through normal construction, ``Header.__post_init__`` rejects any
    value that would overflow the ``<IBBH`` pack format, so the encode
    arm is a belt-and-braces defensive guard. The wire's caller layer
    catches ``EncodeError`` — not ``struct.error`` — so a regression
    that dropped the wrap (or its ``from e`` chain, or the
    ``"Failed to encode header"`` message prefix) would leak the
    lower-level exception out of the wire boundary and erode forensic
    detail. The bypass below mutates a frozen+slots dataclass via
    ``object.__setattr__`` to reach the otherwise-unreachable arm.
    """

    def _bypass_size_words(self, value: int) -> Header:
        """Construct a valid ``Header`` and then overwrite
        ``size_words`` post-construction, bypassing ``__post_init__``
        so the encode-time ``struct.pack`` raises."""
        h = Header(size_words=1, msg_type=1, schema=0, reserved=0)
        object.__setattr__(h, "size_words", value)
        return h

    def test_encode_wraps_struct_error_as_encode_error(self) -> None:
        hdr = self._bypass_size_words(2**32)
        with pytest.raises(EncodeError, match="Failed to encode header"):
            hdr.encode()

    def test_encode_preserves_struct_error_as_cause(self) -> None:
        """``EncodeError.__cause__`` carries the underlying
        ``struct.error`` so operators see the original overflow signal
        (e.g. the ``'I' format requires ...`` text from CPython)."""
        hdr = self._bypass_size_words(2**32)
        try:
            hdr.encode()
        except EncodeError as exc:
            assert isinstance(exc.__cause__, struct.error)
        else:
            pytest.fail("expected EncodeError")
