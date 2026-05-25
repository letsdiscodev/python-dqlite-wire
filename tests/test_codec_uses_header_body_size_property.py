"""Pin: ``MessageDecoder.decode_bytes`` and
``MessageDecoder.decode_continuation`` use the canonical
``Header.body_size`` property for words-to-bytes conversion rather
than inline ``size_words * 8``.

The SSOT discipline keeps ``WORD_SIZE`` and ``Header.body_size`` as
the single source of truth for the conversion. Inlining the literal
``* 8`` works today but denies ``Header.body_size`` its production
caller and weakens the constant's grep-ability — a future cleanup
wave running dead-code linting could plausibly mistake the property
for unused. This test guards against the inline literal returning.
"""

from __future__ import annotations

import inspect

from dqlitewire.codec import MessageDecoder


def test_decode_bytes_uses_header_body_size_property() -> None:
    src = inspect.getsource(MessageDecoder.decode_bytes)
    assert "header.body_size" in src
    assert "size_words * 8" not in src


def test_decode_continuation_uses_header_body_size_property() -> None:
    src = inspect.getsource(MessageDecoder.decode_continuation)
    assert "header.body_size" in src
    assert "size_words * 8" not in src
