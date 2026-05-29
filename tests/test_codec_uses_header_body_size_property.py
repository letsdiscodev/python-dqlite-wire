"""decode_bytes and decode_continuation must use the ``Header.body_size`` property,
not inline ``size_words * 8``, keeping WORD_SIZE/body_size as the single source of
truth and giving the property a production caller.
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
