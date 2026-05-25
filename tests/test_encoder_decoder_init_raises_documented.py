"""Pin: ``MessageEncoder.__init__`` and ``MessageDecoder.__init__``
docstrings document the construction-time failure modes via a
canonical ``Raises:`` block.

Both constructors raise on invalid kwargs (``HandshakeError`` on
unsupported version; ``ValueError`` on negative caps; ``DecodeError``
on a bad ``unknown_role_policy``). Without the docstring contract a
caller running ``help(MessageDecoder)`` would learn the failure modes
only via traceback.
"""

from __future__ import annotations

from dqlitewire.codec import MessageDecoder, MessageEncoder


def test_message_encoder_init_docstring_documents_raises() -> None:
    doc = MessageEncoder.__init__.__doc__ or ""
    assert "Raises:" in doc
    assert "HandshakeError" in doc
    assert "ValueError" in doc


def test_message_decoder_init_docstring_documents_raises() -> None:
    doc = MessageDecoder.__init__.__doc__ or ""
    assert "Raises:" in doc
    assert "HandshakeError" in doc
    assert "ValueError" in doc
    assert "DecodeError" in doc
