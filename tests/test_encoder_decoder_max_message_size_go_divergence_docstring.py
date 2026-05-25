"""Pin: ``MessageEncoder`` and ``MessageDecoder`` ``__init__``
docstrings document the Go-parity asymmetry of the ``max_message_size``
cap (Python enforces per-frame; Go enforces only server-side via
``raft.log.entry_size_max``).
"""

from __future__ import annotations

from dqlitewire.codec import MessageDecoder, MessageEncoder


def test_message_encoder_docstring_cites_go_parity() -> None:
    doc = MessageEncoder.__init__.__doc__ or ""
    assert "Go-parity" in doc
    assert "raft.log.entry_size_max" in doc


def test_message_decoder_docstring_cites_go_parity() -> None:
    doc = MessageDecoder.__init__.__doc__ or ""
    assert "Go-parity" in doc
    assert "raft.log.entry_size_max" in doc
