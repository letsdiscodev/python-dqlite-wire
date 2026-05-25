"""Pin: the asymmetry between ``decode_text(errors="surrogateescape")``
and the strict :func:`encode_text` is documented on BOTH helpers.

A caller reaching for ``surrogateescape`` based on prior Python idiom
(round-trip safe smuggle of opaque bytes) discovers at first
``encode_text`` call that the encoder refuses the lone surrogates.
The docstring is the contract surface for that caveat; pin both
helper docstrings so a future rewrite cannot silently drop the
warning.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.types import decode_text, encode_text


def test_decode_text_docstring_warns_about_surrogateescape_round_trip() -> None:
    doc = decode_text.__doc__ or ""
    assert "surrogateescape" in doc
    assert "round-trippable" in doc or "round-trip" in doc


def test_encode_text_docstring_documents_one_way_pair_with_decode_text() -> None:
    doc = encode_text.__doc__ or ""
    assert "surrogateescape" in doc
    assert "BLOB" in doc, "encoder docstring must point opaque-byte callers at BLOB"


def test_surrogateescape_decoded_string_fails_encode() -> None:
    raw = b"caf\xe9\x00\x00\x00\x00\x00"
    s, _ = decode_text(raw, errors="surrogateescape")
    assert "\udce9" in s
    with pytest.raises(EncodeError, match="surrogate"):
        encode_text(s)
