"""decode_text(errors="surrogateescape") is not round-trippable through strict encode_text."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.types import decode_text, encode_text


def test_surrogateescape_decoded_string_fails_encode() -> None:
    raw = b"caf\xe9\x00\x00\x00\x00\x00"
    s, _ = decode_text(raw, errors="surrogateescape")
    assert "\udce9" in s
    with pytest.raises(EncodeError, match="surrogate"):
        encode_text(s)
