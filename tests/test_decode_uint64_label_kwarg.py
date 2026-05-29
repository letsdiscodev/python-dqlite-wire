"""decode_uint64 accepts a label= kwarg that interpolates into the truncation diagnostic."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.requests import OpenRequest
from dqlitewire.types import decode_uint64


def test_decode_uint64_default_label_preserves_historical_message() -> None:
    with pytest.raises(DecodeError, match="Need 8 bytes for uint64, got 3"):
        decode_uint64(b"abc")


def test_decode_uint64_label_kwarg_appears_in_diagnostic() -> None:
    with pytest.raises(DecodeError, match=r"Need 8 bytes for OpenRequest\.flags, got 3"):
        decode_uint64(b"abc", label="OpenRequest.flags")


def test_open_request_truncated_flags_diagnostic_names_field() -> None:
    """Body decodes db name then truncates inside flags; error must name both class and field."""
    from dqlitewire.types import encode_text

    body = encode_text("test.db") + b"\x00\x00\x00"
    with pytest.raises(DecodeError, match="OpenRequest.flags"):
        OpenRequest.decode_body(body)
