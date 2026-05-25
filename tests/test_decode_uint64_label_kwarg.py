"""Pin: :func:`decode_uint64` accepts a ``label=`` kwarg that
interpolates into the truncation diagnostic so a caller decoding a
specific field gets a per-field error message — symmetric with
:func:`decode_text`'s existing ``label=`` discipline.

The first regression-pin verifies the specific cited example
(``OpenRequest.flags``); the second verifies the primitive-level
kwarg works in isolation; the third pins backwards-compat by
checking the historical default-label message wording.
"""

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
    """Build a body that decodes ``database name`` cleanly then
    truncates inside ``flags``. The resulting DecodeError must name
    both the request class and the field — historically the message
    was the bare ``Need 8 bytes for uint64`` with no class / field
    context."""
    from dqlitewire.types import encode_text

    body = encode_text("test.db") + b"\x00\x00\x00"
    with pytest.raises(DecodeError, match="OpenRequest.flags"):
        OpenRequest.decode_body(body)
