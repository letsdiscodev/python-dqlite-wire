"""decode_text bytes-branch post-cap arms: not-null-terminated and length-exceeds-maximum."""

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.types import decode_text


def test_bytes_branch_post_cap_check_rejects() -> None:
    data = b"X" * 11 + b"\x00"
    with pytest.raises(DecodeError, match="exceeds maximum"):
        decode_text(data, max_size=10)


def test_bytes_branch_at_cap_accepts() -> None:
    data = b"X" * 10 + b"\x00" + b"\x00" * 5  # padded to word
    text, _ = decode_text(data, max_size=10)
    assert text == "X" * 10
