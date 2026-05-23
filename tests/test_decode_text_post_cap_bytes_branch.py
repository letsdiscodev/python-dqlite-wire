"""``decode_text`` defends against attacker-controlled lengths past
the cap on both the bytes branch and the chunked-memoryview branch.
The chunked branch is well-tested; the bytes branch's
not-null-terminated and length-exceeds-maximum arms were uncovered.

The bytes branch caps the NUL scan at ``scan_end = min(len(data),
max_size + 1)``. If no NUL is found within that window but the
payload itself is longer than the window, that is the "length
exceeds maximum" shape. The boundary case (NUL at exactly
``max_size``) is accepted.
"""

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.types import decode_text


def test_bytes_branch_post_cap_check_rejects() -> None:
    """``max_size=10`` with a 12-byte payload whose NUL lies at byte
    11 — outside the ``scan_end = max_size + 1 = 11`` window
    (``find``'s ``end`` argument is exclusive). The
    ``null_pos < 0 and scan_end < len(data)`` arm raises
    ``DecodeError`` with the length-exceeds-maximum wording before
    any UTF-8 decode runs."""
    # 11 bytes of 'X' followed by NUL.
    data = b"X" * 11 + b"\x00"
    with pytest.raises(DecodeError, match="exceeds maximum"):
        decode_text(data, max_size=10)


def test_bytes_branch_at_cap_accepts() -> None:
    """Boundary: NUL at exactly ``max_size`` is the cap inclusive."""
    data = b"X" * 10 + b"\x00" + b"\x00" * 5  # padded to word
    text, _ = decode_text(data, max_size=10)
    assert text == "X" * 10
