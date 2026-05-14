"""``decode_text`` defends against attacker-controlled lengths past
the cap on both the bytes branch and the chunked-memoryview branch.
The chunked branch is well-tested; the bytes branch's not-terminated
and ``null_pos > max_size`` post-cap arms were uncovered.

Pin both:
- A bytes payload whose NUL lands within the scan window but past
  ``max_size`` raises with a length-cited ``DecodeError``.
- The same shape via the bytes-branch ``if null_pos > max_size:``
  post-cap check (the ``raise DecodeError(f"{label} length {null_pos}
  exceeds maximum ({max_size})")`` arm) is exercised by the existing
  chunked path tests; here we assert the bytes path emits the same
  error.
"""

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.types import decode_text


def test_bytes_branch_post_cap_check_rejects() -> None:
    """``max_size=10`` with a 12-byte payload whose NUL lies at byte
    11 — within the ``max_size + 1 = 11`` scan window but past
    ``max_size``. Pre-fix this was caught by an outer
    ``null_pos > max_size`` check after the decode; the bytes-branch
    ``if null_pos > max_size:`` short-circuits BEFORE the UTF-8
    decode."""
    # 11 bytes of 'X' followed by NUL.
    data = b"X" * 11 + b"\x00"
    with pytest.raises(DecodeError, match="exceeds maximum"):
        decode_text(data, max_size=10)


def test_bytes_branch_at_cap_accepts() -> None:
    """Boundary: NUL at exactly ``max_size`` is the cap inclusive."""
    data = b"X" * 10 + b"\x00" + b"\x00" * 5  # padded to word
    text, _ = decode_text(data, max_size=10)
    assert text == "X" * 10
