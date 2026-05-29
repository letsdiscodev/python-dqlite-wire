"""Pin: ``Header.__post_init__`` accepts the full uint8 range for
``schema`` (0..255), not just the per-message-type ceiling.

The docstring previously said "schema (0 or 1)" matching the
per-message-type ceiling, but ``__post_init__`` only range-validates
``0 <= schema < 2**8``. The narrower per-type ceiling is enforced at
decode-dispatch in ``codec.py``, not at construction time. The
docstring now clarifies this layering; pin against a future
re-tightening that would slip a circular import back into ``base.py``.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.base import Header


def test_header_construction_accepts_schema_above_per_type_ceiling() -> None:
    """A user constructing Header(schema=42) must succeed at the
    base.py layer — narrower ceilings are enforced by codec.py
    dispatch, not __post_init__."""
    h = Header(size_words=1, msg_type=0, schema=42, reserved=0)
    assert h.schema == 42


def test_header_construction_rejects_schema_above_uint8() -> None:
    with pytest.raises(EncodeError, match="schema"):
        Header(size_words=1, msg_type=0, schema=256, reserved=0)


def test_header_construction_rejects_negative_schema() -> None:
    with pytest.raises(EncodeError, match="schema"):
        Header(size_words=1, msg_type=0, schema=-1, reserved=0)
