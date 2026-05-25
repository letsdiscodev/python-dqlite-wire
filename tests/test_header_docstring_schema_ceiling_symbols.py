"""Pin: ``Header`` docstring names the real schema-ceiling symbols
``_REQUEST_MAX_SCHEMA`` / ``_RESPONSE_MAX_SCHEMA`` (in ``codec.py``)
and does not cite the long-stale ``MAX_SUPPORTED_SCHEMA`` symbol
(which never existed in the source tree).
"""

from __future__ import annotations

from dqlitewire.messages.base import Header


def test_header_docstring_does_not_reference_stale_symbol() -> None:
    doc = Header.__doc__ or ""
    assert "MAX_SUPPORTED_SCHEMA" not in doc


def test_header_docstring_cites_real_ceiling_symbols() -> None:
    doc = Header.__doc__ or ""
    assert "_REQUEST_MAX_SCHEMA" in doc
    assert "_RESPONSE_MAX_SCHEMA" in doc
