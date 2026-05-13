"""Doc-pin: ``_validate_decoded_schema``'s docstring describes the V1
count width correctly.

Earlier wording said the V1 count was ``uint16`` and that
``_MAX_PARAM_COUNT`` was "the V1 uint16 count, less the marker reserve".
Both claims were wrong: the V1 (TUPLE__PARAMS32) count is a uint32 on
the wire (see ``tuples.py:147`` and upstream ``tuple.c:67-69``), and
``_MAX_PARAM_COUNT`` is ``SQLITE_MAX_VARIABLE_NUMBER`` from SQLite's
engine-side bind cap — not any wire-level marker reserve.

This pin guards against either claim re-appearing in the docstring on
a future refactor.
"""

from __future__ import annotations

from dqlitewire.messages.requests import _validate_decoded_schema


def test_validate_decoded_schema_docstring_does_not_claim_v1_count_is_uint16() -> None:
    doc = _validate_decoded_schema.__doc__ or ""
    assert "V1 uint16 count" not in doc, (
        "V1 (TUPLE__PARAMS32) count is uint32 on the wire — see "
        "tuples.py:147 and tuple.c:67-69. Docstring must not claim "
        "the V1 count is uint16."
    )


def test_validate_decoded_schema_docstring_does_not_claim_marker_reserve() -> None:
    doc = _validate_decoded_schema.__doc__ or ""
    assert "marker reserve" not in doc, (
        "_MAX_PARAM_COUNT = SQLITE_MAX_VARIABLE_NUMBER from SQLite's "
        "engine-side bind cap (tuples.py:_MAX_PARAM_COUNT). There is "
        "no wire-level marker reserve in the V1 params header."
    )
