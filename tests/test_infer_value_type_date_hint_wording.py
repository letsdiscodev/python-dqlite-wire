"""Pin: ``_infer_value_type``'s rejection hint for unsupported types
must NOT suggest ``int (UNIXTIME)`` as a universal mapping. The
previous wording ("datetime/date/etc. must convert to str (ISO8601)
or int (UNIXTIME)") is wrong for ``datetime.date`` and
``datetime.time``: dates have no time component, so
``int(date.toordinal())`` or ``int(datetime.combine(...).timestamp())``
both silently produce garbage on UNIXTIME round-trip (consumer-side
``datetime.utcfromtimestamp`` returns a year-0001 timestamp from a
proleptic Gregorian ordinal).

The corrected wording must call out the per-type mapping explicitly:
datetime gets BOTH options (ISO8601 + UNIXTIME for aware datetimes),
date and time get ISO8601 ONLY.
"""

from __future__ import annotations

import datetime

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.types import encode_value


def test_date_rejection_hint_does_not_suggest_unixtime() -> None:
    """``datetime.date`` has no defensible UNIXTIME mapping. The
    rejection message must specifically call this out so a caller
    cannot follow the hint into a silent ordinal-vs-epoch bug."""
    d = datetime.date(2024, 1, 15)
    with pytest.raises(EncodeError) as exc_info:
        encode_value(d)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    # The hint must explicitly disclaim a UNIXTIME mapping for date.
    assert "no UNIXTIME" in msg or "no time component" in msg
    # And it must point to .isoformat() as the correct conversion.
    assert ".isoformat()" in msg


def test_datetime_rejection_hint_includes_both_options() -> None:
    """For ``datetime.datetime``, both ISO8601 and UNIXTIME mappings
    are documented."""
    dt = datetime.datetime(2024, 1, 15, 12, 30)
    with pytest.raises(EncodeError) as exc_info:
        encode_value(dt)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert ".isoformat()" in msg
    assert ".timestamp()" in msg or "UNIXTIME" in msg


def test_time_rejection_hint_does_not_suggest_unixtime() -> None:
    """``datetime.time`` also has no UNIXTIME mapping."""
    t = datetime.time(12, 30)
    with pytest.raises(EncodeError) as exc_info:
        encode_value(t)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert ".isoformat()" in msg
