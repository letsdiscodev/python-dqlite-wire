"""Pin: ``decode_row_header`` docstring accurately documents the
strictly-tighter-than-Go marker validation.

The function validates all 8 bytes of the marker word against the
full uint64 sentinel (matching the upstream C sentinel comparison),
not just the first byte (which the Go reference does). The docstring
must reflect this so a future contributor reading "matching the Go
reference" doesn't relax the check to first-byte-only and silently
accept torn/corrupt markers like ``0xff 0x00 ...``.
"""

from __future__ import annotations

from dqlitewire.exceptions import DecodeError
from dqlitewire.tuples import decode_row_header


def test_decode_row_header_docstring_does_not_claim_matching_go() -> None:
    doc = decode_row_header.__doc__ or ""
    # The previous wording said "matching the Go reference
    # implementation" — false: we validate all 8 bytes; Go validates
    # only the first byte. Pin against the inverted-claim regression.
    assert "matching the Go reference" not in doc, (
        "Docstring must not claim Go-reference parity — we are strictly "
        "tighter (all 8 bytes vs Go's first byte)"
    )


def test_decode_row_header_docstring_documents_8_byte_validation() -> None:
    doc = decode_row_header.__doc__ or ""
    assert "8 bytes" in doc or "all 8" in doc, (
        "Docstring should explicitly document the 8-byte sentinel comparison"
    )


def test_decode_row_header_torn_marker_rejected_not_silently_consumed() -> None:
    """Behavioural pin: a torn marker (first byte 0xFF but trailing
    bytes diverge) must NOT be silently treated as DONE — it should
    fall through to the type-header decode path."""
    # 0xFF in first byte, rest zeros — Go's first-byte check would
    # accept this as DONE; we must not.
    torn = b"\xff\x00\x00\x00\x00\x00\x00\x00"
    # column_count=1 means header expected length is 1 nibble = 1 byte
    # padded to 8. The first nibble is 0xF (RAW=15) — invalid type code.
    # Either way, the function MUST NOT return RowMarker.DONE.
    try:
        result = decode_row_header(torn, column_count=1)
    except DecodeError:
        # DecodeError is the documented outcome for an underspecified
        # input that doesn't form a valid type-header tuple. Narrowing
        # the catch from bare ``Exception`` ensures a refactor-introduced
        # ``RuntimeError`` (or any non-decode failure mode) propagates
        # to surface the bug, instead of being silently absorbed as
        # if it were "decode error is fine here".
        return
    types_or_marker, _ = result
    from dqlitewire.tuples import RowMarker

    assert types_or_marker is not RowMarker.DONE, (
        "Torn marker (first byte 0xFF, rest zeros) must not be accepted as DONE"
    )
