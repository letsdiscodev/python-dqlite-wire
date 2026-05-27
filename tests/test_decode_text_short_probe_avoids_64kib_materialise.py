"""Pin: ``decode_text``'s memoryview branch peeks a SHORT probe
window first; only the long-cell case pays the 64 KiB
materialise.

The prior shape always materialised
``min(data_len, max_size + 1, _TEXT_ONE_SHOT_MAX)`` bytes per
cell — which collapsed to 64 KiB for any cell sitting in a
multi-MiB body buffer (a 1M-row RowsResponse is the canonical
case). A 16-byte TEXT cell at offset 100 in a 1 MiB body paid
~64 KiB of transient allocation per cell × 1M cells = ~64 GiB
of allocator churn on the loop thread purely to support a
``bytes.find`` that would have terminated in 16 bytes.

The fix peeks a short window (a few hundred bytes) first; when
NUL is found there — the common case for short TEXT cells —
the function returns without ever materialising the 64 KiB
probe. Cells that genuinely exceed the short probe escalate to
the existing 64 KiB path. The C reference uses ``strnlen`` on
the cursor's in-place buffer (zero allocation); this is the
Python adaptive-probe equivalent that mirrors the property
without requiring a callsite-API change.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.types import decode_text


def test_decode_text_short_cell_in_long_buffer_does_not_materialise_64kib() -> None:
    """A 16-byte TEXT cell inside a 1 MiB memoryview must be
    decoded without materialising the full 64 KiB probe window.

    We assert this structurally by patching ``bytes`` in the
    module's namespace to track the largest argument it received.
    Pre-fix: ``bytes(data[:65536])`` runs every call, so the max
    materialise size is ~64 KiB. Post-fix: the short probe stays
    well under 1 KiB.
    """
    from dqlitewire import types as types_mod

    # 16-byte TEXT cell at the head of a 1 MiB buffer. Decode with
    # the DEFAULT ``max_size`` (the wire's 64 MiB row-cell ceiling)
    # so the existing probe-then-escalate path picks
    # ``min(data_len, max_size + 1, _TEXT_ONE_SHOT_MAX) = 65536``
    # — the 64 KiB materialise we want to avoid for short cells.
    payload = b"short-text\x00\x00\x00\x00\x00\x00" + b"\xff" * (1 << 20)
    view = memoryview(payload)

    largest_bytes_size: list[int] = [0]
    real_bytes = bytes

    def _tracking_bytes(arg: object = b"") -> bytes:
        if isinstance(arg, memoryview):
            largest_bytes_size[0] = max(largest_bytes_size[0], len(arg))
        return real_bytes(arg)  # type: ignore[call-overload, no-any-return]

    monkeypatch_token = types_mod.__dict__.get("bytes", real_bytes)
    types_mod.__dict__["bytes"] = _tracking_bytes
    try:
        text, consumed = decode_text(view, label="TEXT")
    finally:
        if monkeypatch_token is real_bytes:
            types_mod.__dict__.pop("bytes", None)
        else:
            types_mod.__dict__["bytes"] = monkeypatch_token

    assert text == "short-text"
    assert consumed == 16  # 10 chars + NUL + 5 pad = 16

    # Pre-fix: 65536 bytes materialised. Post-fix: short probe far
    # smaller (a few hundred bytes). Pin at 4 KiB as the upper
    # acceptable bound for the short probe — well under the 64 KiB
    # pre-fix value, well above any sensible short-probe choice.
    assert largest_bytes_size[0] <= 4096, (
        f"decode_text materialised {largest_bytes_size[0]} bytes for a "
        f"16-byte cell in a 1 MiB buffer; the short-probe optimisation "
        f"should keep per-cell materialise well under the 64 KiB "
        f"one-shot ceiling for short cells."
    )


def test_decode_text_long_cell_still_decodes_correctly() -> None:
    """Regression guard: a cell that EXCEEDS the short probe but
    fits in the 64 KiB one-shot ceiling must still decode
    correctly via the escalated path.
    """
    long_text = "x" * 8192
    payload = long_text.encode("utf-8") + b"\x00" + b"\x00" * 7
    view = memoryview(payload)
    text, consumed = decode_text(view, label="TEXT", max_size=16_384)
    assert text == long_text


def test_decode_text_very_long_cell_uses_chunked_path() -> None:
    """Regression guard: a cell that exceeds the 64 KiB one-shot
    ceiling escalates to the chunked path and still decodes."""
    from dqlitewire.types import _TEXT_ONE_SHOT_MAX

    very_long = "x" * (_TEXT_ONE_SHOT_MAX + 100)
    payload = very_long.encode("utf-8") + b"\x00" + b"\x00" * 7
    view = memoryview(payload)
    text, consumed = decode_text(view, label="TEXT", max_size=_TEXT_ONE_SHOT_MAX * 2)
    assert text == very_long


def test_decode_text_rejects_cell_exceeding_max_size_in_long_buffer() -> None:
    """A cell that fits in the buffer but exceeds ``max_size``
    raises ``DecodeError`` — verify the short-probe optimisation
    does not swallow the max-size check.
    """
    payload = b"a" * 100 + b"\x00" + b"\x00" * 7 + b"\xff" * 1024
    view = memoryview(payload)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        decode_text(view, label="TEXT", max_size=50)


def test_decode_text_rejects_unterminated_cell() -> None:
    """A buffer with no NUL terminator at all raises
    ``DecodeError("...not null-terminated")``. The short-probe
    optimisation must escalate to find the missing-NUL case.
    """
    payload = b"a" * 100  # no NUL anywhere
    view = memoryview(payload)
    with pytest.raises(DecodeError):
        decode_text(view, label="TEXT", max_size=1024)
