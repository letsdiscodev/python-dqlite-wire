"""decode_text peeks a short probe first; only long cells pay the 64 KiB materialise."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.types import decode_text


def test_decode_text_short_cell_in_long_buffer_does_not_materialise_64kib() -> None:
    """A 16-byte TEXT cell in a 1 MiB memoryview decodes without materialising the 64 KiB probe.

    Patches ``bytes`` in the module namespace to track the largest argument it received.
    """
    from dqlitewire import types as types_mod

    # Default max_size makes the probe-then-escalate path pick the
    # 65536-byte one-shot we want to avoid for short cells.
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

    # 4 KiB bound: well under the 64 KiB one-shot, above any sensible short-probe choice.
    assert largest_bytes_size[0] <= 4096, (
        f"decode_text materialised {largest_bytes_size[0]} bytes for a "
        f"16-byte cell in a 1 MiB buffer; the short-probe optimisation "
        f"should keep per-cell materialise well under the 64 KiB "
        f"one-shot ceiling for short cells."
    )


def test_decode_text_long_cell_still_decodes_correctly() -> None:
    """A cell over the short probe but within the 64 KiB one-shot decodes via the escalated path."""
    long_text = "x" * 8192
    payload = long_text.encode("utf-8") + b"\x00" + b"\x00" * 7
    view = memoryview(payload)
    text, consumed = decode_text(view, label="TEXT", max_size=16_384)
    assert text == long_text


def test_decode_text_very_long_cell_uses_chunked_path() -> None:
    """A cell over the 64 KiB one-shot ceiling escalates to the chunked path and still decodes."""
    from dqlitewire.types import _TEXT_ONE_SHOT_MAX

    very_long = "x" * (_TEXT_ONE_SHOT_MAX + 100)
    payload = very_long.encode("utf-8") + b"\x00" + b"\x00" * 7
    view = memoryview(payload)
    text, consumed = decode_text(view, label="TEXT", max_size=_TEXT_ONE_SHOT_MAX * 2)
    assert text == very_long


def test_decode_text_rejects_cell_exceeding_max_size_in_long_buffer() -> None:
    """Short-probe optimisation must not swallow the max-size check."""
    payload = b"a" * 100 + b"\x00" + b"\x00" * 7 + b"\xff" * 1024
    view = memoryview(payload)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        decode_text(view, label="TEXT", max_size=50)


def test_decode_text_rejects_unterminated_cell() -> None:
    """No NUL terminator: short-probe optimisation must escalate to find the missing-NUL case."""
    payload = b"a" * 100  # no NUL anywhere
    view = memoryview(payload)
    with pytest.raises(DecodeError):
        decode_text(view, label="TEXT", max_size=1024)
