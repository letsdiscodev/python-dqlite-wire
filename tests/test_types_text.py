"""Tests for TEXT encoding/decoding: caps, probes, error handling and label symmetry."""

from __future__ import annotations

from unittest import mock

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.limits import MAX_ADDRESS_SIZE, MAX_FAILURE_MESSAGE_SIZE, MAX_TEXT_VALUE_SIZE
from dqlitewire.messages.responses import FailureResponse, LeaderResponse
from dqlitewire.types import decode_text, decode_value, encode_text, encode_value

# ---- merged from test_decode_text_chunked_errors.py ----
# Error-path tests for the memoryview chunked branch of ``decode_text``
# (the one-shot branch already has null-terminator / invalid-UTF-8 tests).


def _align(n: int) -> int:
    """Bytes needed to pad n up to a multiple of 8 (word alignment)."""
    rem = n % 8
    return 0 if rem == 0 else 8 - rem


class TestDecodeTextChunkedErrors:
    def test_chunked_no_null_within_cap_raises_exceeds_maximum(self) -> None:
        """A 70 KiB non-NUL-terminated payload at ``max_size=4 KiB`` raises
        "exceeds maximum" (NOT "not null-terminated") and stops scanning at
        the cap rather than allocating megabytes of chunks first."""
        payload = b"a" * (70 * 1024) + b"\x00" * 8
        buf = memoryview(payload)
        with pytest.raises(DecodeError, match="(?i)exceeds maximum"):
            decode_text(buf, max_size=4 * 1024)

    def test_chunked_null_just_past_cap_raises_exceeds_maximum(self) -> None:
        """A NUL just past the cap surfaces as cap-exceeded, not accepted."""
        # NUL at byte 4097, exactly past max_size=4096.
        payload = b"a" * 4097 + b"\x00" + b"b" * (70 * 1024) + b"\x00" * 8
        buf = memoryview(payload)
        with pytest.raises(DecodeError, match="(?i)exceeds maximum"):
            decode_text(buf, max_size=4096)

    def test_chunked_missing_null_terminator_raises(self) -> None:
        # 70 KiB > 64 KiB threshold, forcing the chunked branch.
        payload = b"a" * (70 * 1024)
        buf = memoryview(payload + b"\x00" * _align(len(payload)))
        with pytest.raises(DecodeError, match="not null-terminated"):
            decode_text(buf)

    def test_chunked_invalid_utf8_before_null_raises(self) -> None:
        # 70 KiB of ASCII + a truncated multi-byte UTF-8 sequence, then null.
        prefix = b"a" * (70 * 1024)
        payload = prefix + b"\xc3" + b"\x00"
        buf = memoryview(payload + b"\x00" * _align(len(payload)))
        with pytest.raises(DecodeError, match="Invalid UTF-8"):
            decode_text(buf)

    def test_chunked_multibyte_codepoint_across_boundary(self) -> None:
        """A 3-byte codepoint straddling the ``_TEXT_SCAN_CHUNK`` boundary
        decodes correctly (the accumulator joins chunks before decoding)."""
        # Place the codepoint one byte before the chunk boundary; total must
        # exceed _TEXT_ONE_SHOT_MAX to force the chunked branch.
        from dqlitewire.types import _TEXT_ONE_SHOT_MAX, _TEXT_SCAN_CHUNK

        filler_len = _TEXT_SCAN_CHUNK - 1
        codepoint = "€".encode()  # 3 bytes: e2 82 ac
        assert len(codepoint) == 3
        tail_padding = b"z" * (_TEXT_ONE_SHOT_MAX + 1 - filler_len - len(codepoint))
        payload = b"a" * filler_len + codepoint + tail_padding + b"\x00"
        buf = memoryview(payload + b"\x00" * _align(len(payload)))

        text, _consumed = decode_text(buf)
        assert text == "a" * filler_len + "€" + "z" * len(tail_padding)


# ---- merged from test_decode_text_errors_kwarg.py ----
# ``decode_text`` accepts an ``errors=`` kwarg threaded to ``bytes.decode``.
# The ``"strict"`` default is a Python-side narrowing (C and Go clients pass
# non-UTF-8 bytes through unvalidated).


def test_strict_default_rejects_invalid_utf8() -> None:
    # 0xff is invalid UTF-8.
    data = b"\xff" + b"\x00" + b"\x00" * 6
    with pytest.raises(DecodeError, match="Invalid UTF-8"):
        decode_text(data)


def test_replace_kwarg_coerces_invalid_utf8() -> None:
    data = b"\xff" + b"\x00" + b"\x00" * 6
    text, _ = decode_text(data, errors="replace")
    assert text == "�"


def test_backslashreplace_kwarg_coerces_invalid_utf8() -> None:
    data = b"\xff" + b"\x00" + b"\x00" * 6
    text, _ = decode_text(data, errors="backslashreplace")
    assert text == "\\xff"


def test_errors_kwarg_works_through_memoryview_branch() -> None:
    """The chunked / one-shot memoryview path also honours ``errors``."""
    data = memoryview(b"\xff" + b"\x00" + b"\x00" * 6)
    text, _ = decode_text(data, errors="replace")
    assert text == "�"


def test_strict_explicit_is_default_behaviour() -> None:
    """Passing ``errors='strict'`` explicitly matches the default."""
    data = b"\xff" + b"\x00" + b"\x00" * 6
    with pytest.raises(DecodeError, match="Invalid UTF-8"):
        decode_text(data, errors="strict")


# ---- merged from test_decode_text_one_shot_probe_for_small_cells_in_long_buffer.py ----
# ``decode_text`` uses the one-shot path for small TEXT cells even when the
# remaining buffer is large. It probes the first
# ``min(data_len, max_size+1, _TEXT_ONE_SHOT_MAX)`` bytes; only cells exceeding
# that probe window escalate to the chunked path. Peak transient stays <= 64 KiB.


def _build_body(cells: list[str]) -> bytes:
    """Build a TEXT-cells body matching the encoder shape: utf-8 + NUL +
    pad-to-word per cell."""

    def pad(n: int) -> int:
        return (-n) & 7

    out = bytearray()
    for cell in cells:
        utf8 = cell.encode("utf-8")
        encoded = utf8 + b"\x00"
        out.extend(encoded)
        out.extend(b"\x00" * pad(len(encoded)))
    return bytes(out)


def test_short_cells_in_long_buffer_use_one_shot_path() -> None:
    """Every short cell in a long buffer round-trips via the one-shot path
    (the next test directly asserts the chunked path never fires)."""
    # 4000 short cells (~64 KB total) spans the prior data_len > 64 KiB regime.
    cells = [f"cell-{i:08d}" for i in range(4000)]
    body = _build_body(cells)

    view = memoryview(body)
    offset = 0
    decoded: list[str] = []
    while offset < len(view):
        text, consumed = decode_text(view[offset:])
        decoded.append(text)
        offset += consumed

    assert decoded == cells, "all cells must round-trip"


def test_chunked_path_only_fires_for_cells_exceeding_probe_window() -> None:
    """The chunked path must NOT fire for short cells in a long buffer."""
    cells = [f"x-{i}" for i in range(1000)]
    body = _build_body(cells)
    # Tail-pad with 1 MiB of NULs so data_len > 64 KiB for the first cells.
    padded = body + b"\x00" * (1 << 20)
    view = memoryview(padded)

    # Detect chunked-path entry by swapping _TEXT_SCAN_CHUNK for a sentinel
    # that counts the index/add ops the chunked path performs on it.
    chunked_path_entered = 0
    import dqlitewire.types as types_mod

    real_scan_chunk = types_mod._TEXT_SCAN_CHUNK

    class _ChunkedSentinel:
        def __index__(self) -> int:
            nonlocal chunked_path_entered
            chunked_path_entered += 1
            return real_scan_chunk

        def __add__(self, other: int) -> int:
            nonlocal chunked_path_entered
            chunked_path_entered += 1
            return real_scan_chunk + other

        def __radd__(self, other: int) -> int:
            nonlocal chunked_path_entered
            chunked_path_entered += 1
            return other + real_scan_chunk

    with mock.patch.object(types_mod, "_TEXT_SCAN_CHUNK", _ChunkedSentinel()):
        offset = 0
        decoded: list[str] = []
        for _ in range(len(cells)):
            text, consumed = decode_text(view[offset:])
            decoded.append(text)
            offset += consumed

    assert decoded == cells, "all short cells must decode correctly"
    assert chunked_path_entered == 0, (
        f"chunked path entered {chunked_path_entered} times for "
        "short cells in a long buffer; expected zero (one-shot probe "
        "should cover every cell)"
    )


def test_chunked_path_still_fires_for_genuinely_long_cells() -> None:
    """Regression: a cell that exceeds the one-shot probe window
    must still decode correctly via the chunked path."""
    from dqlitewire.types import _TEXT_ONE_SHOT_MAX

    long_text = "x" * (_TEXT_ONE_SHOT_MAX + 100)
    body = _build_body([long_text])

    text, consumed = decode_text(memoryview(body))
    assert text == long_text
    assert consumed == len(body)


# ---- merged from test_decode_text_post_cap_bytes_branch.py ----
# decode_text bytes-branch post-cap arms: not-null-terminated and length-exceeds-maximum.


def test_bytes_branch_post_cap_check_rejects() -> None:
    data = b"X" * 11 + b"\x00"
    with pytest.raises(DecodeError, match="exceeds maximum"):
        decode_text(data, max_size=10)


def test_bytes_branch_at_cap_accepts() -> None:
    data = b"X" * 10 + b"\x00" + b"\x00" * 5  # padded to word
    text, _ = decode_text(data, max_size=10)
    assert text == "X" * 10


# ---- merged from test_decode_text_short_probe_avoids_64kib_materialise.py ----
# decode_text peeks a short probe first; only long cells pay the 64 KiB materialise.


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


# ---- merged from test_decode_text_surrogateescape_round_trip_caveat.py ----
# decode_text(errors="surrogateescape") is not round-trippable through strict encode_text.


def test_surrogateescape_decoded_string_fails_encode() -> None:
    raw = b"caf\xe9\x00\x00\x00\x00\x00"
    s, _ = decode_text(raw, errors="surrogateescape")
    assert "\udce9" in s
    with pytest.raises(EncodeError, match="surrogate"):
        encode_text(s)


# ---- merged from test_decode_value_text_label_symmetry.py ----
# decode_value's TEXT/ISO8601 branch forwards ValueType.name as the decode_text label.


def test_decode_value_iso8601_truncated_names_iso8601() -> None:
    """No-NUL buffer must surface "ISO8601" in the DecodeError, not the generic "Text" prefix."""
    truncated = b"20240101"
    with pytest.raises(DecodeError, match="ISO8601"):
        decode_value(truncated, ValueType.ISO8601)


def test_decode_value_text_truncated_names_text() -> None:
    """The TEXT branch must surface "TEXT" (the ValueType name) in the DecodeError."""
    truncated = b"abcdefgh"
    with pytest.raises(DecodeError, match="TEXT"):
        decode_value(truncated, ValueType.TEXT)


# ---- merged from test_encode_value_text_row_cell_cap.py ----
# encode_value TEXT/ISO8601 must enforce the same MAX_TEXT_VALUE_SIZE cap
# as decode_text, else a same-process encode then decode ships bytes the
# decoder rejects.


def test_encode_value_text_at_cap_round_trips() -> None:
    value = "x" * MAX_TEXT_VALUE_SIZE
    encoded, _ = encode_value(value, ValueType.TEXT)
    decoded, _ = decode_value(encoded, ValueType.TEXT)
    assert decoded == value


def test_encode_value_text_one_over_cap_rejected_at_encode() -> None:
    value = "x" * (MAX_TEXT_VALUE_SIZE + 1)
    with pytest.raises(EncodeError):
        encode_value(value, ValueType.TEXT)


def test_encode_value_iso8601_at_cap_round_trips() -> None:
    value = "0" * MAX_TEXT_VALUE_SIZE
    encoded, _ = encode_value(value, ValueType.ISO8601)
    decoded, _ = decode_value(encoded, ValueType.ISO8601)
    assert decoded == value


def test_encode_value_iso8601_one_over_cap_rejected() -> None:
    value = "0" * (MAX_TEXT_VALUE_SIZE + 1)
    with pytest.raises(EncodeError):
        encode_value(value, ValueType.ISO8601)


def test_encode_value_text_error_message_names_the_value_type() -> None:
    value = "x" * (MAX_TEXT_VALUE_SIZE + 1)
    with pytest.raises(EncodeError, match="TEXT"):
        encode_value(value, ValueType.TEXT)
    with pytest.raises(EncodeError, match="ISO8601"):
        encode_value(value, ValueType.ISO8601)


# ---- merged from test_text_size_cap_byte_unit.py ----
# encode_text and decode_text cap on UTF-8 BYTES, not codepoints, so
# round-trip identity holds for content where byte length != codepoint length.


class TestEncodeTextMaxSizeBytes:
    def test_max_size_caps_on_bytes_not_codepoints(self) -> None:
        # 😀 is 4 bytes; 32 codepoints = 128 bytes. cap=130 admits, cap=120 rejects.
        text = "😀" * 32
        encode_text(text, max_size=130)
        with pytest.raises(EncodeError, match="length 128 exceeds maximum"):
            encode_text(text, max_size=120)

    def test_default_no_cap(self) -> None:
        # Without max_size, there is no per-call cap.
        encode_text("a" * 100_000)

    def test_label_in_error(self) -> None:
        with pytest.raises(EncodeError, match="leader address length"):
            encode_text("a" * 16, max_size=8, label="leader address")


class TestRoundTripIdentityAtCap:
    def test_failure_message_round_trip_at_byte_cap(self) -> None:
        text = "😀" * (MAX_FAILURE_MESSAGE_SIZE // 4)  # 4 bytes each
        body = FailureResponse(code=1, message=text).encode_body()
        decoded = FailureResponse.decode_body(body)
        assert decoded.message == text

    def test_failure_message_rejected_above_byte_cap(self) -> None:
        text = "😀" * ((MAX_FAILURE_MESSAGE_SIZE // 4) + 1)
        with pytest.raises(EncodeError, match="(?i)failure message"):
            FailureResponse(code=1, message=text).encode_body()

    def test_leader_address_round_trip_at_byte_cap(self) -> None:
        text = "ä" * (MAX_ADDRESS_SIZE // 2)  # 2-byte UTF-8 codepoint
        body = LeaderResponse(node_id=1, address=text).encode_body()
        decoded = LeaderResponse.decode_body(body)
        assert decoded.address == text


class TestDecodeTextCapBeforeAllocate:
    def test_cap_bound_materialisation_window(self) -> None:
        # NUL well past max_size: decoder must cap-error without materialising
        # the whole buffer (we can only observe the error shape, not allocation).
        too_long = b"a" * 100 + b"\x00"
        view = memoryview(too_long)
        with pytest.raises(Exception, match="exceeds maximum"):
            decode_text(view, max_size=10)
