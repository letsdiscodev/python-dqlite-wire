"""Tests for BLOB encoding/decoding: bytes-like inputs, memoryview formats, mmap, caps."""

from __future__ import annotations

import array
import mmap
from typing import Any
from unittest.mock import patch

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.limits import MAX_BLOB_SIZE
from dqlitewire.types import (
    _reject_non_byte_format_memoryview,
    decode_blob,
    decode_value,
    encode_blob,
    encode_value,
)

# ---- merged from test_blob_cap_probe_uses_nbytes.py ----
# encode_value's BLOB cap-before-materialise probe must use memoryview.nbytes
# (byte count), not len(memoryview) (element count for multi-byte formats), so an
# oversize multi-byte view is rejected before the materialise allocation
# regardless of _reject_non_byte_format_memoryview's ordering.


def test_blob_cap_probe_rejects_oversize_before_materialise() -> None:
    """A multi-byte memoryview whose element count fits under the cap but
    whose nbytes exceeds it must be rejected by the outer probe before
    bytes(value) materialises 8x the bytes."""
    import array as _array

    element_count = MAX_BLOB_SIZE // 8 + 1
    mv = memoryview(_array.array("Q", [0] * element_count))
    assert mv.itemsize == 8
    assert len(mv) == element_count
    assert mv.nbytes == element_count * 8
    assert mv.nbytes > MAX_BLOB_SIZE
    assert len(mv) < MAX_BLOB_SIZE  # len-based probe would pass; nbytes-based catches it

    # Patch the format check to no-op (simulate a refactor that drops it) and
    # make encode_blob fail if reached — the outer probe must fire first.
    def _should_not_be_reached(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Outer cap probe failed to reject before materialise — "
            "encode_blob was reached, indicating the probe used "
            "len(mv) (element count) instead of mv.nbytes (byte count)."
        )

    with (
        patch("dqlitewire.types._reject_non_byte_format_memoryview"),
        patch("dqlitewire.types.encode_blob", side_effect=_should_not_be_reached),
        pytest.raises(EncodeError, match=r"Blob length \d+ exceeds maximum"),
    ):
        encode_value(mv, ValueType.BLOB)


def test_blob_cap_probe_reports_byte_count_not_element_count() -> None:
    """The EncodeError message must name the byte count (mv.nbytes), not the
    element count (len(mv)), to match the cap's units."""
    import array as _array

    element_count = MAX_BLOB_SIZE // 8 + 1
    mv = memoryview(_array.array("Q", [0] * element_count))
    expected_nbytes = element_count * 8

    def _stub(*args: object, **kwargs: object) -> bytes:
        return b""

    with (
        patch("dqlitewire.types._reject_non_byte_format_memoryview"),
        patch("dqlitewire.types.encode_blob", side_effect=_stub),
        pytest.raises(EncodeError) as exc_info,
    ):
        encode_value(mv, ValueType.BLOB)

    assert str(expected_nbytes) in str(exc_info.value)
    assert f"Blob length {len(mv)} " not in str(exc_info.value)


def test_blob_cap_probe_bytes_still_uses_len() -> None:
    """For bytes/bytearray/mmap (no .nbytes), the probe falls back to
    len(value), which is byte-correct by type."""
    payload = b"\x00" * (MAX_BLOB_SIZE + 1)
    with pytest.raises(EncodeError, match=r"Blob length \d+ exceeds maximum"):
        encode_value(payload, ValueType.BLOB)


# ---- merged from test_blob_closed_mmap_encode_error.py ----
# A closed mmap or released memoryview on the BLOB encode path must surface as
# EncodeError, not the raw ValueError/BufferError that bytes() raises, so callers
# catching EncodeError don't miss the failure.


def test_encode_value_explicit_blob_closed_mmap_raises_encode_error() -> None:
    payload = b"hello"
    mm = mmap.mmap(-1, len(payload))
    mm.write(payload)
    mm.close()
    with pytest.raises(EncodeError):
        encode_value(mm, ValueType.BLOB)


def test_encode_value_inferred_blob_closed_mmap_raises_encode_error() -> None:
    payload = b"hello"
    mm = mmap.mmap(-1, len(payload))
    mm.write(payload)
    mm.close()
    with pytest.raises(EncodeError):
        encode_value(mm)


def test_encode_value_explicit_blob_released_memoryview_raises_encode_error() -> None:
    """A released memoryview raises BufferError from bytes(view) — same wrap."""
    buf = bytearray(b"hello")
    view = memoryview(buf)
    view.release()
    with pytest.raises(EncodeError):
        encode_value(view, ValueType.BLOB)


def test_encode_value_explicit_blob_preserves_original_exception_via_cause() -> None:
    """The wrap must preserve the underlying error via __cause__."""
    payload = b"hello"
    mm = mmap.mmap(-1, len(payload))
    mm.write(payload)
    mm.close()
    try:
        encode_value(mm, ValueType.BLOB)
    except EncodeError as e:
        assert isinstance(e.__cause__, (ValueError, BufferError))
    else:  # pragma: no cover
        pytest.fail("expected EncodeError")


# ---- merged from test_blob_docstring_cites_actual_cap.py ----
# encode_blob/decode_blob docstrings must cite the actual MAX_BLOB_SIZE (or a
# "minus framing" qualifier), not the rounded "64 MiB" figure: the constant sits
# 64 bytes below DEFAULT_MAX_MESSAGE_SIZE, so bare "64 MiB" traps a caller into a
# 64-byte EncodeError shortfall.


def test_blob_at_documented_cap_actually_encodes() -> None:
    """The cited cap must actually round-trip (guards constant/docstring drift)."""
    encoded = encode_blob(b"\x00" * MAX_BLOB_SIZE)
    assert isinstance(encoded, bytes)
    decoded, _ = decode_blob(encoded)
    assert len(decoded) == MAX_BLOB_SIZE


# ---- merged from test_blob_inference_mmap.py ----
# ``mmap.mmap`` infers to ``ValueType.BLOB`` (parity with stdlib sqlite3).


def test_encode_value_infers_blob_from_mmap() -> None:
    payload = b"hello world!"
    mm = mmap.mmap(-1, len(payload))
    try:
        mm.write(payload)
        mm.seek(0)
        encoded, vtype = encode_value(mm)
        assert vtype == ValueType.BLOB
        decoded, _ = decode_value(encoded, ValueType.BLOB)
        assert decoded == payload
    finally:
        mm.close()


def test_encode_value_blob_explicit_accepts_mmap() -> None:
    payload = b"explicit blob payload"
    mm = mmap.mmap(-1, len(payload))
    try:
        mm.write(payload)
        mm.seek(0)
        encoded, vtype = encode_value(mm, ValueType.BLOB)
        assert vtype == ValueType.BLOB
        decoded, _ = decode_value(encoded, ValueType.BLOB)
        assert decoded == payload
    finally:
        mm.close()


def test_encode_value_blob_mmap_decodes_to_bytes() -> None:
    """The decoder yields ``bytes``, not a view over ``mmap.mmap``."""
    raw = b"mmap-payload-content"
    mm = mmap.mmap(-1, len(raw))
    try:
        mm.write(raw)
        mm.seek(0)
        encoded, _ = encode_value(mm, ValueType.BLOB)
        decoded, _ = decode_value(encoded, ValueType.BLOB)
        assert isinstance(decoded, bytes)
        assert decoded == raw
    finally:
        mm.close()


# ---- merged from test_blob_max_size_kwarg.py ----
# ``encode_blob``/``decode_blob`` accept a ``max_blob_size`` kwarg overriding
# the 16 MiB default.


def test_default_max_blob_size_unchanged() -> None:
    oversize = b"\x00" * (MAX_BLOB_SIZE + 1)
    with pytest.raises(EncodeError, match="exceeds maximum"):
        encode_blob(oversize)


def test_caller_can_lower_encode_cap() -> None:
    with pytest.raises(EncodeError, match="exceeds maximum"):
        encode_blob(b"\x00" * 100, max_blob_size=99)


def test_caller_can_raise_encode_cap() -> None:
    big = b"\x00" * (MAX_BLOB_SIZE + 16)
    encoded = encode_blob(big, max_blob_size=MAX_BLOB_SIZE * 2)
    decoded, _ = decode_blob(encoded, max_blob_size=MAX_BLOB_SIZE * 2)
    assert decoded == big


def test_caller_can_lower_decode_cap() -> None:
    payload = encode_blob(b"\x00" * 100)
    with pytest.raises(DecodeError, match="exceeds maximum"):
        decode_blob(payload, max_blob_size=50)


# ---- merged from test_blob_memoryview_non_byte_format_rejected.py ----
# A non-byte-format ``memoryview`` (e.g. over ``array.array('i', ...)``)
# is rejected: its bytes are host-endian/host-width, the same corruption
# hazard that bars bare ``array.array``. Byte-format views still work.


def test_encode_value_infer_rejects_non_byte_format_memoryview() -> None:
    mv = memoryview(array.array("i", [1, 2, 3]))
    assert mv.format == "i" and mv.itemsize == 4
    with pytest.raises(EncodeError, match="format"):
        encode_value(mv)


def test_encode_value_blob_explicit_rejects_non_byte_format_memoryview() -> None:
    mv = memoryview(array.array("d", [1.5, 2.5, 3.5]))
    assert mv.format == "d" and mv.itemsize == 8
    with pytest.raises(EncodeError, match="format"):
        encode_value(mv, ValueType.BLOB)


def test_encode_value_infer_rejects_non_byte_format_memoryview_int64() -> None:
    mv = memoryview(array.array("q", [1, 2, 3]))
    assert mv.format in ("q", "l") and mv.itemsize == 8
    with pytest.raises(EncodeError, match="format"):
        encode_value(mv)


def test_encode_value_accepts_byte_format_memoryview_infer() -> None:
    mv = memoryview(b"\x01\x02\x03")
    assert mv.format == "B"
    encoded, vtype = encode_value(mv)
    assert vtype == ValueType.BLOB
    decoded, _ = decode_value(encoded, ValueType.BLOB)
    assert decoded == b"\x01\x02\x03"


def test_encode_value_accepts_byte_format_memoryview_explicit_blob() -> None:
    mv = memoryview(bytearray(b"explicit"))
    assert mv.format == "B"
    encoded, vtype = encode_value(mv, ValueType.BLOB)
    assert vtype == ValueType.BLOB
    decoded, _ = decode_value(encoded, ValueType.BLOB)
    assert decoded == b"explicit"


def test_encode_value_accepts_signed_byte_format_memoryview() -> None:
    """``'b'`` (signed char) is single-byte and must work; rejection targets multi-byte formats."""
    mv = memoryview(array.array("b", [1, -2, 3]))
    assert mv.format == "b" and mv.itemsize == 1
    encoded, vtype = encode_value(mv)
    assert vtype == ValueType.BLOB
    decoded, _ = decode_value(encoded, ValueType.BLOB)
    assert decoded == bytes(mv)


# ---- merged from test_decode_blob_round_trip_type_asymmetry.py ----
# ``decode_blob`` always returns ``bytes`` regardless of bind input type
# (bytes/bytearray/memoryview), matching stdlib ``sqlite3``. If the contract
# is ever tightened to re-wrap as ``memoryview``, delete this pin and update
# the ``decode_blob`` docstring.


def test_decode_blob_returns_bytes_for_bytes_input() -> None:
    encoded = encode_blob(b"hello")
    decoded, _ = decode_blob(encoded)
    assert type(decoded) is bytes
    assert decoded == b"hello"


def test_decode_blob_returns_bytes_for_bytearray_input() -> None:
    """A ``bytearray`` bind round-trips to ``bytes``, NOT ``bytearray``."""
    encoded, vt = encode_value(bytearray(b"hello"))
    assert vt == ValueType.BLOB
    decoded, _ = decode_value(encoded, vt)
    assert type(decoded) is bytes
    assert decoded == b"hello"


def test_decode_blob_returns_bytes_for_memoryview_input() -> None:
    """A ``memoryview`` bind round-trips to ``bytes``, NOT ``memoryview``."""
    encoded, vt = encode_value(memoryview(b"hello"))
    assert vt == ValueType.BLOB
    decoded, _ = decode_value(encoded, vt)
    assert type(decoded) is bytes
    assert decoded == b"hello"


def test_decode_blob_returns_bytes_when_input_is_memoryview_buffer() -> None:
    """The ``data`` argument shape (bytes vs memoryview) does not affect the
    return type either — always bytes."""
    encoded = encode_blob(b"world")
    via_bytes, _ = decode_blob(encoded)
    via_memoryview, _ = decode_blob(memoryview(encoded))
    assert type(via_bytes) is bytes
    assert type(via_memoryview) is bytes
    assert via_bytes == via_memoryview == b"world"


# ---- merged from test_encode_blob_accepts_bytes_like.py ----
# ``encode_blob`` accepts any bytes-like input (``bytes``, ``bytearray``,
# ``memoryview``), matching the ``WireInput`` contract; direct external callers
# previously hit a runtime rejection of ``memoryview``.


@pytest.mark.parametrize(
    "value",
    [
        b"hello",
        bytearray(b"hello"),
        memoryview(b"hello"),
        memoryview(bytearray(b"hello")),
    ],
    ids=["bytes", "bytearray", "memoryview-of-bytes", "memoryview-of-bytearray"],
)
def test_encode_blob_accepts_bytes_like_inputs(value: object) -> None:
    encoded = encode_blob(value)  # type: ignore[arg-type]
    expected = encode_blob(b"hello")
    assert encoded == expected


def test_encode_blob_rejects_non_bytes_like() -> None:
    # str is the canonical footgun a caller might pass by accident.
    with pytest.raises(EncodeError, match="Blob value must be bytes"):
        encode_blob("hello")  # type: ignore[arg-type]


def test_encode_blob_rejects_object_type() -> None:
    with pytest.raises(EncodeError, match="Blob value must be bytes"):
        encode_blob(object())  # type: ignore[arg-type]


# ---- merged from test_encode_value_blob_len_probe.py ----
# ``encode_value`` for BLOB checks ``len(value)`` against the cap before
# materialising via ``bytes(value)``, so a hostile-large memoryview/mmap is
# rejected without first copying its whole contents.


def test_encode_value_blob_len_probe_rejects_via_memoryview() -> None:
    """A memoryview just over the cap is rejected on the ``len()`` probe before
    ``bytes(value)`` runs."""
    # Tiny patched cap exercises the same path as the production cap cheaply.
    with patch("dqlitewire.types.MAX_BLOB_SIZE", 8):
        buf = memoryview(bytearray(16))
        with pytest.raises(EncodeError, match="exceeds maximum"):
            encode_value(buf, ValueType.BLOB)


def test_encode_value_blob_len_probe_rejects_via_bytearray() -> None:
    """``bytearray`` exercises the same cap-before-materialise branch."""
    with patch("dqlitewire.types.MAX_BLOB_SIZE", 8):
        buf = bytearray(16)
        with pytest.raises(EncodeError, match="exceeds maximum"):
            encode_value(buf, ValueType.BLOB)


def test_encode_value_blob_at_cap_accepts() -> None:
    """A buffer at exactly the cap encodes fine."""
    with patch("dqlitewire.types.MAX_BLOB_SIZE", 16):
        buf = bytearray(16)
        encoded, vt = encode_value(buf, ValueType.BLOB)
        assert vt == ValueType.BLOB
        assert isinstance(encoded, bytes)


def test_encode_value_blob_materialise_failure_wrapped() -> None:
    """If ``len()`` passes but ``bytes(value)`` raises (released memoryview /
    closed mmap), the wrap surfaces ``EncodeError`` not bare ValueError."""
    buf = bytearray(16)
    view = memoryview(buf)
    view.release()
    with pytest.raises(EncodeError, match="Cannot materialise BLOB"):
        encode_value(view, ValueType.BLOB)


# ---- merged from test_encode_value_format_check_fires_once.py ----
# _reject_non_byte_format_memoryview must fire exactly once per
# encode_value call (it was previously run in both _infer_value_type and the
# explicit BLOB arm, doubling the per-cell cost).


def test_format_check_fires_once_on_inferred_blob_path() -> None:
    mv = memoryview(b"\x01\x02\x03")
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value(mv)
    assert mock_reject.call_count == 1


def test_format_check_fires_once_on_explicit_blob_path() -> None:
    mv = memoryview(b"\x01\x02\x03")
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value(mv, ValueType.BLOB)
    assert mock_reject.call_count == 1


def test_format_check_does_not_fire_for_non_memoryview_blob() -> None:
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value(b"abc")
        encode_value(b"abc", ValueType.BLOB)
    assert mock_reject.call_count == 0


def test_format_check_does_not_fire_for_non_blob_paths() -> None:
    with patch("dqlitewire.types._reject_non_byte_format_memoryview") as mock_reject:
        encode_value("hello")
        encode_value(42)
        encode_value(1.5)
        encode_value(True)
        encode_value(None)
    assert mock_reject.call_count == 0


# ---- merged from test_reject_non_byte_format_memoryview_released_buffer.py ----
# Pin: ``_reject_non_byte_format_memoryview`` falls through silently
# when ``.format`` raises EITHER ``ValueError`` or ``BufferError``, so the
# helper does not hinge on which exception a given CPython raises for a
# released buffer (downstream ``bytes(value)`` then surfaces EncodeError).


def test_released_memoryview_falls_through_silently() -> None:
    """A released ``memoryview`` (non-byte format) must fall through
    silently so the downstream materialise step surfaces EncodeError."""
    mv = memoryview(array.array("i", [1, 2, 3]))
    mv.release()
    _reject_non_byte_format_memoryview(mv)  # no exception


def test_format_raising_buffer_error_falls_through_silently() -> None:
    """``.format`` raising ``BufferError`` must still fall through silently."""

    class _FakeMv:
        @property
        def format(self) -> Any:
            raise BufferError("simulated future-CPython narrowing")

        @property
        def itemsize(self) -> int:
            return 1

    # Runtime check is duck-typed on .format / .itemsize, so the fake stands in.
    _reject_non_byte_format_memoryview(_FakeMv())  # type: ignore[arg-type]


def test_format_raising_value_error_falls_through_silently() -> None:
    """``.format`` raising ``ValueError`` also falls through silently."""

    class _FakeMv:
        @property
        def format(self) -> Any:
            raise ValueError("operation forbidden on released memoryview object")

        @property
        def itemsize(self) -> int:
            return 1

    _reject_non_byte_format_memoryview(_FakeMv())  # type: ignore[arg-type]
