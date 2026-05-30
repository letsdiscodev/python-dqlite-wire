"""Primitive type encoding/decoding for dqlite wire protocol.

Integers are little-endian; text is NUL-terminated UTF-8, padded to 8 bytes.
Deals only in wire primitives; datetime conversions belong in the driver layer.
"""

import datetime
import mmap
import struct
from collections.abc import Callable
from typing import Final, cast

from dqlitewire.constants import WORD_SIZE, ValueType
from dqlitewire.exceptions import DecodeError, EncodeError

__all__ = [
    "WireInput",
    "WireValue",
    "decode_blob",
    "decode_double",
    "decode_int64",
    "decode_text",
    "decode_uint32",
    "decode_uint64",
    "decode_value",
    "encode_blob",
    "encode_double",
    "encode_int64",
    "encode_text",
    "encode_uint32",
    "encode_uint64",
    "encode_value",
    "pad_to_word",
]

type WireInput = bool | int | float | str | bytes | bytearray | memoryview | mmap.mmap | None

type WireValue = bool | int | float | str | bytes | None

# Sits 64 bytes below DEFAULT_MAX_MESSAGE_SIZE so a blob at the cap round-trips
# through the default decoder, leaving room for the length prefix + row framing.
_MAX_BLOB_SIZE: Final[int] = 64 * 1024 * 1024 - 64  # 64 MiB minus framing overhead

# Symmetric with _MAX_BLOB_SIZE; keeps decode_text from allocating
# attacker-controlled lengths that exceed the default frame envelope.
_MAX_TEXT_VALUE_SIZE: Final[int] = 64 * 1024 * 1024 - 64  # 64 MiB minus framing overhead

# Cap on stringified ints in EncodeError messages so 10**500 doesn't bake a
# kilobyte of digits into the error text (and every log line that quotes it).
_MAX_VALUE_REPR: Final[int] = 64

# NULL is 8 bytes on the wire (a uint64 read into a never-inspected union member);
# upstream has a TODO to allow 0-byte NULLs. Centralised so encode/decode flip in lockstep.
NULL_CELL_WIDTH: Final[int] = 8


def _bounded_repr(value: int) -> str:
    s = str(value)
    if len(s) <= _MAX_VALUE_REPR:
        return s
    return f"{s[:_MAX_VALUE_REPR]}... [{len(s)} digits]"


def _bounded_repr_any(value: object) -> str:
    """``repr(value)`` capped at ``_MAX_VALUE_REPR`` chars (for arbitrary objects)."""
    s = repr(value)
    if len(s) <= _MAX_VALUE_REPR:
        return s
    return f"{s[:_MAX_VALUE_REPR]}... [{len(s)} chars]"


# Single-byte buffer-protocol formats accepted as zero-copy BLOB binds. Other
# formats are typed numeric memoryviews carrying host-endian, non-portable bytes.
_BYTE_FORMAT_MEMORYVIEW: Final[frozenset[str]] = frozenset({"B", "b", "c"})


def _reject_non_byte_format_memoryview(value: memoryview) -> None:
    """Reject a memoryview whose buffer-protocol format is not single-byte."""
    try:
        fmt = value.format
    # Released memoryview raises ValueError (or BufferError on future CPython);
    # fall through so bytes(value) surfaces the canonical EncodeError.
    except (ValueError, BufferError):
        return
    if fmt not in _BYTE_FORMAT_MEMORYVIEW:
        raise EncodeError(
            f"memoryview must have single-byte format ('B'/'b'/'c') to "
            f"encode as BLOB; got format={fmt!r} "
            f"itemsize={value.itemsize}. Typed numeric memoryviews "
            f"(e.g. memoryview(array.array('i', ...))) carry host-"
            f"endian, host-int-width bytes and are non-portable — "
            f"identical hazard to the bare array.array rejection. "
            f"Materialise to bytes via a portable layout (e.g. "
            f"struct.pack with '<' little-endian) before binding."
        )


def _is_int_not_bool(value: object) -> bool:
    """True if ``value`` is an int but not a bool (bool is an int subclass)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_uint64(name: str, value: int) -> None:
    """Validate ``value`` is an int (not bool) in the uint64 range."""
    if not _is_int_not_bool(value):
        raise EncodeError(f"{name} must be int, got {type(value).__name__}")
    if not 0 <= value < 2**64:
        raise EncodeError(
            f"{name}={_bounded_repr(value)} out of range for uint64 (0 to {(1 << 64) - 1})"
        )


def _validate_uint32(name: str, value: int) -> None:
    """Validate ``value`` is an int (not bool) in the uint32 range."""
    if not _is_int_not_bool(value):
        raise EncodeError(f"{name} must be int, got {type(value).__name__}")
    if not 0 <= value < 2**32:
        raise EncodeError(
            f"{name}={_bounded_repr(value)} out of range for uint32 (0 to {(1 << 32) - 1})"
        )


def _validate_int64(name: str, value: int) -> None:
    """Validate ``value`` is an int (not bool) in the signed int64 range."""
    if not _is_int_not_bool(value):
        raise EncodeError(f"{name} must be int, got {type(value).__name__}")
    if not -(2**63) <= value < 2**63:
        raise EncodeError(f"Value {_bounded_repr(value)} out of range for int64")


def encode_uint64(value: int) -> bytes:
    """Encode an unsigned 64-bit integer (little-endian)."""
    _validate_uint64("value", value)
    return struct.pack("<Q", value)


def decode_uint64(data: bytes | memoryview, *, label: str = "uint64") -> int:
    """Decode an unsigned 64-bit integer (little-endian); ``label`` names the field in errors."""
    if len(data) < 8:
        raise DecodeError(f"Need 8 bytes for {label}, got {len(data)}")
    result: int = struct.unpack("<Q", data[:8])[0]
    return result


def encode_int64(value: int) -> bytes:
    """Encode a signed 64-bit integer (little-endian); rejects bool to surface caller typos."""
    _validate_int64("value", value)
    return struct.pack("<q", value)


def decode_int64(data: bytes | memoryview, *, label: str = "int64") -> int:
    """Decode a signed 64-bit integer (little-endian); ``label`` names the field in errors."""
    if len(data) < 8:
        raise DecodeError(f"Need 8 bytes for {label}, got {len(data)}")
    result: int = struct.unpack("<q", data[:8])[0]
    return result


def encode_uint32(value: int) -> bytes:
    """Encode an unsigned 32-bit integer (little-endian)."""
    _validate_uint32("value", value)
    return struct.pack("<I", value)


def decode_uint32(data: bytes | memoryview) -> int:
    """Decode an unsigned 32-bit integer (little-endian)."""
    if len(data) < 4:
        raise DecodeError(f"Need 4 bytes for uint32, got {len(data)}")
    result: int = struct.unpack("<I", data[:4])[0]
    return result


def encode_double(value: float) -> bytes:
    """Encode a 64-bit float (little-endian); accepts all IEEE 754 incl. NaN/inf.

    Rejects int/bool/numeric proxies: their implicit coercion silently loses bits
    (|x| >= 2**53) or overflows. Callers wanting these must float(x) explicitly.
    """
    if isinstance(value, bool):
        raise EncodeError(f"encode_double rejects bool, got {value!r}")
    if not isinstance(value, float):
        raise EncodeError(
            f"encode_double requires float, got {type(value).__name__}; "
            "cast with float(x) explicitly if int / numeric-proxy "
            "semantics are intended."
        )
    return struct.pack("<d", value)


def decode_double(data: bytes | memoryview, *, label: str = "double") -> float:
    """Decode a 64-bit float (little-endian); ``label`` names the field in errors."""
    if len(data) < 8:
        raise DecodeError(f"Need 8 bytes for {label}, got {len(data)}")
    result: float = struct.unpack("<d", data[:8])[0]
    return result


def pad_to_word(size: int) -> int:
    """Calculate padding needed to align to word boundary."""
    remainder = size % WORD_SIZE
    if remainder == 0:
        return 0
    return WORD_SIZE - remainder


def encode_text(value: str, *, max_size: int | None = None, label: str = "Text") -> bytes:
    """Encode text as NUL-terminated UTF-8, padded to 8-byte boundary.

    Rejects embedded NULs (decode_text stops at the first NUL); use BLOB for
    arbitrary bytes. Strict UTF-8 only, so a surrogateescape decode result
    cannot round-trip back through here. ``max_size`` caps UTF-8 bytes (not
    codepoints) to stay symmetric with the decoder's byte-offset cap.
    """
    if not isinstance(value, str):
        raise EncodeError(f"encode_text expected str, got {type(value).__name__}")
    try:
        utf8 = value.encode("utf-8")
    except UnicodeEncodeError as e:
        raise EncodeError(
            f"Text cannot be encoded as UTF-8 (likely a lone surrogate or "
            f"surrogate-escape character): {e}"
        ) from e
    if max_size is not None and len(utf8) > max_size:
        raise EncodeError(f"{label} length {len(utf8)} exceeds maximum ({max_size})")
    nul_byte_offset = utf8.find(b"\x00")
    if nul_byte_offset != -1:
        raise EncodeError(
            f"Text value contains embedded null byte at byte offset "
            f"{nul_byte_offset}; null-terminated encoding would lose data"
        )
    # bytearray(N) zero-fills, so the NUL terminator and padding are already set.
    payload_len = len(utf8)
    padding = pad_to_word(payload_len + 1)
    buf = bytearray(payload_len + 1 + padding)
    buf[:payload_len] = utf8
    return bytes(buf)


# Below this, materialise a memoryview to bytes in one shot; above it, fall back
# to chunked scanning to bound peak memory for pathologically long texts.
_TEXT_ONE_SHOT_MAX: Final[int] = 65_536
_TEXT_SCAN_CHUNK: Final[int] = 4096
# Short-probe window: peek the first few hundred bytes before the 64 KiB one-shot
# to handle typical short cells (names, ids, timestamps) without a full allocation.
_TEXT_SHORT_PROBE: Final[int] = 256


def decode_text(
    data: bytes | memoryview,
    *,
    max_size: int = _MAX_TEXT_VALUE_SIZE,
    label: str = "Text",
    errors: str = "strict",
) -> tuple[str, int]:
    """Decode NUL-terminated UTF-8 text; return (text, bytes_consumed).

    ``errors="replace"``/``"surrogateescape"`` yield codepoints the log
    sanitiser misses and that break encode_text round-trip. Embedded NULs in
    server data read back truncated at the first NUL (no length prefix).
    """
    if isinstance(data, memoryview):
        data_len = len(data)
        # Short probe first: handles the common short-cell case in ~256 bytes
        # of materialise, avoiding allocator churn from the 64 KiB one-shot.
        short_window = min(data_len, max_size + 1, _TEXT_SHORT_PROBE)
        short_materialized = bytes(data[:short_window])
        short_null = short_materialized.find(b"\x00")
        if short_null >= 0:
            try:
                text = short_materialized[:short_null].decode("utf-8", errors=errors)
            except UnicodeDecodeError as e:
                raise DecodeError(f"Invalid UTF-8 in {label}: {e}") from e
            total_size = short_null + 1 + pad_to_word(short_null + 1)
            if data_len < total_size:
                raise DecodeError(
                    f"Not enough data for text padding: need {total_size}, got {data_len}"
                )
            return text, total_size
        if short_window >= max_size + 1:
            raise DecodeError(f"{label} length exceeds maximum ({max_size})")
        if short_window >= data_len:
            raise DecodeError(f"{label} not null-terminated")
        # Short probe missed; escalate to the 64 KiB one-shot probe. The
        # chunked path runs only for cells exceeding this window, keeping
        # transient memory bounded at <= 64 KiB.
        probe_window = min(data_len, max_size + 1, _TEXT_ONE_SHOT_MAX)
        materialized = bytes(data[:probe_window])
        null_pos = materialized.find(b"\x00")
        if null_pos >= 0:
            try:
                text = materialized[:null_pos].decode("utf-8", errors=errors)
            except UnicodeDecodeError as e:
                raise DecodeError(f"Invalid UTF-8 in {label}: {e}") from e
        elif probe_window >= max_size + 1:
            raise DecodeError(f"{label} length exceeds maximum ({max_size})")
        elif probe_window >= data_len:
            raise DecodeError(f"{label} not null-terminated")
        else:
            # Chunked scan from where the probe ended, reusing the probe
            # bytes as the first chunk.
            scan_limit = min(data_len, max_size + 1)
            chunks: list[bytes] = [materialized]
            scanned = probe_window
            chunked_null = -1
            while scanned < scan_limit:
                chunk_end = min(scanned + _TEXT_SCAN_CHUNK, scan_limit)
                chunk = bytes(data[scanned:chunk_end])
                local = chunk.find(b"\x00")
                if local >= 0:
                    chunks.append(chunk[:local])
                    chunked_null = scanned + local
                    break
                chunks.append(chunk)
                scanned = chunk_end
            if chunked_null < 0:
                if scan_limit < data_len:
                    raise DecodeError(f"{label} length exceeds maximum ({max_size})")
                raise DecodeError(f"{label} not null-terminated")
            null_pos = chunked_null
            try:
                text = b"".join(chunks).decode("utf-8", errors=errors)
            except UnicodeDecodeError as e:
                raise DecodeError(f"Invalid UTF-8 in {label}: {e}") from e
    else:
        # Bound the NUL scan at max_size+1 so a hostile unterminated payload
        # can't force a full-buffer memchr before the cap check fires.
        scan_end = min(len(data), max_size + 1)
        null_pos = data.find(b"\x00", 0, scan_end)
        if null_pos < 0:
            if scan_end < len(data):
                raise DecodeError(f"{label} length exceeds maximum ({max_size})")
            raise DecodeError(f"{label} not null-terminated")
        try:
            text = data[:null_pos].decode("utf-8", errors=errors)
        except UnicodeDecodeError as e:
            raise DecodeError(f"Invalid UTF-8 in {label}: {e}") from e

    # Every branch caps its scan at max_size+1 before the find, so
    # null_pos <= max_size here by construction; no tail check needed.
    total_size = null_pos + 1 + pad_to_word(null_pos + 1)
    if len(data) < total_size:
        raise DecodeError(f"Not enough data for text padding: need {total_size}, got {len(data)}")
    return text, total_size


def encode_blob(
    value: bytes | bytearray | memoryview, *, max_blob_size: int = _MAX_BLOB_SIZE
) -> bytes:
    """Encode a blob (uint64 length + data + padding to 8-byte boundary).

    Accepts any bytes-like input. ``max_blob_size`` is the per-blob cap.
    """
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise EncodeError(
            f"Blob value must be bytes, bytearray, or memoryview; got {type(value).__name__}"
        )
    length = len(value)
    if length > max_blob_size:
        raise EncodeError(f"Blob length {length} exceeds maximum ({max_blob_size})")
    # bytearray(N) zero-fills the padding; slice assignment accepts any
    # buffer-protocol object, so value is written in-place without materialising.
    padding = pad_to_word(length)
    buf = bytearray(8 + length + padding)
    buf[:8] = encode_uint64(length)
    buf[8 : 8 + length] = value
    return bytes(buf)


def decode_blob(
    data: bytes | memoryview, *, max_blob_size: int = _MAX_BLOB_SIZE
) -> tuple[bytes, int]:
    """Decode a blob; return (data, bytes_consumed).

    Result is always an owned ``bytes`` copy regardless of encode-side input
    shape, matching stdlib sqlite3 (memoryview on bind, bytes on fetch).
    """
    if len(data) < 8:
        raise DecodeError("Not enough data for blob length")

    length = decode_uint64(data[:8])
    if length > max_blob_size:
        raise DecodeError(f"Blob length {length} exceeds maximum ({max_blob_size})")
    total_size = 8 + length + pad_to_word(length)

    if len(data) < total_size:
        raise DecodeError(f"Not enough data for blob: need {total_size}, got {len(data)}")

    return bytes(data[8 : 8 + length]), total_size


def _infer_value_type(value: WireInput) -> ValueType:
    """Infer the wire ``ValueType`` from a Python value (tag only, no encode).

    Lets the row-header build path get a per-row type tuple at O(1) per cell
    instead of running the full encoder and discarding the bytes.
    """
    if value is None:
        return ValueType.NULL
    if isinstance(value, bool):
        return ValueType.BOOLEAN
    if isinstance(value, int):
        return ValueType.INTEGER
    if isinstance(value, float):
        return ValueType.FLOAT
    if isinstance(value, str):
        return ValueType.TEXT
    if isinstance(value, (bytes, bytearray, memoryview, mmap.mmap)):
        return ValueType.BLOB
    raise EncodeError(
        f"Cannot infer wire type for value of type {type(value).__name__!r}. "
        f"The wire codec only accepts bool, int, float, str, bytes-like, "
        f"or None. Conversion guidance for common date/time types:\n"
        f"  - datetime.datetime: str via .isoformat() (ValueType.ISO8601), "
        f"or int via int(dt.timestamp()) (ValueType.UNIXTIME, aware "
        f"datetimes only)\n"
        f"  - datetime.date: str via .isoformat() (ValueType.ISO8601) "
        f"only — no UNIXTIME mapping (UNIXTIME is seconds since epoch; "
        f"dates have no time component, so ordinal-vs-epoch conversions "
        f"silently round-trip to year 0001)\n"
        f"  - datetime.time: str via .isoformat() (ValueType.ISO8601) "
        f"only — no UNIXTIME mapping\n"
        f"  - decimal.Decimal / fractions.Fraction: float(x) "
        f"(ValueType.FLOAT) with explicit precision-loss "
        f"acknowledgement, or str(x) (ValueType.TEXT) for exact "
        f"preservation\n"
        f"Convert at the driver layer before binding."
    )


def encode_value(value: WireInput, value_type: ValueType | None = None) -> tuple[bytes, ValueType]:
    """Encode a Python value to wire format; return (encoded_data, value_type).

    If value_type is omitted it's inferred. bool infers to BOOLEAN (not INTEGER),
    but both encode identically on the wire and SQLite stores them as integers, so
    a round-trip through an INTEGER column returns int, never bool.
    """
    if value is None:
        if value_type is not None and value_type != ValueType.NULL:
            raise EncodeError(
                f"Cannot encode None with explicit type {value_type.name}. "
                f"Pass value_type=ValueType.NULL or omit value_type."
            )
        return b"\x00" * NULL_CELL_WIDTH, ValueType.NULL

    # Hoisted so the format check fires exactly once per call, before any
    # inferred-vs-explicit / BLOB-vs-other branching.
    if isinstance(value, memoryview):
        _reject_non_byte_format_memoryview(value)

    if value_type is None:
        value_type = _infer_value_type(value)

    if value_type == ValueType.BOOLEAN:
        # bool or exactly 0/1 only; reject arbitrary ints (5 -> True would lose
        # the value). C bind collapses any non-zero to 1, so re-encoding a
        # peer's out-of-{0,1} BOOLEAN is lossy by design.
        if isinstance(value, bool):
            return encode_uint64(1 if value else 0), value_type
        if isinstance(value, int) and value in (0, 1):
            return encode_uint64(value), value_type
        raise EncodeError(
            f"BOOLEAN requires bool (or exactly 0/1), "
            f"got {type(value).__name__}={_bounded_repr_any(value)}. "
            f"Numeric proxies (numpy.int64, numpy.bool_, Decimal, "
            f"Fraction, etc.) are not int subclasses and must be "
            f"coerced via int(x) or bool(x) at the bind site."
        )
    elif value_type in (ValueType.INTEGER, ValueType.UNIXTIME):
        # UNIXTIME is server-to-client only (C tuple_decoder has no inbound
        # case); explicit encoding here exists for round-trip tests. Reject
        # bool so an explicit-INTEGER bool doesn't silently encode as 1.
        if isinstance(value, bool):
            raise EncodeError(
                f"Cannot encode bool as {value_type.name}; cast with int(x) "
                "explicitly if integer semantics are intended."
            )
        if not isinstance(value, int):
            raise EncodeError(f"Expected int for {value_type.name}, got {type(value).__name__}")
        return encode_int64(value), value_type
    elif value_type == ValueType.FLOAT:
        # Reject int/bool: implicit float() coercion silently loses bits
        # (|x| >= 2**53) or overflows. Numeric proxies fall through to
        # encode_double, which rejects them with a richer message.
        if isinstance(value, bool):
            raise EncodeError(f"Expected float for FLOAT, got {type(value).__name__}")
        if isinstance(value, int):
            raise EncodeError(
                "Cannot encode int as FLOAT; cast with float(x) explicitly. "
                "Implicit coercion would silently lose precision for ints "
                "with absolute value >= 2**53."
            )
        return encode_double(cast(float, value)), value_type
    elif value_type in (ValueType.TEXT, ValueType.ISO8601):
        if not isinstance(value, str):
            raise EncodeError(f"Expected str for {value_type.name}, got {type(value).__name__}")
        if value_type == ValueType.ISO8601:
            # Probe-and-discard so a non-ISO string fails at the bind site, not
            # far away in the consumer. Mirrors the consumer's datetime/time
            # parser pair; diverges from C, which binds without parsing.
            try:
                datetime.datetime.fromisoformat(value)
            except ValueError:
                try:
                    datetime.time.fromisoformat(value)
                except ValueError as e:
                    raise EncodeError(
                        f"ISO8601 cell rejects non-ISO string "
                        f"{_bounded_repr_any(value)}: {e}. Use "
                        f"ValueType.TEXT if the column is plain text, "
                        f"or format the datetime via .isoformat() "
                        f"before binding."
                    ) from e
        # Cap text length so encode and decode stay round-trip-symmetric.
        return encode_text(value, max_size=_MAX_TEXT_VALUE_SIZE, label=value_type.name), value_type
    elif value_type == ValueType.BLOB:
        if not isinstance(value, (bytes, bytearray, memoryview, mmap.mmap)):
            raise EncodeError(f"Expected bytes for BLOB, got {type(value).__name__}")
        # Cap size before materialising via bytes(value) to avoid a wasted copy.
        # memoryview must probe .nbytes (not len(), the element count) so a
        # multi-byte view can't bypass the cap if the format check is ever moved.
        # A closed mmap / released memoryview raises here; fall through so the
        # materialise step surfaces the canonical EncodeError.
        try:
            length: int | None = value.nbytes if isinstance(value, memoryview) else len(value)
        except (ValueError, BufferError, TypeError):
            length = None
        if length is not None and length > _MAX_BLOB_SIZE:
            raise EncodeError(f"Blob length {length} exceeds maximum ({_MAX_BLOB_SIZE})")
        # Wrap materialise failures (closed mmap, released memoryview) as
        # EncodeError so the wire-layer all-failures-are-EncodeError convention holds.
        try:
            materialized = bytes(value)
        except (ValueError, BufferError) as e:
            raise EncodeError(f"Cannot materialise BLOB from {type(value).__name__}: {e}") from e
        return encode_blob(materialized), value_type
    elif value_type == ValueType.NULL:
        # None is handled by the early return above; a non-None value here is a bug.
        raise EncodeError(
            f"Cannot encode non-None value {_bounded_repr_any(value)} as NULL. "
            f"Pass value=None or use the appropriate ValueType."
        )
    else:
        raise EncodeError(f"Unknown value type: {value_type}")


def _decode_value_integer(data: bytes | memoryview, _text_errors: str) -> tuple[WireValue, int]:
    return decode_int64(data, label="INTEGER cell"), 8


def _decode_value_float(data: bytes | memoryview, _text_errors: str) -> tuple[WireValue, int]:
    return decode_double(data, label="FLOAT cell"), 8


def _decode_value_text(data: bytes | memoryview, text_errors: str) -> tuple[WireValue, int]:
    return decode_text(data, label="TEXT", errors=text_errors)


def _decode_value_iso8601(data: bytes | memoryview, text_errors: str) -> tuple[WireValue, int]:
    # ISO8601 is text on the wire; parsing to datetime is the driver layer's job.
    return decode_text(data, label="ISO8601", errors=text_errors)


def _decode_value_blob(data: bytes | memoryview, _text_errors: str) -> tuple[WireValue, int]:
    return decode_blob(data)


def _decode_value_null(data: bytes | memoryview, _text_errors: str) -> tuple[WireValue, int]:
    # Permissive: read and discard 8 bytes (matching Go/C, which read into an
    # unused union member). encode_value(None) always writes zeros, so a
    # captured non-zero NULL can't round-trip byte-identically through the encoder.
    decode_uint64(data, label="NULL cell")
    return None, NULL_CELL_WIDTH


def _decode_value_unixtime(data: bytes | memoryview, _text_errors: str) -> tuple[WireValue, int]:
    # Returns raw int64, not datetime: epoch-seconds is the wire primitive; the
    # datetime semantic is driver-layer. Context-blind on purpose — params-vs-row
    # rejection lives in decode_params_tuple, since the row decoder needs UNIXTIME.
    return decode_int64(data, label="UNIXTIME cell"), 8


def _decode_value_boolean(data: bytes | memoryview, _text_errors: str) -> tuple[WireValue, int]:
    # Permissive: read uint64 and coerce to bool, so a raw value outside {0,1}
    # (SQLite is dynamically typed — INSERT INTO t(b BOOLEAN) VALUES (5) is legal)
    # collapses to True instead of poisoning the decoder. Loses the original int.
    raw = decode_uint64(data, label="BOOLEAN cell")
    return bool(raw), 8


# Dispatch table indexed by value-type code: one dict lookup + call per cell
# instead of a 6-arm if/elif chain.
_VALUE_TYPE_DECODERS: Final[
    dict[int, Callable[[bytes | memoryview, str], tuple[WireValue, int]]]
] = {
    int(ValueType.INTEGER): _decode_value_integer,
    int(ValueType.FLOAT): _decode_value_float,
    int(ValueType.TEXT): _decode_value_text,
    int(ValueType.BLOB): _decode_value_blob,
    int(ValueType.NULL): _decode_value_null,
    int(ValueType.UNIXTIME): _decode_value_unixtime,
    int(ValueType.ISO8601): _decode_value_iso8601,
    int(ValueType.BOOLEAN): _decode_value_boolean,
}


def decode_value(
    data: bytes | memoryview,
    value_type: ValueType,
    *,
    text_errors: str = "strict",
) -> tuple[WireValue, int]:
    """Decode a value from wire format; return (value, bytes_consumed).

    ``text_errors`` is forwarded to decode_text for TEXT/ISO8601 cells; see its
    WARNING before using a non-strict mode.
    """
    decoder = _VALUE_TYPE_DECODERS.get(int(value_type))
    if decoder is None:
        raise DecodeError(f"Unknown value type: {value_type}")
    return decoder(data, text_errors)
