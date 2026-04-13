"""Primitive type encoding/decoding for dqlite wire protocol.

All multi-byte integers are little-endian.
Text is null-terminated UTF-8, padded to 8-byte boundary.
"""

import datetime
import struct
from typing import Any

from dqlitewire.constants import WORD_SIZE, ValueType
from dqlitewire.exceptions import DecodeError, EncodeError


def encode_uint64(value: int) -> bytes:
    """Encode an unsigned 64-bit integer (little-endian)."""
    if not 0 <= value < 2**64:
        raise EncodeError(f"Value {value} out of range for uint64")
    return struct.pack("<Q", value)


def decode_uint64(data: bytes) -> int:
    """Decode an unsigned 64-bit integer (little-endian)."""
    if len(data) < 8:
        raise DecodeError(f"Need 8 bytes for uint64, got {len(data)}")
    result: int = struct.unpack("<Q", data[:8])[0]
    return result


def encode_int64(value: int) -> bytes:
    """Encode a signed 64-bit integer (little-endian)."""
    if not -(2**63) <= value < 2**63:
        raise EncodeError(f"Value {value} out of range for int64")
    return struct.pack("<q", value)


def decode_int64(data: bytes) -> int:
    """Decode a signed 64-bit integer (little-endian)."""
    if len(data) < 8:
        raise DecodeError(f"Need 8 bytes for int64, got {len(data)}")
    result: int = struct.unpack("<q", data[:8])[0]
    return result


def encode_uint32(value: int) -> bytes:
    """Encode an unsigned 32-bit integer (little-endian)."""
    if not 0 <= value < 2**32:
        raise EncodeError(f"Value {value} out of range for uint32")
    return struct.pack("<I", value)


def decode_uint32(data: bytes) -> int:
    """Decode an unsigned 32-bit integer (little-endian)."""
    if len(data) < 4:
        raise DecodeError(f"Need 4 bytes for uint32, got {len(data)}")
    result: int = struct.unpack("<I", data[:4])[0]
    return result


def encode_double(value: float) -> bytes:
    """Encode a 64-bit floating point number (little-endian).

    All IEEE 754 values are accepted, including NaN and infinity,
    matching the Go reference implementation behavior.
    """
    return struct.pack("<d", value)


def decode_double(data: bytes) -> float:
    """Decode a 64-bit floating point number (little-endian).

    All IEEE 754 values are accepted, including NaN and infinity,
    matching the Go reference implementation behavior.
    """
    if len(data) < 8:
        raise DecodeError(f"Need 8 bytes for double, got {len(data)}")
    result: float = struct.unpack("<d", data[:8])[0]
    return result


def pad_to_word(size: int) -> int:
    """Calculate padding needed to align to word boundary."""
    remainder = size % WORD_SIZE
    if remainder == 0:
        return 0
    return WORD_SIZE - remainder


def encode_text(value: str) -> bytes:
    """Encode text as null-terminated UTF-8, padded to 8-byte boundary."""
    if "\x00" in value:
        raise EncodeError(
            f"Text value contains embedded null byte at position {value.index(chr(0))}; "
            "null-terminated encoding would lose data"
        )
    encoded = value.encode("utf-8") + b"\x00"
    padding = pad_to_word(len(encoded))
    return encoded + (b"\x00" * padding)


def decode_text(data: bytes) -> tuple[str, int]:
    """Decode null-terminated UTF-8 text.

    Returns the decoded string and the number of bytes consumed (including padding).
    """
    # Find null terminator
    try:
        null_pos = data.index(b"\x00")
    except ValueError as e:
        raise DecodeError("Text not null-terminated") from e

    try:
        text = data[:null_pos].decode("utf-8")
    except UnicodeDecodeError as e:
        raise DecodeError(f"Invalid UTF-8 in text field: {e}") from e
    # Calculate total size including padding
    total_size = null_pos + 1 + pad_to_word(null_pos + 1)
    if len(data) < total_size:
        raise DecodeError(f"Not enough data for text padding: need {total_size}, got {len(data)}")
    return text, total_size


def encode_blob(value: bytes) -> bytes:
    """Encode a blob (length-prefixed binary data, padded to 8-byte boundary).

    Format: uint64 length + data + padding
    """
    length = len(value)
    padding = pad_to_word(length)
    return encode_uint64(length) + value + (b"\x00" * padding)


def decode_blob(data: bytes) -> tuple[bytes, int]:
    """Decode a blob.

    Returns the blob data and the number of bytes consumed.
    """
    if len(data) < 8:
        raise DecodeError("Not enough data for blob length")

    length = decode_uint64(data[:8])
    total_size = 8 + length + pad_to_word(length)

    if len(data) < total_size:
        raise DecodeError(f"Not enough data for blob: need {total_size}, got {len(data)}")

    return data[8 : 8 + length], total_size


def _format_datetime_iso8601(value: datetime.datetime) -> str:
    """Format a datetime to ISO 8601 string matching Go's time format."""
    formatted = f"{value.year:04d}" + value.strftime("-%m-%d %H:%M:%S")
    if value.microsecond:
        formatted += f".{value.microsecond:06d}".rstrip("0")
    utcoffset = value.utcoffset()
    if utcoffset is not None:
        total_seconds = int(utcoffset.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        hours, remainder = divmod(abs(total_seconds), 3600)
        minutes = remainder // 60
        formatted += f"{sign}{hours:02d}:{minutes:02d}"
    else:
        # Naive datetime: assume UTC to match Go's always-offset format
        formatted += "+00:00"
    return formatted


def encode_value(value: Any, value_type: ValueType | None = None) -> tuple[bytes, ValueType]:
    """Encode a Python value to wire format.

    If value_type is not provided, it's inferred from the Python type.
    Returns (encoded_data, value_type).
    """
    if value is None:
        if value_type is not None and value_type != ValueType.NULL:
            raise EncodeError(
                f"Cannot encode None with explicit type {value_type.name}. "
                f"Pass value_type=ValueType.NULL or omit value_type."
            )
        return b"\x00" * 8, ValueType.NULL

    if value_type is None:
        if isinstance(value, bool):
            value_type = ValueType.BOOLEAN
        elif isinstance(value, int):
            value_type = ValueType.INTEGER
        elif isinstance(value, float):
            value_type = ValueType.FLOAT
        elif isinstance(value, datetime.datetime):
            value_type = ValueType.ISO8601
            value = _format_datetime_iso8601(value)
        elif isinstance(value, datetime.date):
            # Must come after datetime.datetime check (datetime is a subclass of date)
            value_type = ValueType.ISO8601
            value = value.isoformat()
        elif isinstance(value, str):
            value_type = ValueType.TEXT
        elif isinstance(value, bytes):
            value_type = ValueType.BLOB
        else:
            raise EncodeError(f"Cannot infer type for value: {type(value)}")

    if value_type == ValueType.BOOLEAN:
        if not isinstance(value, (bool, int)):
            raise EncodeError(f"Expected bool or int for BOOLEAN, got {type(value).__name__}")
        return encode_uint64(1 if value else 0), value_type
    elif value_type in (ValueType.INTEGER, ValueType.UNIXTIME):
        if isinstance(value, bool):
            value = 1 if value else 0
        if not isinstance(value, int):
            raise EncodeError(f"Expected int for {value_type.name}, got {type(value).__name__}")
        return encode_int64(value), value_type
    elif value_type == ValueType.FLOAT:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EncodeError(f"Expected int or float for FLOAT, got {type(value).__name__}")
        return encode_double(float(value)), value_type
    elif value_type in (ValueType.TEXT, ValueType.ISO8601):
        # ISO8601 accepts datetime objects and converts them to string.
        # This is needed when encode_row_values passes a datetime with
        # explicit ISO8601 type (the auto-inference path converts earlier).
        if value_type == ValueType.ISO8601 and isinstance(value, datetime.datetime):
            value = _format_datetime_iso8601(value)
        elif value_type == ValueType.ISO8601 and isinstance(value, datetime.date):
            value = value.isoformat()
        if not isinstance(value, str):
            raise EncodeError(f"Expected str for {value_type.name}, got {type(value).__name__}")
        return encode_text(value), value_type
    elif value_type == ValueType.BLOB:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise EncodeError(f"Expected bytes for BLOB, got {type(value).__name__}")
        return encode_blob(bytes(value)), value_type
    elif value_type == ValueType.NULL:
        # None is already handled in the early-return branch above, so reaching
        # here means value is not None with explicit NULL type — always a bug.
        raise EncodeError(
            f"Cannot encode non-None value {value!r} as NULL. "
            f"Pass value=None or use the appropriate ValueType."
        )
    else:
        raise EncodeError(f"Unknown value type: {value_type}")


def _parse_iso8601(text: str) -> datetime.datetime:
    """Parse an ISO 8601 datetime string, trying multiple formats.

    Matches Go's iso8601Formats parsing which tries 9 patterns including
    with/without timezone, with/without fractional seconds, T vs space
    separator, and date-only.
    """
    # Strip trailing Z (Go does this before parsing)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    # Try Python's fromisoformat first — handles most formats since Python 3.11
    try:
        dt = datetime.datetime.fromisoformat(text)
        # Go parses all formats without explicit timezone in UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.UTC)
        return dt
    except ValueError:
        pass

    # Fallback: date-only format
    try:
        d = datetime.date.fromisoformat(text)
        return datetime.datetime(d.year, d.month, d.day, tzinfo=datetime.UTC)
    except ValueError:
        pass

    raise DecodeError(f"Cannot parse ISO 8601 datetime: {text!r}")


def decode_value(data: bytes, value_type: ValueType) -> tuple[Any, int]:
    """Decode a value from wire format.

    Returns (value, bytes_consumed).
    """
    if value_type == ValueType.BOOLEAN:
        return bool(decode_uint64(data)), 8
    elif value_type == ValueType.INTEGER:
        return decode_int64(data), 8
    elif value_type == ValueType.UNIXTIME:
        # Return raw int64 to match Go's getInt64() behavior and preserve
        # round-trip identity. Previously returned datetime.datetime, which
        # caused type-changing re-encode (UNIXTIME → ISO8601).
        return decode_int64(data), 8
    elif value_type == ValueType.FLOAT:
        return decode_double(data), 8
    elif value_type == ValueType.TEXT:
        return decode_text(data)
    elif value_type == ValueType.ISO8601:
        text, consumed = decode_text(data)
        if not text:
            return None, consumed
        return _parse_iso8601(text), consumed
    elif value_type == ValueType.BLOB:
        return decode_blob(data)
    elif value_type == ValueType.NULL:
        if len(data) < 8:
            raise DecodeError(f"Need 8 bytes for NULL value, got {len(data)}")
        return None, 8
    else:
        raise DecodeError(f"Unknown value type: {value_type}")
