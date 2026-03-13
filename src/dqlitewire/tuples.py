"""Tuple encoding/decoding for dqlite wire protocol.

Row tuples and parameter tuples have different formats:
- Parameter tuples: value type followed by value, for each parameter
- Row tuples: column types header, then values
"""

from collections.abc import Sequence
from enum import Enum
from typing import Any

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError
from dqlitewire.types import decode_value, encode_value, pad_to_word

# Marker bytes detected per-byte in row headers (matching Go implementation)
ROW_DONE_BYTE = 0xFF
ROW_PART_BYTE = 0xEE


class RowMarker(Enum):
    """Row marker detected during header parsing."""

    DONE = "done"
    PART = "part"


def encode_params_tuple(params: Sequence[Any]) -> bytes:
    """Encode parameters as a params tuple.

    Format per wire protocol spec:
    - 1 byte: count of parameters
    - N bytes: type codes (one byte per parameter)
    - Zero padding to word boundary (including count byte)
    - Values encoded according to their types
    """
    if not params:
        # Even with no params, we send count=0 padded to word boundary
        return b"\x00" * 8

    # First, encode all values and collect types
    types: list[int] = []
    values: list[bytes] = []

    for param in params:
        encoded, value_type = encode_value(param)
        types.append(value_type)
        values.append(encoded)

    # Build header: count byte + type codes
    header = bytearray()
    header.append(len(params))  # count byte
    for t in types:
        header.append(t)

    # Pad header to word boundary
    padding = pad_to_word(len(header))
    header.extend(b"\x00" * padding)

    # Concatenate header and values
    return bytes(header) + b"".join(values)


def decode_params_tuple(data: bytes, count: int | None = None) -> tuple[list[Any], int]:
    """Decode a params tuple.

    If count is None, reads the count from the first byte of data.
    Returns (values, bytes_consumed).
    """
    if len(data) < 8:
        if count == 0:
            return [], 0
        raise DecodeError(f"Not enough data for params tuple header: got {len(data)}")

    # Read count from first byte if not provided
    if count is None:
        count = data[0]

    if count == 0:
        # Empty params still have 8-byte header (count + padding)
        return [], 8

    # Header: count byte + type codes, padded to word boundary
    header_len = 1 + count  # count byte + type codes
    padded_header_len = header_len + pad_to_word(header_len)

    if len(data) < padded_header_len:
        raise DecodeError(
            f"Not enough data for param types: need {padded_header_len}, got {len(data)}"
        )

    # Read type codes (skip count byte at index 0)
    types = [ValueType(data[1 + i]) for i in range(count)]
    offset = padded_header_len

    # Read values
    values: list[Any] = []
    for vtype in types:
        value, consumed = decode_value(data[offset:], vtype)
        values.append(value)
        offset += consumed

    return values, offset


def encode_row_header(types: Sequence[ValueType]) -> bytes:
    """Encode row column type header.

    Format: 4-bit type codes packed two per byte, padded to word boundary.
    """
    if not types:
        return b""

    header = bytearray()
    for i in range(0, len(types), 2):
        low = types[i]
        high = types[i + 1] if i + 1 < len(types) else 0
        # Pack two 4-bit codes into one byte: low in lower nibble, high in upper
        header.append((high << 4) | low)

    # Pad to word boundary
    padding = pad_to_word(len(header))
    header.extend(b"\x00" * padding)
    return bytes(header)


def decode_row_header(data: bytes, column_count: int) -> tuple[list[ValueType] | RowMarker, int]:
    """Decode row column type header.

    Format: 4-bit type codes packed two per byte, padded to word boundary.
    Detects row markers (0xFF=done, 0xEE=part) byte-by-byte during parsing,
    matching the Go reference implementation.
    Returns (types_or_marker, bytes_consumed).
    """
    if column_count == 0:
        return [], 0

    # Calculate bytes needed: 2 types per byte, rounded up
    bytes_for_types = (column_count + 1) // 2
    header_size = bytes_for_types + pad_to_word(bytes_for_types)

    if len(data) < header_size:
        raise DecodeError(f"Not enough data for row header: need {header_size}, got {len(data)}")

    types: list[ValueType] = []
    for i in range(column_count):
        byte_idx = i // 2
        if i % 2 == 0:
            # Check for marker bytes at the start of each slot (matching Go)
            slot = data[byte_idx]
            if slot == ROW_DONE_BYTE:
                return RowMarker.DONE, header_size
            if slot == ROW_PART_BYTE:
                return RowMarker.PART, header_size
            types.append(ValueType(slot & 0x0F))
        else:
            types.append(ValueType((data[byte_idx] >> 4) & 0x0F))

    return types, header_size


def encode_row_values(values: Sequence[Any], types: Sequence[ValueType]) -> bytes:
    """Encode row values according to specified types."""
    result = bytearray()
    for value, vtype in zip(values, types, strict=True):
        encoded, _ = encode_value(value, vtype)
        result.extend(encoded)
    return bytes(result)


def decode_row_values(data: bytes, types: Sequence[ValueType]) -> tuple[list[Any], int]:
    """Decode row values according to column types.

    Returns (values, bytes_consumed).
    """
    values: list[Any] = []
    offset = 0

    for vtype in types:
        value, consumed = decode_value(data[offset:], vtype)
        values.append(value)
        offset += consumed

    return values, offset
