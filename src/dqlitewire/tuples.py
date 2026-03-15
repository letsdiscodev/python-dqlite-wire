"""Tuple encoding/decoding for dqlite wire protocol.

Row tuples and parameter tuples have different formats:
- Parameter tuples: value type followed by value, for each parameter
- Row tuples: column types header, then values
"""

from collections.abc import Sequence
from enum import Enum
from typing import Any

from dqlitewire.constants import ROW_DONE_BYTE, ROW_PART_BYTE, ValueType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.types import decode_value, encode_value, pad_to_word


class RowMarker(Enum):
    """Row marker detected during header parsing."""

    DONE = "done"
    PART = "part"


def encode_params_tuple(params: Sequence[Any], schema: int = 0) -> bytes:
    """Encode parameters as a params tuple.

    Schema 0 (V0): uint8 count + type codes + padding + values (max 255 params)
    Schema 1 (V1): uint32 count + type codes + padding + values (max ~4B params)
    """
    if not params:
        # Go writes nothing for empty params
        return b""

    # First, encode all values and collect types
    types: list[int] = []
    values: list[bytes] = []

    for param in params:
        encoded, value_type = encode_value(param)
        types.append(value_type)
        values.append(encoded)

    # Build header: count field + type codes
    header = bytearray()
    if schema == 1:
        # V1: uint32 count
        import struct

        header.extend(struct.pack("<I", len(params)))
    else:
        # V0: uint8 count
        if len(params) > 255:
            raise EncodeError(
                f"V0 params tuple supports at most 255 parameters, got {len(params)}. "
                f"Use schema=1 for larger parameter lists."
            )
        header.append(len(params))
    for t in types:
        header.append(t)

    # Pad header to word boundary.
    # NOTE: Go pads based on the absolute buffer offset, not the relative
    # header length. This produces the same result only when the params tuple
    # starts at a word-aligned offset within the message body. All existing
    # message types satisfy this (ExecRequest/QueryRequest: 4+4=8 bytes
    # before params; ExecSqlRequest/QuerySqlRequest: 8 + word-aligned text).
    padding = pad_to_word(len(header))
    header.extend(b"\x00" * padding)

    # Concatenate header and values
    return bytes(header) + b"".join(values)


def decode_params_tuple(
    data: bytes, count: int | None = None, schema: int = 0
) -> tuple[list[Any], int]:
    """Decode a params tuple.

    Schema 0 (V0): uint8 count, schema 1 (V1): uint32 count.
    If count is None, reads the count from data.
    Returns (values, bytes_consumed).
    """
    # Go writes nothing for empty params
    if len(data) == 0:
        return [], 0

    if len(data) < 8:
        if count == 0:
            return [], 0
        raise DecodeError(f"Not enough data for params tuple header: got {len(data)}")

    # Read count from data if not provided
    count_size: int
    if count is None:
        if schema == 1:
            import struct

            count = struct.unpack("<I", data[:4])[0]
            count_size = 4
        else:
            count = data[0]
            count_size = 1

        if count == 0:
            # Count field was consumed; return padded header size
            header_len = count_size
            consumed = header_len + pad_to_word(header_len)
            return [], consumed
    elif count == 0:
        # Count was externally provided, no data consumed
        return [], 0

    # Header: count field + type codes, padded to word boundary
    count_size = 4 if schema == 1 else 1
    header_len = count_size + count
    padded_header_len = header_len + pad_to_word(header_len)

    if len(data) < padded_header_len:
        raise DecodeError(
            f"Not enough data for param types: need {padded_header_len}, got {len(data)}"
        )

    # Read type codes (skip count field)
    types: list[ValueType] = []
    for i in range(count):
        raw_type = data[count_size + i]
        try:
            types.append(ValueType(raw_type))
        except ValueError:
            raise DecodeError(f"Invalid value type code {raw_type} at param index {i}") from None
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

    for i, t in enumerate(types):
        if int(t) > 15:
            raise EncodeError(f"Value type {t} at index {i} exceeds 4-bit nibble range (max 15)")

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
    Detects row markers (0xFF=done, 0xEE=part) by checking the first byte
    before validating header size, matching the Go reference implementation.
    Returns (types_or_marker, bytes_consumed).
    """
    if column_count == 0:
        return [], 0

    # Check for markers first — markers are always exactly one 8-byte word,
    # regardless of column count. Must check before header size validation
    # because for large column counts the header would be >8 bytes.
    # Go checks the first byte (0xFF -> DONE, 0xEE -> PART); we match that
    # behavior so non-uniform markers are also detected.
    if len(data) >= 8:
        first_byte = data[0]
        if first_byte == ROW_DONE_BYTE:
            return RowMarker.DONE, 8
        if first_byte == ROW_PART_BYTE:
            return RowMarker.PART, 8

    # Calculate bytes needed: 2 types per byte, rounded up
    bytes_for_types = (column_count + 1) // 2
    header_size = bytes_for_types + pad_to_word(bytes_for_types)

    if len(data) < header_size:
        raise DecodeError(f"Not enough data for row header: need {header_size}, got {len(data)}")

    types: list[ValueType] = []
    for i in range(column_count):
        byte_idx = i // 2
        nibble = data[byte_idx] & 0x0F if i % 2 == 0 else (data[byte_idx] >> 4) & 0x0F
        try:
            types.append(ValueType(nibble))
        except ValueError:
            raise DecodeError(f"Invalid value type code {nibble} at column index {i}") from None

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
