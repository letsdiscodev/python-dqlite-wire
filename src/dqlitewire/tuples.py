"""Tuple encoding/decoding for dqlite wire protocol."""

import struct
from collections.abc import Sequence
from enum import Enum
from typing import Final

from dqlitewire.constants import ROW_DONE_BYTE, ROW_PART_BYTE, ValueType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.types import WireInput, WireValue, decode_value, encode_value, pad_to_word

__all__ = [
    "RowMarker",
    "decode_params_tuple",
    "decode_row_header",
    "decode_row_values",
    "encode_params_tuple",
    "encode_row_header",
    "encode_row_values",
]

# Valid ValueType codes as integers, for fast membership testing in hot paths.
_VALID_TYPE_CODES: Final[frozenset[int]] = frozenset(int(v) for v in ValueType)

# Precomputed nibble → ValueType lookup, avoiding a ValueType() constructor
# call per cell in the decode_row_header hot loop.
_NIBBLE_TO_VALUETYPE: Final[tuple[ValueType | None, ...]] = tuple(
    ValueType(i) if i in _VALID_TYPE_CODES else None for i in range(16)
)

# Full 8-byte sentinels (DQLITE_RESPONSE_ROWS_DONE/PART), compared against
# data[:8] slices to reject torn markers. The uint64 int forms used at encode
# time live in constants.py (ROW_DONE_MARKER / ROW_PART_MARKER).
_ROW_DONE_MARKER: Final[bytes] = bytes([ROW_DONE_BYTE]) * 8
_ROW_PART_MARKER: Final[bytes] = bytes([ROW_PART_BYTE]) * 8

# Defense-in-depth cap matching SQLITE_MAX_VARIABLE_NUMBER (standard build);
# above this the upstream server cannot accept the bind anyway.
_MAX_PARAM_COUNT: Final[int] = 32_766


class RowMarker(Enum):
    DONE = "done"
    PART = "part"


def encode_params_tuple(
    params: Sequence[WireInput],
    schema: int = 0,
    buffer_offset: int = 0,
    *,
    emit_empty_header: bool = False,
) -> bytes:
    """Encode parameters as a params tuple (V0: uint8 count, V1: uint32 count).

    buffer_offset is the absolute body position used for padding (matching Go's
    m.Offset); it need not be word-aligned. Empty params emit zero bytes (Go
    style) by default, or 8 zero bytes (C style) when emit_empty_header is set,
    so a C-encoded snapshot can re-emit byte-identically.
    """
    if schema not in (0, 1):
        raise EncodeError(f"Unsupported params tuple schema version: {schema} (expected 0 or 1)")

    if len(params) > _MAX_PARAM_COUNT:
        raise EncodeError(f"Parameter count {len(params)} exceeds maximum ({_MAX_PARAM_COUNT})")

    if not params:
        if emit_empty_header:
            return b"\x00" * 8
        return b""

    types: list[int] = []
    values: list[bytes] = []

    for i, param in enumerate(params):
        encoded, value_type = encode_value(param)
        # UNIXTIME is server-to-client only; the server cannot decode it.
        if value_type == ValueType.UNIXTIME:
            raise EncodeError(
                f"Parameter {i} resolved to UNIXTIME, which the server "
                "cannot decode; use INTEGER or ISO8601 instead"
            )
        types.append(value_type)
        values.append(encoded)

    header = bytearray()
    if schema == 1:
        header.extend(struct.pack("<I", len(params)))
    else:
        if len(params) > 255:
            raise EncodeError(
                f"V0 params tuple supports at most 255 parameters, got {len(params)}. "
                f"Use schema=1 for larger parameter lists."
            )
        header.append(len(params))
    for t in types:
        header.append(t)

    # Pad based on the absolute buffer position, matching Go's m.Offset.
    absolute_offset = buffer_offset + len(header)
    padding = pad_to_word(absolute_offset)
    header.extend(b"\x00" * padding)

    for value_bytes in values:
        header.extend(value_bytes)
    return bytes(header)


def decode_params_tuple(
    data: bytes | memoryview,
    count: int | None = None,
    schema: int = 0,
    buffer_offset: int = 0,
) -> tuple[list[WireValue], int]:
    """Decode a params tuple; return (values, bytes_consumed).

    V0 uses a uint8 count, V1 a uint32 count. If count is given, the count
    field is assumed absent from data. See encode_params_tuple for the
    buffer_offset padding contract.
    """
    if schema not in (0, 1):
        raise DecodeError(f"Unsupported params tuple schema version: {schema} (expected 0 or 1)")

    # Go writes nothing for empty params.
    if len(data) == 0:
        if count is not None and count > 0:
            raise DecodeError(f"Expected data for {count} parameters, but data is empty")
        return [], 0

    if len(data) < 8:
        if count == 0:
            return [], 0
        raise DecodeError(f"Not enough data for params tuple header: got {len(data)}")

    count_from_data = count is None
    count_size: int
    if count is None:
        if schema == 1:
            count = struct.unpack("<I", data[:4])[0]
            count_size = 4
        else:
            count = data[0]
            count_size = 1

        if count == 0:
            # Count field was consumed; return padded header size.
            header_len = count_size
            consumed = header_len + pad_to_word(buffer_offset + header_len)
            return [], consumed
    elif count == 0:
        return [], 0

    if count > _MAX_PARAM_COUNT:
        raise DecodeError(f"Parameter count {count} exceeds maximum ({_MAX_PARAM_COUNT})")

    # When count was externally provided, data starts directly with type codes.
    count_size = (4 if schema == 1 else 1) if count_from_data else 0
    header_len = count_size + count
    padded_header_len = header_len + pad_to_word(buffer_offset + header_len)

    if len(data) < padded_header_len:
        raise DecodeError(
            f"Not enough data for param types: need {padded_header_len}, got {len(data)}"
        )

    types: list[ValueType] = []
    for i in range(count):
        raw_type = data[count_size + i]
        try:
            value_type = ValueType(raw_type)
        except ValueError as e:
            raise DecodeError(f"Invalid value type code {raw_type} at param index {i}") from e
        # UNIXTIME is server-to-client only; the upstream C gateway rejects it
        # on inbound params, so reject it here too (else mock servers accept
        # traffic that regresses against a real cluster).
        if value_type == ValueType.UNIXTIME:
            raise DecodeError(
                f"UNIXTIME tag at param index {i}: server-to-client only; "
                "the upstream C gateway rejects DQLITE_UNIXTIME on inbound "
                "params with DQLITE_PARSE"
            )
        types.append(value_type)
    offset = padded_header_len

    values: list[WireValue] = []
    for vtype in types:
        value, consumed = decode_value(data[offset:], vtype)
        values.append(value)
        offset += consumed

    return values, offset


def encode_row_header(types: Sequence[ValueType]) -> bytes:
    """Encode row column type header: 4-bit codes packed two per byte, word-padded."""
    if not types:
        return b""

    for i, t in enumerate(types):
        code = int(t)
        if code > 15:
            raise EncodeError(f"Value type {t} at index {i} exceeds 4-bit nibble range (max 15)")
        if code not in _VALID_TYPE_CODES:
            raise EncodeError(f"Invalid type code {code} at index {i}: not a valid ValueType")

    header = bytearray()
    for i in range(0, len(types), 2):
        low = types[i]
        high = types[i + 1] if i + 1 < len(types) else 0
        header.append((high << 4) | low)

    padding = pad_to_word(len(header))
    header.extend(b"\x00" * padding)
    return bytes(header)


def decode_row_header(
    data: bytes | memoryview, column_count: int
) -> tuple[list[ValueType] | RowMarker, int]:
    """Decode row column type header; return (types_or_marker, bytes_consumed).

    Format: 4-bit codes packed two per byte, word-padded. Row markers
    (DONE=0xFF..FF, PART=0xEE..EE) are matched on all 8 bytes (tighter than
    Go's first-byte check) so torn markers are rejected rather than silently
    truncating the stream.
    """
    # Markers are always exactly one 8-byte word; check before the size
    # validation, since a large column count makes the header >8 bytes.
    if len(data) >= 8:
        # memoryview compares against bytes via the buffer protocol, no copy.
        prefix = data[:8]
        if prefix == _ROW_DONE_MARKER:
            return RowMarker.DONE, 8
        if prefix == _ROW_PART_MARKER:
            return RowMarker.PART, 8

    bytes_for_types = (column_count + 1) // 2
    header_size = bytes_for_types + pad_to_word(bytes_for_types)

    if len(data) < header_size:
        raise DecodeError(f"Not enough data for row header: need {header_size}, got {len(data)}")

    types: list[ValueType] = []
    for i in range(column_count):
        byte_idx = i // 2
        nibble = data[byte_idx] & 0x0F if i % 2 == 0 else (data[byte_idx] >> 4) & 0x0F
        vt = _NIBBLE_TO_VALUETYPE[nibble]
        if vt is None:
            # Replay the constructor to chain the ValueError as __cause__.
            try:
                ValueType(nibble)
            except ValueError as e:
                raise DecodeError(f"Invalid value type code {nibble} at column index {i}") from e
            raise AssertionError(f"unreachable: nibble {nibble} accepted by ValueType")
        types.append(vt)

    return types, header_size


def encode_row_values(values: Sequence[WireInput], types: Sequence[ValueType]) -> bytes:
    """Encode row values according to specified types."""
    if len(values) != len(types):
        raise EncodeError(
            f"Row values count ({len(values)}) does not match types count ({len(types)})"
        )
    result = bytearray()
    for value, vtype in zip(values, types, strict=True):
        encoded, _ = encode_value(value, vtype)
        result.extend(encoded)
    return bytes(result)


def decode_row_values(
    data: bytes | memoryview,
    types: Sequence[ValueType],
    *,
    text_errors: str = "strict",
) -> tuple[list[WireValue], int]:
    """Decode row values according to column types; return (values, bytes_consumed).

    text_errors is forwarded to decode_value for TEXT/ISO8601 cells; the default
    "strict" matches the wire-spec contract.
    """
    values: list[WireValue] = []
    offset = 0

    for vtype in types:
        value, consumed = decode_value(data[offset:], vtype, text_errors=text_errors)
        values.append(value)
        offset += consumed

    return values, offset
