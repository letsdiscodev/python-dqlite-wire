"""Tuple encoding/decoding for dqlite wire protocol.

Row tuples and parameter tuples have different formats:
- Parameter tuples: value type followed by value, for each parameter
- Row tuples: column types header, then values
"""

import struct
from collections.abc import Sequence
from enum import Enum
from typing import Any

from dqlitewire.constants import ROW_DONE_BYTE, ROW_PART_BYTE, ValueType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.types import decode_value, encode_value, pad_to_word

# Valid ValueType codes as integers, for fast membership testing in hot paths.
_VALID_TYPE_CODES = frozenset(int(v) for v in ValueType)

# Full 8-byte sentinels matching DQLITE_RESPONSE_ROWS_DONE/PART. Used to
# reject torn/corrupt row markers instead of accepting any 8 bytes that
# happen to start with 0xff/0xee.
_ROW_DONE_MARKER = bytes([ROW_DONE_BYTE]) * 8
_ROW_PART_MARKER = bytes([ROW_PART_BYTE]) * 8

# Defense-in-depth cap on parameter count, matching the pattern used by
# _MAX_COLUMN_COUNT, _MAX_FILE_COUNT, and _MAX_NODE_COUNT in responses.py.
# SQLite's default limit is 999 (compile-time max 32766).
_MAX_PARAM_COUNT = 100_000


class RowMarker(Enum):
    """Row marker detected during header parsing."""

    DONE = "done"
    PART = "part"


def encode_params_tuple(
    params: Sequence[Any],
    schema: int = 0,
    buffer_offset: int = 0,
    *,
    emit_empty_header: bool = False,
) -> bytes:
    """Encode parameters as a params tuple.

    Schema 0 (V0): uint8 count + type codes + padding + values (max 255 params)
    Schema 1 (V1): uint32 count + type codes + padding + values (max ~4B params)

    Args:
        params: Parameter values to encode.
        schema: 0 for V0 (uint8 count), 1 for V1 (uint32 count).
        buffer_offset: Absolute byte offset where this tuple starts in the
            message body. Used to compute padding from the absolute position,
            matching Go's putNamedValues which pads based on m.Offset.
            Need not be word-aligned — the padding math handles any
            offset and the encode/decode pair round-trip faithfully
            (``TestParamsTupleBufferOffset`` pins this). In-tree request
            bodies happen to pass 8-aligned offsets (params tuple
            always starts at a word boundary) but the function is not
            restricted to that.
        emit_empty_header: When ``True`` and ``params`` is empty, emit
            8 zero bytes (C-style: count + padding) instead of the
            Go-style 0 bytes. Used by request encoders carrying a
            ``_decoded_schema`` round-trip hint so a foreign-encoded
            empty-params body re-encodes byte-identically. Default
            False matches the Go reference and the project's own
            self-originated requests, where the Go-style 0-byte form
            is canonical.

    Empty parameters: this implementation matches go-dqlite by default
    and emits zero bytes when ``params`` is empty. The C reference
    (``tuple_encoder__init`` in ``dqlite-upstream/src/tuple.c``) instead
    always writes 1 byte for the count followed by 7 bytes of padding
    (8 bytes total). Both shapes round-trip through
    ``decode_params_tuple`` (the decoder accepts a 0-byte body returning
    ``([], 0)`` and an 8-byte zero body returning ``([], 8)`` — see
    ``test_decode_empty`` and ``test_decode_zero_count_v0_consumes_header``).
    A C-encoded snapshot can be re-emitted byte-identically by passing
    ``emit_empty_header=True`` (used by request encoders that preserved
    the schema byte via ``_decoded_schema``).
    """
    if schema not in (0, 1):
        raise EncodeError(f"Unsupported params tuple schema version: {schema} (expected 0 or 1)")

    # Mirror the decoder's defense-in-depth cap so that a caller who
    # accidentally passes an enormous params sequence (e.g. from a bug
    # that builds WHERE IN (?,...) with millions of placeholders) fails
    # fast with a clear message instead of burning allocations until the
    # 64 MiB frame cap fires with an opaque "buffer size exceeds maximum"
    # error.
    if len(params) > _MAX_PARAM_COUNT:
        raise EncodeError(f"Parameter count {len(params)} exceeds maximum ({_MAX_PARAM_COUNT})")

    if not params:
        if emit_empty_header:
            # C-style: 1-byte uint8 count (V0) or 4-byte uint32 count
            # (V1) followed by zero-padding to 8 bytes. Both forms
            # produce 8 zero bytes since the count itself is zero —
            # the decoder uses the schema flag to know how many bytes
            # of header to consume, but the actual byte content is
            # eight zeros either way.
            return b"\x00" * 8
        # Go writes nothing for empty params (default).
        return b""

    # First, encode all values and collect types
    types: list[int] = []
    values: list[bytes] = []

    for i, param in enumerate(params):
        encoded, value_type = encode_value(param)
        # Defense in depth: UNIXTIME is server-to-client only. Inference
        # never picks it, but guard against a future caller threading an
        # explicit UNIXTIME hint through this path.
        if value_type == ValueType.UNIXTIME:
            raise EncodeError(
                f"Parameter {i} resolved to UNIXTIME, which the server "
                "cannot decode; use INTEGER or ISO8601 instead"
            )
        types.append(value_type)
        values.append(encoded)

    # Build header: count field + type codes
    header = bytearray()
    if schema == 1:
        # V1: uint32 count
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

    # Pad to word boundary using absolute offset, matching Go's behavior.
    # Go pads based on m.Offset (the absolute buffer position), not the
    # relative header length.
    absolute_offset = buffer_offset + len(header)
    padding = pad_to_word(absolute_offset)
    header.extend(b"\x00" * padding)

    # Concatenate header and values
    return bytes(header) + b"".join(values)


def decode_params_tuple(
    data: bytes | memoryview,
    count: int | None = None,
    schema: int = 0,
    buffer_offset: int = 0,
) -> tuple[list[Any], int]:
    """Decode a params tuple.

    Schema 0 (V0): uint8 count, schema 1 (V1): uint32 count.
    If count is None, reads the count from data.
    Returns (values, bytes_consumed).

    Args:
        data: Raw bytes to decode from.
        count: If provided, the number of params (count field not in data).
        schema: 0 for V0, 1 for V1.
        buffer_offset: Absolute byte offset where this tuple starts in the
            message body. Used for padding calculation matching Go's
            behaviour. Need not be word-aligned — see
            ``encode_params_tuple`` for the round-trip contract.
    """
    if schema not in (0, 1):
        raise DecodeError(f"Unsupported params tuple schema version: {schema} (expected 0 or 1)")

    # Go writes nothing for empty params
    if len(data) == 0:
        if count is not None and count > 0:
            raise DecodeError(f"Expected data for {count} parameters, but data is empty")
        return [], 0

    if len(data) < 8:
        if count == 0:
            return [], 0
        raise DecodeError(f"Not enough data for params tuple header: got {len(data)}")

    # Read count from data if not provided
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
            # Count field was consumed; return padded header size
            header_len = count_size
            consumed = header_len + pad_to_word(buffer_offset + header_len)
            return [], consumed
    elif count == 0:
        # Count was externally provided, no data consumed
        return [], 0

    # Defense-in-depth: reject unreasonable parameter counts early,
    # before allocating type-code lists or iterating over values.
    if count > _MAX_PARAM_COUNT:
        raise DecodeError(f"Parameter count {count} exceeds maximum ({_MAX_PARAM_COUNT})")

    # Header: count field (if read from data) + type codes, padded to word boundary.
    # When count was externally provided, the data starts directly with type codes
    # so count_size is 0.
    count_size = (4 if schema == 1 else 1) if count_from_data else 0
    header_len = count_size + count
    padded_header_len = header_len + pad_to_word(buffer_offset + header_len)

    if len(data) < padded_header_len:
        raise DecodeError(
            f"Not enough data for param types: need {padded_header_len}, got {len(data)}"
        )

    # Read type codes (skip count field)
    types: list[ValueType] = []
    for i in range(count):
        raw_type = data[count_size + i]
        try:
            value_type = ValueType(raw_type)
        except ValueError as e:
            raise DecodeError(f"Invalid value type code {raw_type} at param index {i}") from e
        # Symmetric with the encode-side reject in
        # ``encode_params_tuple``: the upstream C server's
        # ``tuple_decoder__next`` (``src/tuple.c`` lines 130-167)
        # treats ``DQLITE_UNIXTIME`` as ``default → DQLITE_PARSE``
        # for the inbound params decode. UNIXTIME is server-to-
        # client only — a peer (or python mock-server) producing
        # the tag on a params body must be rejected here just as
        # a real C gateway would. Without this guard,
        # mock-server harnesses silently accept malformed traffic
        # that regresses when migrated to a real cluster.
        if value_type == ValueType.UNIXTIME:
            raise DecodeError(
                f"UNIXTIME tag at param index {i}: server-to-client only; "
                "the upstream C gateway rejects DQLITE_UNIXTIME on inbound "
                "params with DQLITE_PARSE"
            )
        types.append(value_type)
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
        if int(t) not in _VALID_TYPE_CODES:
            raise EncodeError(f"Invalid type code {int(t)} at index {i}: not a valid ValueType")

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


def decode_row_header(
    data: bytes | memoryview, column_count: int
) -> tuple[list[ValueType] | RowMarker, int]:
    """Decode row column type header.

    Format: 4-bit type codes packed two per byte, padded to word boundary.
    Detects row markers (0xFF=done, 0xEE=part) by checking the first byte
    before validating header size, matching the Go reference implementation.
    Returns (types_or_marker, bytes_consumed).
    """
    # Check for markers first — markers are always exactly one 8-byte word
    # of a repeated sentinel byte, regardless of column count. Must check
    # before header size validation because for large column counts the
    # header would be >8 bytes.
    #
    # Upstream C uses the full uint64 sentinel (DQLITE_RESPONSE_ROWS_DONE
    # = 0xff..ff, _PART = 0xee..ee). Go's reference client checks only the
    # first byte; we validate all 8 bytes so torn/corrupt markers like
    # ``0xff 0x00..`` are rejected as malformed rather than silently
    # truncating the result stream. This is strictly tighter
    # than the Go behavior.
    if len(data) >= 8:
        marker = bytes(data[:8]) if isinstance(data, memoryview) else data[:8]
        if marker == _ROW_DONE_MARKER:
            return RowMarker.DONE, 8
        if marker == _ROW_PART_MARKER:
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
        except ValueError as e:
            raise DecodeError(f"Invalid value type code {nibble} at column index {i}") from e

    return types, header_size


def encode_row_values(values: Sequence[Any], types: Sequence[ValueType]) -> bytes:
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
    data: bytes | memoryview, types: Sequence[ValueType]
) -> tuple[list[Any], int]:
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
