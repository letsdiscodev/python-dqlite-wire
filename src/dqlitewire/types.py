"""Primitive type encoding/decoding for dqlite wire protocol.

All multi-byte integers are little-endian.
Text is null-terminated UTF-8, padded to 8-byte boundary.

The codec deals only in wire primitives (int, float, str, bool, bytes, None).
Higher-level conversions — like ``DQLITE_ISO8601`` → ``datetime.datetime`` or
``DQLITE_UNIXTIME`` → epoch-based ``datetime.datetime`` — belong in the
driver/DBAPI layer, matching the split used by the C reference client and
by Go's ``database/sql`` driver.
"""

import mmap
import struct

from dqlitewire.constants import WORD_SIZE, ValueType
from dqlitewire.exceptions import DecodeError, EncodeError

# Exact set of Python types ``encode_value`` accepts. Callers see a
# type-checker error if they pass something else, instead of a runtime
# EncodeError at the first wire round-trip. ``bytes``-like siblings
# (``bytearray``, ``memoryview``, ``mmap.mmap``) are accepted because
# the BLOB encoder normalises them through ``bytes(value)``; the
# inference path maps them to ValueType.BLOB for the same reason
# stdlib ``sqlite3`` does.
type WireInput = bool | int | float | str | bytes | bytearray | memoryview | mmap.mmap | None

# Exact set of Python types ``decode_value`` may return (first element
# of the ``(value, consumed)`` tuple). Narrower than ``Any`` — wire
# values are always one of these primitives, and the driver layer
# widens to ``Any`` only at the PEP 249 row-tuple boundary.
type WireValue = bool | int | float | str | bytes | None

# Per-BLOB byte cap. The overall frame-size cap in ``buffer.py`` (64 MiB)
# already bounds any single message, but a hostile or buggy peer can
# otherwise pack a single BLOB field that consumes the whole frame. The
# cap is a defensive ceiling — real applications do not send
# multi-megabyte blobs over the wire — and keeps the decoder fast-failing
# well before large allocations or arithmetic on attacker-controlled
# lengths. Sits beside ``_MAX_PARAM_COUNT`` / ``_MAX_COLUMN_COUNT`` /
# ``_MAX_FILE_COUNT`` / ``_MAX_NODE_COUNT`` in spirit.
_MAX_BLOB_SIZE = 16 * 1024 * 1024  # 16 MiB

# Per-TEXT cell byte cap. Symmetric with ``_MAX_BLOB_SIZE`` — a TEXT
# row cell (TEXT or ISO8601 wire type) is NUL-terminated UTF-8 and
# otherwise unbounded within the frame envelope. Matches the 16 MiB
# BLOB ceiling: real applications never send multi-megabyte string
# columns over the wire, and this defensive cap keeps ``decode_text``
# from scanning or allocating attacker-controlled lengths that
# exceed the BLOB ceiling.
_MAX_TEXT_VALUE_SIZE = 16 * 1024 * 1024  # 16 MiB

# Cap on the stringified representation of an out-of-range integer in
# EncodeError messages. A hostile or buggy caller passing ``10 ** 500``
# would otherwise bake a kilobyte of digits into the error text (and
# every log line / traceback that quotes it). Parity with
# ``_truncate_error`` in the client layer and ``_MAX_FAILURE_MESSAGE_SIZE``
# on the decode side.
_MAX_VALUE_REPR = 64


def _bounded_repr(value: int) -> str:
    s = str(value)
    if len(s) <= _MAX_VALUE_REPR:
        return s
    return f"{s[:_MAX_VALUE_REPR]}... [{len(s)} digits]"


def _validate_uint64(name: str, value: int) -> None:
    """Validate ``value`` is an int (not bool) in the uint64 range.

    Shared by ``encode_uint64`` and message dataclass ``__post_init__``
    sites so construction-time and encode-time checks raise the same
    exception class (``EncodeError``) and both reject ``bool`` (which
    ``isinstance(True, int)`` otherwise silently admits).
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise EncodeError(f"{name} must be int, got {type(value).__name__}")
    if not 0 <= value < 2**64:
        raise EncodeError(
            f"{name}={_bounded_repr(value)} out of range for uint64 (0 to {(1 << 64) - 1})"
        )


def _validate_uint32(name: str, value: int) -> None:
    """Validate ``value`` is an int (not bool) in the uint32 range.

    Symmetric with :func:`_validate_uint64`.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise EncodeError(f"{name} must be int, got {type(value).__name__}")
    if not 0 <= value < 2**32:
        raise EncodeError(
            f"{name}={_bounded_repr(value)} out of range for uint32 (0 to {(1 << 32) - 1})"
        )


def encode_uint64(value: int) -> bytes:
    """Encode an unsigned 64-bit integer (little-endian)."""
    _validate_uint64("value", value)
    return struct.pack("<Q", value)


def decode_uint64(data: bytes | memoryview) -> int:
    """Decode an unsigned 64-bit integer (little-endian).

    Accepts ``bytes`` or ``memoryview`` so hot-path body decoders
    can pass memoryview slices without copying.
    """
    if len(data) < 8:
        raise DecodeError(f"Need 8 bytes for uint64, got {len(data)}")
    result: int = struct.unpack("<Q", data[:8])[0]
    return result


def encode_int64(value: int) -> bytes:
    """Encode a signed 64-bit integer (little-endian).

    Rejects ``bool`` explicitly (``isinstance(True, int)`` is True
    and would silently encode as ``1``); symmetric with the
    ``encode_uint32`` / ``encode_uint64`` validators that reject
    bool too. A caller-side typo like ``encode_int64(True)`` is more
    likely a bug than a deliberate ``True → 1`` coercion; the
    rejection surfaces it.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise EncodeError(f"value must be int, got {type(value).__name__}")
    if not -(2**63) <= value < 2**63:
        raise EncodeError(f"Value {_bounded_repr(value)} out of range for int64")
    return struct.pack("<q", value)


def decode_int64(data: bytes | memoryview) -> int:
    """Decode a signed 64-bit integer (little-endian).

    Accepts ``bytes`` or ``memoryview``.
    """
    if len(data) < 8:
        raise DecodeError(f"Need 8 bytes for int64, got {len(data)}")
    result: int = struct.unpack("<q", data[:8])[0]
    return result


def encode_uint32(value: int) -> bytes:
    """Encode an unsigned 32-bit integer (little-endian)."""
    _validate_uint32("value", value)
    return struct.pack("<I", value)


def decode_uint32(data: bytes | memoryview) -> int:
    """Decode an unsigned 32-bit integer (little-endian).

    Accepts ``bytes`` or ``memoryview``.
    """
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


def decode_double(data: bytes | memoryview) -> float:
    """Decode a 64-bit floating point number (little-endian).

    All IEEE 754 values are accepted, including NaN and infinity,
    matching the Go reference implementation behavior. Accepts
    ``bytes`` or ``memoryview``.
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


def encode_text(value: str, *, max_size: int | None = None, label: str = "Text") -> bytes:
    """Encode text as null-terminated UTF-8, padded to 8-byte boundary.

    Wire-protocol limitation: TEXT is NUL-terminated, so the encoder
    rejects values with embedded NULs — they cannot round-trip through
    the wire. Callers that need to transmit arbitrary bytes (including
    NULs) must use the BLOB type instead. SQLite itself allows TEXT
    columns to contain embedded NULs on disk, but such values cannot be
    reliably read back through the dqlite wire protocol (``decode_text``
    stops at the first NUL). This matches Go / C client behaviour;
    it is a protocol-level asymmetry, not a Python-side design choice.

    ``max_size`` (UTF-8 byte cap, NOT codepoints) caps the encoded
    payload. The decoder caps on bytes too (``decode_text`` uses
    ``max_size`` against ``null_pos`` which is a byte offset), so
    matching units here means encode → decode round-trip is symmetric:
    a string the encoder accepts, the decoder also accepts. Caller-side
    inline ``len(s)`` (codepoints) checks would admit non-ASCII payloads
    near the boundary that the decoder then rejects, breaking the
    round-trip. ``label`` is forwarded into the ``EncodeError`` message
    so a caller passing ``label="leader address"`` gets ``"leader
    address length N exceeds maximum (M)"`` rather than the generic
    ``"Text length..."``.
    """
    if not isinstance(value, str):
        raise EncodeError(f"encode_text expected str, got {type(value).__name__}")
    try:
        utf8 = value.encode("utf-8")
    except UnicodeEncodeError as e:
        raise EncodeError(f"Text contains invalid UTF-8: {e}") from e
    if max_size is not None and len(utf8) > max_size:
        raise EncodeError(f"{label} length {len(utf8)} exceeds maximum ({max_size})")
    nul_byte_offset = utf8.find(b"\x00")
    if nul_byte_offset != -1:
        # Report the byte offset of the embedded NUL rather than the
        # Python-string character index — the encoder produces bytes so
        # operators debugging a wire capture expect byte offsets.
        raise EncodeError(
            f"Text value contains embedded null byte at byte offset "
            f"{nul_byte_offset}; null-terminated encoding would lose data"
        )
    encoded = utf8 + b"\x00"
    padding = pad_to_word(len(encoded))
    return encoded + (b"\x00" * padding)


# Threshold below which we materialize a memoryview to bytes in one
# shot (one allocation + one ``bytes.find``) instead of the chunked
# scan. Row text payloads are almost always well under 64 KiB, so the
# one-shot path dominates the common case. Above the
# threshold we fall back to chunked scanning to bound peak memory for
# pathologically long texts.
_TEXT_ONE_SHOT_MAX = 65_536
_TEXT_SCAN_CHUNK = 4096


def decode_text(
    data: bytes | memoryview,
    *,
    max_size: int = _MAX_TEXT_VALUE_SIZE,
    label: str = "Text",
) -> tuple[str, int]:
    """Decode null-terminated UTF-8 text.

    Accepts either ``bytes`` or ``memoryview``. Returns the decoded
    string and the number of bytes consumed (including padding).

    ``max_size`` caps the decoded length (excluding the terminator) —
    defaults to ``_MAX_TEXT_VALUE_SIZE``. Callers that enforce a
    smaller cap (e.g. ``decode_column_name`` at 4 KiB) pass their own
    ceiling; callers that legitimately need the full row-cell cap use
    the default. ``label`` is used in the ``DecodeError`` message when
    ``max_size`` is exceeded, so a caller passing
    ``label="leader address"`` gets ``"leader address length N exceeds
    maximum (M)"`` rather than the generic ``"Text length..."``.

    Wire-protocol limitation: TEXT is NUL-terminated, so a value with
    embedded NULs on the server side (e.g. written by a non-dqlite
    SQLite client that accepts TEXT with NULs) reads back **truncated
    at the first NUL**. No diagnostic is raised — the protocol provides
    no length prefix, so the decoder cannot tell an embedded NUL apart
    from the terminator. The encoder refuses such values outright;
    callers who need round-trip-safe binary data should use BLOB.

    The decoder's hot body loops (RowsResponse, FilesResponse,
    ServersResponse) wrap the body in a ``memoryview`` so
    per-iteration slices are O(1) rather than O(remaining).
    ``bytes`` inputs use zero-copy ``.index(b"\\x00")``.

    ``memoryview`` inputs use a single ``bytes(mv).find(b"\\x00")``
    when the remaining buffer is small (< 64 KiB). This is one
    allocation and one C-level scan, matching the hot-path cost of the
    ``bytes`` branch. For larger buffers we fall back to a chunked
    scan so peak memory stays bounded.
    """
    if isinstance(data, memoryview):
        data_len = len(data)
        if data_len <= _TEXT_ONE_SHOT_MAX:
            # One-shot path: single materialization + C-level find.
            # Bound the materialization to ``max_size + 1`` bytes so a
            # caller-supplied small cap (e.g. 4 KiB column-name) does
            # not allocate a 16 MiB scratch buffer just to scan a few
            # bytes for the NUL. Keeps the cap-before-allocate
            # invariant symmetric with ``decode_blob`` (which checks
            # the length prefix BEFORE materialising the payload).
            scan_window = min(data_len, max_size + 1)
            materialized = bytes(data[:scan_window])
            null_pos = materialized.find(b"\x00")
            if null_pos < 0:
                if scan_window < data_len:
                    # The remaining bytes (past the cap) cannot legally
                    # contain the terminator since `null_pos > max_size`
                    # would fail the cap check below anyway. Surface
                    # the cap-exceeded error directly.
                    raise DecodeError(f"{label} length exceeds maximum ({max_size})")
                raise DecodeError(f"{label} not null-terminated")
            try:
                text = materialized[:null_pos].decode("utf-8")
            except UnicodeDecodeError as e:
                raise DecodeError(f"Invalid UTF-8 in {label}: {e}") from e
        else:
            # Chunked fallback for pathologically long text payloads.
            # Bound the total scan window to ``max_size + 1`` bytes so
            # a small caller-supplied cap (e.g. 4 KiB column-name) on
            # a 16 MiB memoryview does not allocate megabytes of
            # chunks before the cap-after-scan check fires. Symmetric
            # with the one-shot path's ``min(data_len, max_size+1)``
            # window.
            scan_limit = min(data_len, max_size + 1)
            chunks: list[bytes] = []
            scanned = 0
            null_pos = -1
            while scanned < scan_limit:
                chunk_end = min(scanned + _TEXT_SCAN_CHUNK, scan_limit)
                chunk = bytes(data[scanned:chunk_end])
                local = chunk.find(b"\x00")
                if local >= 0:
                    chunks.append(chunk[:local])
                    null_pos = scanned + local
                    break
                chunks.append(chunk)
                scanned = chunk_end
            if null_pos < 0:
                if scan_limit < data_len:
                    raise DecodeError(f"{label} length exceeds maximum ({max_size})")
                raise DecodeError(f"{label} not null-terminated")
            try:
                text = b"".join(chunks).decode("utf-8")
            except UnicodeDecodeError as e:
                raise DecodeError(f"Invalid UTF-8 in {label}: {e}") from e
    else:
        try:
            null_pos = data.index(b"\x00")
        except ValueError as e:
            raise DecodeError(f"{label} not null-terminated") from e
        # Cap on bytes BEFORE the UTF-8 decode allocation. Without
        # this, a 100 MiB hostile-peer payload that happens to be
        # NUL-terminated near the end is materialised in full just to
        # be rejected. Symmetric with the memoryview branch's
        # ``scan_window = min(data_len, max_size+1)``.
        if null_pos > max_size:
            raise DecodeError(f"{label} length {null_pos} exceeds maximum ({max_size})")
        try:
            text = data[:null_pos].decode("utf-8")
        except UnicodeDecodeError as e:
            raise DecodeError(f"Invalid UTF-8 in {label}: {e}") from e

    if null_pos > max_size:
        raise DecodeError(f"{label} length {null_pos} exceeds maximum ({max_size})")
    # Calculate total size including padding
    total_size = null_pos + 1 + pad_to_word(null_pos + 1)
    if len(data) < total_size:
        raise DecodeError(f"Not enough data for text padding: need {total_size}, got {len(data)}")
    return text, total_size


def encode_blob(value: bytes, *, max_blob_size: int = _MAX_BLOB_SIZE) -> bytes:
    """Encode a blob (length-prefixed binary data, padded to 8-byte boundary).

    Format: uint64 length + data + padding

    ``max_blob_size`` is the per-blob cap. Defaults to
    ``_MAX_BLOB_SIZE`` (16 MiB) — a defense-in-depth ceiling matching
    the decode-side cap. Callers handling larger BLOBs (per their
    deployment's actual ``raft.log.entry_size_max``) can raise this
    explicitly. The outer ``max_message_size`` cap on the codec
    always wins regardless.
    """
    length = len(value)
    if length > max_blob_size:
        raise EncodeError(f"Blob length {length} exceeds maximum ({max_blob_size})")
    padding = pad_to_word(length)
    return encode_uint64(length) + value + (b"\x00" * padding)


def decode_blob(
    data: bytes | memoryview, *, max_blob_size: int = _MAX_BLOB_SIZE
) -> tuple[bytes, int]:
    """Decode a blob.

    Accepts either ``bytes`` or ``memoryview``. Returns the blob data
    (always as ``bytes``) and the number of bytes consumed.

    ``max_blob_size`` mirrors the ``encode_blob`` parameter — see that
    docstring for the rationale.
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


def encode_value(value: WireInput, value_type: ValueType | None = None) -> tuple[bytes, ValueType]:
    """Encode a Python value to wire format.

    If value_type is not provided, it's inferred from the Python type.
    Returns (encoded_data, value_type).

    Inference: Python ``bool`` infers to ``ValueType.BOOLEAN``, NOT
    ``ValueType.INTEGER``. On the wire both encode as ``uint64(0)`` or
    ``uint64(1)``; the difference is only in the type tag. SQLite
    stores both as an integer column value regardless of that tag, so
    a round-trip through a column declared INTEGER returns a Python
    ``int`` (``1`` or ``0``), never a Python ``bool``. Callers who
    need ``True`` back must either coerce at read time
    (``bool(row[0])``) or pass ``column_types=[...BOOLEAN...]`` on the
    response side so the driver's typed decode materialises bool.
    Inference matches the Go / C clients' DQLITE_BOOLEAN handling.
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
        elif isinstance(value, str):
            value_type = ValueType.TEXT
        elif isinstance(value, (bytes, bytearray, memoryview, mmap.mmap)):
            # Parity with the explicit BLOB branch and with stdlib
            # ``sqlite3``: bytes-like types (incl. memoryview and
            # memory-mapped regions) infer to BLOB. Callers building
            # payloads via mutation (bytearray), zero-copy slicing
            # (memoryview), or mmap-backed I/O no longer need to wrap
            # values in ``bytes(...)`` before passing them here.
            # ``array.array`` is intentionally NOT accepted: typed
            # numeric arrays have platform-dependent bytes
            # representation, so silent acceptance would yield
            # non-portable BLOBs.
            value_type = ValueType.BLOB
        else:
            raise EncodeError(
                f"Cannot infer wire type for value of type {type(value).__name__!r}. "
                f"The wire codec only accepts bool, int, float, str, bytes-like, "
                f"or None. Callers passing datetime/date/etc. must convert to str "
                f"(for ISO8601) or int (for UNIXTIME) at the driver layer."
            )

    if value_type == ValueType.BOOLEAN:
        # Accept bool directly; allow the exact ints 0 and 1 as a
        # pragmatic escape for callers working with raw column values.
        # Reject arbitrary ints — the previous ``1 if value else 0``
        # coercion silently mapped values like ``5`` or ``-1`` to True,
        # which round-trips as the bool True and loses the caller's
        # original value.
        if isinstance(value, bool):
            return encode_uint64(1 if value else 0), value_type
        if isinstance(value, int) and value in (0, 1):
            return encode_uint64(value), value_type
        raise EncodeError(
            f"BOOLEAN requires bool (or exactly 0/1), got {type(value).__name__}={value!r}"
        )
    elif value_type in (ValueType.INTEGER, ValueType.UNIXTIME):
        # Note: UNIXTIME is a server-to-client-only type (the C server's
        # tuple_decoder has no inbound case for DQLITE_UNIXTIME). Explicit
        # UNIXTIME encoding here is supported for roundtrip tests that
        # simulate server-to-client frames; encode_params_tuple, which is
        # the outgoing-params path, uses inference and never picks
        # UNIXTIME, so the server-rejection case cannot arise via the
        # documented client API.
        #
        # Reject bool under explicit non-BOOLEAN types for symmetry with
        # the FLOAT branch. The default-inference path (no explicit
        # value_type) still picks BOOLEAN for bools, so a caller who
        # wants a bool encoded as an integer must coerce explicitly via
        # ``int(x)``. This prevents the silent "True in an INTEGER
        # column decodes as 1 (int), not True (bool)" surprise.
        if isinstance(value, bool):
            raise EncodeError(
                f"Cannot encode bool as {value_type.name}; cast with int(x) "
                "explicitly if integer semantics are intended."
            )
        if not isinstance(value, int):
            raise EncodeError(f"Expected int for {value_type.name}, got {type(value).__name__}")
        return encode_int64(value), value_type
    elif value_type == ValueType.FLOAT:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EncodeError(f"Expected int or float for FLOAT, got {type(value).__name__}")
        return encode_double(float(value)), value_type
    elif value_type in (ValueType.TEXT, ValueType.ISO8601):
        if not isinstance(value, str):
            raise EncodeError(f"Expected str for {value_type.name}, got {type(value).__name__}")
        return encode_text(value), value_type
    elif value_type == ValueType.BLOB:
        if not isinstance(value, (bytes, bytearray, memoryview, mmap.mmap)):
            raise EncodeError(f"Expected bytes for BLOB, got {type(value).__name__}")
        # Cap the size BEFORE materialising via ``bytes(value)`` —
        # symmetric with ``decode_blob``'s cap-before-allocate
        # ordering. Without this, a 100 MB ``bytearray`` is copied
        # to ``bytes`` first, then rejected by ``encode_blob``'s
        # length check; the materialise allocation is wasted.
        # ``len()`` works on every supported open container without
        # materialising. A closed ``mmap.mmap`` raises
        # ``ValueError("mmap closed or invalid")`` from ``len()``;
        # a released ``memoryview`` raises ``ValueError`` likewise.
        # Fall through to the materialise step in that case so the
        # original ``EncodeError`` shape (with the same message
        # prefix) surfaces — preserving backwards compatibility for
        # callers catching on the existing message text.
        try:
            length: int | None = len(value)
        except (ValueError, BufferError, TypeError):
            length = None
        if length is not None and length > _MAX_BLOB_SIZE:
            raise EncodeError(f"Blob length {length} exceeds maximum ({_MAX_BLOB_SIZE})")
        # Materialise via ``bytes(value)``. For ``bytes`` / ``bytearray``
        # / open ``memoryview`` this cannot fail; for a closed
        # ``mmap.mmap`` it raises ``ValueError("mmap closed or invalid")``,
        # for a released ``memoryview`` it raises ``BufferError``. Wrap
        # both as ``EncodeError`` to maintain the wire-layer convention
        # that all encode failures surface as ``EncodeError``; callers
        # using ``except EncodeError`` would otherwise silently miss the
        # failure.
        try:
            materialized = bytes(value)
        except (ValueError, BufferError) as e:
            raise EncodeError(f"Cannot materialise BLOB from {type(value).__name__}: {e}") from e
        return encode_blob(materialized), value_type
    elif value_type == ValueType.NULL:
        # None is already handled in the early-return branch above, so reaching
        # here means value is not None with explicit NULL type — always a bug.
        raise EncodeError(
            f"Cannot encode non-None value {value!r} as NULL. "
            f"Pass value=None or use the appropriate ValueType."
        )
    else:
        raise EncodeError(f"Unknown value type: {value_type}")


def decode_value(data: bytes | memoryview, value_type: ValueType) -> tuple[WireValue, int]:
    """Decode a value from wire format.

    Returns (value, bytes_consumed).
    """
    if value_type == ValueType.BOOLEAN:
        # Symmetric with encode_value: BOOLEAN must be exactly 0 or 1 on
        # the wire. Silently coercing any uint64 to True/False would make
        # round-trips lossy (encoding restores 1/0) and hide malformed
        # frames produced by a broken or hostile peer.
        raw = decode_uint64(data)
        if raw not in (0, 1):
            raise DecodeError(f"BOOLEAN wire value must be 0 or 1, got {raw}")
        return bool(raw), 8
    elif value_type == ValueType.INTEGER:
        return decode_int64(data), 8
    elif value_type == ValueType.UNIXTIME:
        # Return raw int64 to preserve round-trip identity at the wire level.
        # Higher-level clients (like the dqlite DBAPI) turn this into a
        # datetime, matching what Go's Rows.Next() does in the database/sql
        # driver layer.
        return decode_int64(data), 8
    elif value_type == ValueType.FLOAT:
        return decode_double(data), 8
    elif value_type in (ValueType.TEXT, ValueType.ISO8601):
        # ISO8601 is treated as text at the wire level — the C reference
        # uses text__encode / text__decode for DQLITE_ISO8601 (see dqlite
        # src/tuple.c) and Go returns the raw string from the codec.
        # Parsing to datetime belongs in the driver/DBAPI layer.
        return decode_text(data)
    elif value_type == ValueType.BLOB:
        return decode_blob(data)
    elif value_type == ValueType.NULL:
        if len(data) < 8:
            raise DecodeError(f"Need 8 bytes for NULL value, got {len(data)}")
        # Strict reserved-field hygiene: ``encode_null`` writes exactly
        # ``b"\x00" * 8``; a non-zero NULL payload is either bug or
        # malice. Symmetric with the BOOLEAN strict-decode precedent
        # and the reserved-field policy pinned across Header /
        # DbResponse / LeaderRequest decoders.
        if bytes(data[:8]) != b"\x00" * 8:
            raise DecodeError(f"NULL payload must be 8 zero bytes, got {bytes(data[:8])!r}")
        return None, 8
    else:
        raise DecodeError(f"Unknown value type: {value_type}")
