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

# Per-BLOB byte cap. Sits 64 bytes below ``DEFAULT_MAX_MESSAGE_SIZE``
# (64 MiB) so a blob at the documented cap round-trips through the
# default decoder. The previous "raise to 64 MiB exact" fix narrowed
# the legacy 48 MiB asymmetry but left a residual: ``encode_blob`` of
# a 64 MiB payload returns ``8 (length prefix) + 64 MiB = 64 MiB + 8``
# bytes, and a single-row ``RowsResponse`` carrying that blob produces
# ~64 MiB + 48 bytes (8 message header + 8 col_count + 8 col_name +
# 8 row_header + 8 length prefix + 8 row marker). The same-default
# decoder then rejected the bytes its own encoder produced.
#
# Worst-case in-row framing breakdown:
#   8 message header + 8 col_count + 8 col_name + 8 row_header +
#   8 length prefix + 8 row marker = 48 bytes; round up to 64 for
#   schema/alignment slop and breathing room.
#
# Per-call ``max_blob_size=`` on ``encode_blob`` / ``decode_blob``
# still allows callers to override this default at the boundary.
_MAX_BLOB_SIZE: Final[int] = 64 * 1024 * 1024 - 64  # 64 MiB minus framing overhead

# Per-TEXT cell byte cap. Symmetric with ``_MAX_BLOB_SIZE`` — a TEXT
# row cell (TEXT or ISO8601 wire type) is NUL-terminated UTF-8 and
# otherwise unbounded within the frame envelope. Matches the BLOB
# ceiling: real applications never send multi-megabyte string columns
# over the wire, and this defensive cap keeps ``decode_text`` from
# scanning or allocating attacker-controlled lengths that would land
# the encoded message outside the default frame envelope.
_MAX_TEXT_VALUE_SIZE: Final[int] = 64 * 1024 * 1024 - 64  # 64 MiB minus framing overhead

# Cap on the stringified representation of an out-of-range integer in
# EncodeError messages. A hostile or buggy caller passing ``10 ** 500``
# would otherwise bake a kilobyte of digits into the error text (and
# every log line / traceback that quotes it). Parity with
# ``_truncate_error`` in the client layer and ``_MAX_FAILURE_MESSAGE_SIZE``
# on the decode side.
_MAX_VALUE_REPR: Final[int] = 64


def _bounded_repr(value: int) -> str:
    s = str(value)
    if len(s) <= _MAX_VALUE_REPR:
        return s
    return f"{s[:_MAX_VALUE_REPR]}... [{len(s)} digits]"


def _bounded_repr_any(value: object) -> str:
    """``repr(value)`` capped at ``_MAX_VALUE_REPR`` chars.

    Sibling of :func:`_bounded_repr` (which is int-specific and emits
    raw digits without quotes). Apply at every encode-error site that
    quotes caller-controlled data via ``{value!r}`` so the documented
    discipline (``_MAX_VALUE_REPR`` rationale above) holds uniformly
    across all rejection branches — not just the int-validator
    helpers.
    """
    s = repr(value)
    if len(s) <= _MAX_VALUE_REPR:
        return s
    return f"{s[:_MAX_VALUE_REPR]}... [{len(s)} chars]"


# Single-byte buffer-protocol formats. ``memoryview`` accepts these as
# documented zero-copy BLOB binding shapes (``memoryview(b'...')`` is
# 'B', ``memoryview(bytearray(...))`` is 'B', and ``array.array('b'/
# 'B'/'c', ...)`` matches the typed-but-still-byte-wide case). Any
# other format is a typed numeric memoryview whose bytes representation
# is host-endian and host-int-width, which is exactly the cross-
# platform-corruption hazard the bare ``array.array`` rejection
# prevents — silently accepting it via ``memoryview(array.array('i',
# ...))`` would re-open that hole through the buffer-protocol back
# door.
_BYTE_FORMAT_MEMORYVIEW: Final[frozenset[str]] = frozenset({"B", "b", "c"})


def _reject_non_byte_format_memoryview(value: memoryview) -> None:
    """Reject ``memoryview`` whose buffer-protocol ``format`` is not a
    single-byte type. See ``_BYTE_FORMAT_MEMORYVIEW``.

    Shared by the inference and explicit-BLOB branches of
    :func:`encode_value` so both entry points apply the same
    rejection (parity with the ``mmap.mmap`` acceptance pins in
    ``tests/test_blob_inference_mmap.py``).

    A released memoryview raises ``ValueError`` from ``.format``;
    fall through silently in that case so the downstream
    ``bytes(value)`` materialise step surfaces the canonical
    ``EncodeError("Cannot materialise BLOB...")``. The format check
    is a portability guard, not the released-buffer guard.
    """
    try:
        fmt = value.format
    except ValueError:
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


def _validate_uint64(name: str, value: int) -> None:
    """Validate ``value`` is an int (not bool) in the uint64 range.

    Shared by ``encode_uint64`` and message dataclass ``__post_init__``
    sites so construction-time and encode-time checks raise the same
    exception class (``EncodeError``) and both reject ``bool`` (which
    ``isinstance(True, int)`` otherwise silently admits).

    The predicate orders ``isinstance(value, bool)`` first so the three
    sibling validators (``_validate_uint32`` / ``_validate_int64``) and
    the package's other type-narrowing guards (``Header.__post_init__``,
    every dataclass ``__post_init__``) all read the same shape. The two
    clauses commute on every input, but a single canonical form keeps a
    future maintainer's grep across the family productive.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise EncodeError(f"{name} must be int, got {type(value).__name__}")
    if not 0 <= value < 2**64:
        raise EncodeError(
            f"{name}={_bounded_repr(value)} out of range for uint64 (0 to {(1 << 64) - 1})"
        )


def _validate_uint32(name: str, value: int) -> None:
    """Validate ``value`` is an int (not bool) in the uint32 range.

    Symmetric with :func:`_validate_uint64` — see that helper for the
    rationale on the bool-first predicate order.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise EncodeError(f"{name} must be int, got {type(value).__name__}")
    if not 0 <= value < 2**32:
        raise EncodeError(
            f"{name}={_bounded_repr(value)} out of range for uint32 (0 to {(1 << 32) - 1})"
        )


def _validate_int64(name: str, value: int) -> None:
    """Validate ``value`` is an int (not bool) in the signed int64 range.

    Symmetric with :func:`_validate_uint64` / :func:`_validate_uint32`.
    The existing exception wording (``"Value <repr> out of range for
    int64"``) is preserved so callers and test pins that match that
    text continue to pass — this helper is a refactor of the inline
    check inside :func:`encode_int64`, not a wording change.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise EncodeError(f"{name} must be int, got {type(value).__name__}")
    if not -(2**63) <= value < 2**63:
        raise EncodeError(f"Value {_bounded_repr(value)} out of range for int64")


def encode_uint64(value: int) -> bytes:
    """Encode an unsigned 64-bit integer (little-endian)."""
    _validate_uint64("value", value)
    return struct.pack("<Q", value)


def decode_uint64(data: bytes | memoryview, *, label: str = "uint64") -> int:
    """Decode an unsigned 64-bit integer (little-endian).

    Accepts ``bytes`` or ``memoryview`` so hot-path body decoders
    can pass memoryview slices without copying.

    ``label`` is interpolated into the truncation diagnostic so a
    caller decoding a specific field (e.g. ``OpenRequest.flags``)
    can surface a per-field error message — symmetric with
    :func:`decode_text`'s ``label=`` discipline. The default
    ``"uint64"`` preserves the historical message wording for callers
    that omit the kwarg.
    """
    if len(data) < 8:
        raise DecodeError(f"Need 8 bytes for {label}, got {len(data)}")
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
    _validate_int64("value", value)
    return struct.pack("<q", value)


def decode_int64(data: bytes | memoryview, *, label: str = "int64") -> int:
    """Decode a signed 64-bit integer (little-endian).

    Accepts ``bytes`` or ``memoryview``.

    ``label`` is interpolated into the truncation diagnostic so a
    caller decoding a specific field (e.g. ``"UNIXTIME cell"``) can
    surface a per-field error message — symmetric with the
    :func:`decode_text` / :func:`decode_uint64` label discipline.
    """
    if len(data) < 8:
        raise DecodeError(f"Need 8 bytes for {label}, got {len(data)}")
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

    All IEEE 754 ``float`` values are accepted, including NaN and
    infinity, matching the Go reference implementation behavior.

    Non-``float`` inputs are rejected:

    - ``bool`` is rejected explicitly because ``isinstance(True, int)``
      is True and ``float(True) == 1.0`` — without the guard a caller
      passing ``True`` would silently encode ``1.0`` onto the wire.
    - Plain ``int`` is rejected: ``struct.pack("<d", 42)`` succeeds via
      C-level coercion but the same coercion silently loses bits for
      ``|x| >= 2**53`` and raises ``OverflowError`` (outside the
      ``EncodeError`` family) for ``|x| >= 2**1024``. Callers that
      genuinely want int → FLOAT should call ``float(x)`` themselves
      and accept the precision boundary explicitly.
    - Third-party bool/numeric proxies (``numpy.bool_``,
      ``numpy.float64``, ``Decimal``, ``Fraction``) are all rejected
      via the float-subclass check. ``numpy.bool_`` in particular is
      not a Python ``bool`` subclass, so the bool guard alone leaves
      it slip through to ``struct.pack``'s ``__float__`` coercion —
      the exact silent-coerce footgun this function exists to prevent.
      Callers passing third-party numerics must coerce explicitly
      (``float(np_value)``).

    Mirrors the discipline already applied at ``_validate_uint64``,
    ``encode_int64``, and the ``encode_value`` FLOAT arm.
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
    """Decode a 64-bit floating point number (little-endian).

    All IEEE 754 values are accepted, including NaN and infinity,
    matching the Go reference implementation behavior. Accepts
    ``bytes`` or ``memoryview``.

    ``label`` is interpolated into the truncation diagnostic so a
    caller decoding a specific field can surface a per-field error
    message — symmetric with the :func:`decode_text` /
    :func:`decode_uint64` / :func:`decode_int64` label discipline.
    """
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

    One-way pair with permissive decode: this encoder uses strict
    UTF-8 with no symmetric ``errors=`` kwarg. A str obtained from
    :func:`decode_text` with ``errors="surrogateescape"`` (which
    deliberately smuggles raw bytes as ``U+DC80``-``U+DCFF``
    surrogates for round-trip recovery in stdlib idiom) CANNOT be
    re-encoded here — the strict UTF-8 encoder rejects the
    surrogates and surfaces ``EncodeError`` with the "lone surrogate
    or surrogate-escape character" diagnostic. dqlite's wire spec is
    UTF-8 by design; the asymmetry is deliberate. Callers needing
    round-trip-safe carriage of opaque bytes must use BLOB.
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
_TEXT_ONE_SHOT_MAX: Final[int] = 65_536
_TEXT_SCAN_CHUNK: Final[int] = 4096


def decode_text(
    data: bytes | memoryview,
    *,
    max_size: int = _MAX_TEXT_VALUE_SIZE,
    label: str = "Text",
    errors: str = "strict",
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

    ``errors`` controls UTF-8 decode error handling — passes through
    to ``bytes.decode``. The default ``"strict"`` matches the dqlite
    contract (UTF-8 by spec). Legacy / non-UTF-8 data (e.g. a SQLite
    database from a pre-3.x install storing TEXT in latin-1) can use
    ``"replace"`` or ``"backslashreplace"`` to coerce. C / Go clients
    pass non-UTF-8 bytes through without validation; the strict
    default is a Python-side defensive narrowing, not a wire-spec
    requirement.

    WARNING: ``errors="replace"`` produces ``U+FFFD`` (REPLACEMENT
    CHARACTER) and ``errors="surrogateescape"`` produces the
    ``U+DC80``-``U+DCFF`` surrogate range; neither is covered by the
    ``_CONTROL_CHARS_RE`` character class consulted by
    :func:`dqlitewire.messages.responses.sanitize_for_log` /
    :func:`dqlitewire.messages.responses.sanitize_server_text`, so a
    log-side scrub does NOT replace them. ``surrogateescape`` is also
    not round-trippable via :func:`encode_text` (the surrogates fail
    re-encode) and breaks naive ``json.dumps``. Callers using a
    non-strict mode AND forwarding the result to a log sink should
    apply their own additional sanitisation (e.g.
    ``s.encode("ascii", errors="replace").decode("ascii")``) or route
    the value through a structured-log encoder that handles the
    residual codepoints explicitly.

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
                text = materialized[:null_pos].decode("utf-8", errors=errors)
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
                text = b"".join(chunks).decode("utf-8", errors=errors)
            except UnicodeDecodeError as e:
                raise DecodeError(f"Invalid UTF-8 in {label}: {e}") from e
    else:
        # Bound the NUL scan at ``max_size + 1`` bytes — a hostile peer
        # supplying a 100 MiB payload with the terminator (or no
        # terminator at all) past the cap previously caused
        # ``data.index`` to ``memchr``-scan the full buffer before
        # the cap check fired. ``bytes.find`` is bounded; if the NUL
        # is not within the window, ``find`` returns -1. Symmetric
        # with the memoryview branch's
        # ``scan_window = min(data_len, max_size+1)`` cap-before-scan.
        scan_end = min(len(data), max_size + 1)
        null_pos = data.find(b"\x00", 0, scan_end)
        if null_pos < 0:
            if scan_end < len(data):
                raise DecodeError(f"{label} length exceeds maximum ({max_size})")
            raise DecodeError(f"{label} not null-terminated")
        # ``bytes.find(needle, 0, scan_end)`` returns ``-1`` or a
        # value strictly less than ``scan_end = min(len(data),
        # max_size + 1)``, so ``null_pos <= max_size`` by
        # construction. The cap is enforced by the ``scan_end``
        # bound; no post-find check is needed.
        try:
            text = data[:null_pos].decode("utf-8", errors=errors)
        except UnicodeDecodeError as e:
            raise DecodeError(f"Invalid UTF-8 in {label}: {e}") from e

    # Each branch above is solely responsible for cap-exceeded rejection
    # (lines 317, 347, 366, 372). Every branch caps the scan window to
    # ``min(data_len, max_size + 1)`` before the find, so by construction
    # ``null_pos <= max_size`` at this convergence point — a tail check
    # here would be unreachable.
    total_size = null_pos + 1 + pad_to_word(null_pos + 1)
    if len(data) < total_size:
        raise DecodeError(f"Not enough data for text padding: need {total_size}, got {len(data)}")
    return text, total_size


def encode_blob(
    value: bytes | bytearray | memoryview, *, max_blob_size: int = _MAX_BLOB_SIZE
) -> bytes:
    """Encode a blob (length-prefixed binary data, padded to 8-byte boundary).

    Format: uint64 length + data + padding

    Accepts any bytes-like input (``bytes``, ``bytearray``, ``memoryview``);
    the wire payload is materialised once before concatenation.

    ``max_blob_size`` is the per-blob cap. Defaults to
    ``_MAX_BLOB_SIZE`` (``64 MiB - 64 B`` = 67_108_800 bytes — sits
    64 bytes below ``DEFAULT_MAX_MESSAGE_SIZE`` to leave room for the
    length prefix plus row framing so a blob at the documented cap
    fits inside the default 64 MiB message envelope; see the
    ``_MAX_BLOB_SIZE`` definition for the full per-row arithmetic).
    Callers handling larger BLOBs (per their deployment's actual
    ``raft.log.entry_size_max``) can raise this explicitly. The outer
    ``max_message_size`` cap on the codec always wins regardless.
    """
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise EncodeError(
            f"Blob value must be bytes, bytearray, or memoryview; got {type(value).__name__}"
        )
    length = len(value)
    if length > max_blob_size:
        raise EncodeError(f"Blob length {length} exceeds maximum ({max_blob_size})")
    payload = bytes(value)
    padding = pad_to_word(length)
    return encode_uint64(length) + payload + (b"\x00" * padding)


def decode_blob(
    data: bytes | memoryview, *, max_blob_size: int = _MAX_BLOB_SIZE
) -> tuple[bytes, int]:
    """Decode a blob.

    Accepts either ``bytes`` or ``memoryview``. Returns the blob data
    (always as ``bytes``) and the number of bytes consumed.

    **Round-trip type asymmetry.** Callers that bind a ``memoryview``
    or ``bytearray`` on the encode side see ``bytes`` on the decode
    side — ``decode_blob`` always materialises an owned ``bytes``
    copy regardless of the input shape that was originally encoded.
    This matches stdlib ``sqlite3`` semantics (``sqlite3.Binary =
    memoryview`` on bind; ``bytes`` on fetch) and is the right owned-
    copy shape across the deserialisation boundary. Callers needing
    a ``memoryview`` view of the result must wrap explicitly:
    ``memoryview(decoded)``.

    ``max_blob_size`` mirrors the ``encode_blob`` parameter — defaults
    to ``_MAX_BLOB_SIZE`` (``64 MiB - 64 B`` = 67_108_800 bytes; the
    64-byte shortfall reserves room for length prefix + row framing
    under the default 64 MiB message envelope). See ``encode_blob``
    for the rationale.
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
    """Infer the wire ``ValueType`` from a Python value without running
    the encoder.

    Mirrors ``encode_value``'s leading isinstance ladder, returning the
    tag only — no length-cap check, no BLOB materialisation, no UTF-8
    encode. The helper lets the row-header build path in
    ``RowsResponse._get_row_types`` get a per-row type tuple at O(1)
    per cell instead of running the full ``encode_value`` pipeline and
    discarding the bytes (a 2x encode cost on the inference fallback
    path, observed as ``2 * N * M`` ``encode_value`` calls for an
    ``N`` row by ``M`` column inference frame).

    The ``memoryview`` over a non-byte format is rejected here too so
    the inference path stays consistent with the explicit-BLOB branch:
    callers passing ``memoryview(array.array('i', ...))`` see the same
    rejection regardless of whether the inference ladder or the BLOB
    arm fires first. Oversize BLOB / TEXT values are NOT rejected by
    this helper — that cap discipline still lives inside
    ``encode_value`` and fires on the subsequent emission pass. The
    behaviour is "fail at SOME point" not "fail at the inference pass";
    the diagnostic now names the emission cap, not the inference.
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
        # The format-reject check used to live here, but is now hoisted
        # to the top of ``encode_value`` so it fires exactly once per
        # call regardless of inferred-vs-explicit path. Inferred BLOB
        # callers used to pay the cost twice (here + in the explicit
        # BLOB arm); the hoist restores O(1)-per-cell on the row-encode
        # fallback path the helper docstring promises.
        return ValueType.BLOB
    raise EncodeError(
        f"Cannot infer wire type for value of type {type(value).__name__!r}. "
        f"The wire codec only accepts bool, int, float, str, bytes-like, "
        f"or None. Callers passing datetime/date/etc. must convert to str "
        f"(for ISO8601) or int (for UNIXTIME) at the driver layer."
    )


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

    Inference matches the Go client's ``appendValue`` ``case bool:``
    arm (``internal/protocol/message.go``). The C client has no
    inference layer — ``dqlite-upstream/src/bind.c::bind_one``
    switches on a pre-set ``value->type`` field, with the column-
    decltype-to-tag mapping happening on the server side at
    ``query.c``'s ``value_type`` helper. The Python layer adds
    inference for bind-site ergonomics; the wire-format contract is
    identical across all three clients.
    """
    if value is None:
        if value_type is not None and value_type != ValueType.NULL:
            raise EncodeError(
                f"Cannot encode None with explicit type {value_type.name}. "
                f"Pass value_type=ValueType.NULL or omit value_type."
            )
        return b"\x00" * 8, ValueType.NULL

    # Hoist the memoryview format check to fire exactly once, before
    # any branching (inferred-vs-explicit, BLOB-vs-other). Previously
    # the check ran in BOTH ``_infer_value_type`` (when value_type was
    # None) AND the explicit BLOB arm (after inference filled in
    # BLOB), so the inferred-BLOB path paid the cost twice — defeating
    # the O(1)-per-cell promise of the inference helper. The hoist
    # makes the contract grep-visible: "the check fires exactly once
    # per encode_value call, before any path branching."
    if isinstance(value, memoryview):
        _reject_non_byte_format_memoryview(value)

    if value_type is None:
        # Parity with the explicit BLOB branch and with stdlib
        # ``sqlite3``: bytes-like types (incl. memoryview and memory-
        # mapped regions) infer to BLOB. ``array.array`` is intentionally
        # NOT accepted by the inference helper: typed numeric arrays
        # have platform-dependent bytes representation, so silent
        # acceptance would yield non-portable BLOBs. ``memoryview`` over
        # a non-byte format is rejected via ``_reject_non_byte_format_
        # memoryview`` inside the helper.
        value_type = _infer_value_type(value)

    if value_type == ValueType.BOOLEAN:
        # Accept bool directly; allow the exact ints 0 and 1 as a
        # pragmatic escape for callers working with raw column values.
        # Reject arbitrary ints — the previous ``1 if value else 0``
        # coercion silently mapped values like ``5`` or ``-1`` to True,
        # which round-trips as the bool True and loses the caller's
        # original value.
        #
        # Deliberate asymmetry with C bind (``dqlite-upstream/src/
        # bind.c::DQLITE_BOOLEAN``: ``sqlite3_bind_int64(stmt, n,
        # value->boolean == 0 ? 0 : 1)``, which accepts any uint64
        # and collapses non-zero to 1). A peer-emitted BOOLEAN cell
        # with raw=5 decodes to ``True`` here (see :func:`decode_value`
        # BOOLEAN arm), but attempting to re-encode that ``True``
        # writes raw=1 — so a decode -> re-encode round-trip of an
        # out-of-{0, 1} raw BOOLEAN cell is lossy by design. The
        # Python strictness exists to surface caller bugs
        # (``encode_value(5, ValueType.BOOLEAN)``) which the C path
        # silently rewrites to 1.
        #
        # The ``isinstance(value, int)`` guard intentionally admits
        # int SUBCLASSES (e.g. ``enum.IntEnum`` members) but not
        # numeric proxies like ``numpy.int64`` / ``numpy.bool_`` /
        # ``decimal.Decimal`` / ``fractions.Fraction`` — they expose
        # ``__int__`` / ``__index__`` but do NOT subclass ``int``.
        # Silently accepting them would round-trip ``numpy.int64(5)``
        # through ``True`` and lose the caller's original value (same
        # footgun as the strictness above). Mirrors the documented
        # rejection rationale at ``encode_double``: callers passing
        # numeric proxies must coerce explicitly (``int(x)`` /
        # ``bool(x)``) at the bind site.
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
        # Strict reject of ``int`` (and ``bool``, which subclasses int)
        # for explicit FLOAT. Implicit ``float(int_value)`` would lose
        # bits silently for |x| >= 2**53 (round-to-nearest-double) and
        # raise ``OverflowError`` outside the ``EncodeError`` family
        # for |x| >= 2**1024. Both surfaces collapse into the same
        # rejection here; callers who genuinely want int -> FLOAT
        # write ``float(x)`` themselves and accept the precision
        # boundary explicitly. Mirrors the bool-reject discipline at
        # the INTEGER / UNIXTIME branch above.
        if isinstance(value, bool):
            raise EncodeError(f"Expected float for FLOAT, got {type(value).__name__}")
        if isinstance(value, int):
            raise EncodeError(
                "Cannot encode int as FLOAT; cast with float(x) explicitly. "
                "Implicit coercion would silently lose precision for ints "
                "with absolute value >= 2**53."
            )
        # The residual non-float case (numeric proxies like
        # ``numpy.float64``, ``Decimal``, ``Fraction``) is rejected by
        # ``encode_double``'s float-subclass check with the canonical
        # "encode_double requires float ... cast with float(x)
        # explicitly" wording. A redundant ``if not isinstance(value,
        # float)`` guard here would fire first with strictly poorer
        # diagnostics — let the richer message win. The cast is a
        # static-typing concession: ``encode_double`` is annotated
        # ``float``-only and surfaces the rejection at runtime for
        # anything else.
        return encode_double(cast(float, value)), value_type
    elif value_type in (ValueType.TEXT, ValueType.ISO8601):
        if not isinstance(value, str):
            raise EncodeError(f"Expected str for {value_type.name}, got {type(value).__name__}")
        # Cap the row-cell text length symmetric with the BLOB branch
        # below and with ``decode_text``'s ``_MAX_TEXT_VALUE_SIZE`` cap
        # — without ``max_size=`` here, a >16 MiB string encodes
        # successfully but the same Python decoder rejects the bytes,
        # so a same-process Python encode → decode round-trip can
        # fail. Pass ``label=value_type.name`` so the diagnostic names
        # the wire-cell type (TEXT / ISO8601), not the generic
        # "Text".
        return encode_text(value, max_size=_MAX_TEXT_VALUE_SIZE, label=value_type.name), value_type
    elif value_type == ValueType.BLOB:
        if not isinstance(value, (bytes, bytearray, memoryview, mmap.mmap)):
            raise EncodeError(f"Expected bytes for BLOB, got {type(value).__name__}")
        # NOTE: the ``_reject_non_byte_format_memoryview`` check used
        # to live here (symmetric with the inference branch). It is
        # now hoisted to the top of ``encode_value`` so it fires
        # exactly once per call regardless of inferred-vs-explicit
        # path — see the comment at the function head.
        # Cap the size BEFORE materialising via ``bytes(value)`` —
        # symmetric with ``decode_blob``'s cap-before-allocate
        # ordering. Without this, a 100 MB ``bytearray`` is copied
        # to ``bytes`` first, then rejected by ``encode_blob``'s
        # length check; the materialise allocation is wasted.
        #
        # For ``memoryview`` the probe must use ``.nbytes`` — NOT
        # ``len(mv)`` — because ``len(memoryview(array.array('Q',
        # [...])))`` is the ELEMENT count, not the byte count. The
        # earlier ``_reject_non_byte_format_memoryview`` only admits
        # single-byte formats where ``len == nbytes`` by coincidence,
        # so a ``len()`` probe is correct today only because of that
        # ordering. ``.nbytes`` makes the cap correctness robust by
        # construction: a future refactor that moves or removes the
        # format check cannot silently re-open a cap bypass through a
        # multi-byte memoryview where ``8 * len(mv)`` bytes would
        # materialise. For ``bytes`` / ``bytearray`` / ``mmap.mmap``
        # the byte count is ``len(value)`` by type.
        #
        # ``len()`` works on every supported open container without
        # materialising. A closed ``mmap.mmap`` raises
        # ``ValueError("mmap closed or invalid")`` from ``len()``;
        # a released ``memoryview`` raises ``ValueError`` likewise
        # from ``.nbytes`` / ``len()``. Fall through to the materialise
        # step in that case so the original ``EncodeError`` shape (with
        # the same message prefix) surfaces — preserving backwards
        # compatibility for callers catching on the existing message
        # text.
        try:
            length: int | None = value.nbytes if isinstance(value, memoryview) else len(value)
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
            f"Cannot encode non-None value {_bounded_repr_any(value)} as NULL. "
            f"Pass value=None or use the appropriate ValueType."
        )
    else:
        raise EncodeError(f"Unknown value type: {value_type}")


def decode_value(
    data: bytes | memoryview,
    value_type: ValueType,
    *,
    text_errors: str = "strict",
) -> tuple[WireValue, int]:
    """Decode a value from wire format.

    Returns (value, bytes_consumed).

    ``text_errors`` is forwarded to :func:`decode_text` for TEXT and
    ISO8601 cells; the default ``"strict"`` matches the dqlite
    wire-spec contract (UTF-8 only). Cluster data that violates the
    contract (e.g. rows from a legacy SQLite store with latin-1
    TEXT) can be coerced via ``text_errors="replace"`` /
    ``"backslashreplace"`` / ``"surrogateescape"`` — same set
    :func:`decode_text` accepts — without bypassing the row-decode
    pipeline. The same knob is reachable at the streaming layer via
    ``MessageDecoder(text_errors=...)`` and propagates through to
    every TEXT / ISO8601 cell.

    See :func:`decode_text`'s WARNING about ``"replace"`` /
    ``"surrogateescape"`` bypassing :func:`sanitize_for_log` and
    failing re-encode before adopting a non-strict mode.
    """
    if value_type == ValueType.BOOLEAN:
        # Returns ``bool`` (NOT raw int). Justified by the module
        # charter at the top of this file, which enumerates ``bool``
        # as a wire primitive — not by raw preservation. Contrast with
        # the UNIXTIME arm below, which DOES preserve the raw int64
        # because epoch-seconds is the primitive and ``datetime`` is
        # the driver-layer semantic. The principled split: a type is
        # "raw"-preserved at the wire layer iff the wire primitive is
        # itself the canonical representation; ``bool`` is canonical
        # for BOOLEAN, ``int`` is canonical for UNIXTIME, and the
        # datetime conversion is deferred to the driver
        # (``dqlitedbapi``). A peer-emitted BOOLEAN cell carrying a
        # raw uint64 outside ``{0, 1}`` loses the original integer
        # through this collapse — see the encode-side asymmetry note
        # on ``encode_value``'s BOOLEAN arm.
        #
        # Permissive truthiness decode: read the wire cell as an
        # unsigned uint64 (matching C's ``uint64__decode(d->cursor,
        # &value->boolean)`` in ``dqlite-upstream/src/tuple.c``,
        # ``DQLITE_BOOLEAN`` case in ``tuple_decoder__next``) and
        # coerce to ``bool``. Go's reference decoder uses signed
        # ``getInt64() != 0`` instead
        # (``internal/protocol/message.go:566``), which agrees on
        # truthiness for every bit pattern but diverges on the raw
        # int for the upper half of the range
        # (``[2**63, 2**64 - 1]``); Python tracks the C side so a
        # hypothetical future "keep_raw=True" debug knob would yield
        # C-shaped values, not Go-shaped.
        #
        # The dqlite C server emits a ``DQLITE_BOOLEAN`` cell whenever
        # the column's decltype is ``BOOLEAN`` and the value comes
        # from ``sqlite3_column_int64`` (``encodeColumn`` in
        # ``dqlite-upstream/src/query.c``, ``DQLITE_BOOLEAN`` case) —
        # i.e. any int the column actually contains. SQLite typing is
        # dynamic, so a row inserted as
        # ``INSERT INTO t(b BOOLEAN) VALUES (5)`` arrives with raw=5
        # on the wire. Strict-rejection would deterministically poison
        # the decoder for legitimate cluster data while Go and C
        # clients work fine.
        raw = decode_uint64(data, label="BOOLEAN cell")
        return bool(raw), 8
    elif value_type == ValueType.INTEGER:
        return decode_int64(data, label="INTEGER cell"), 8
    elif value_type == ValueType.UNIXTIME:
        # Returns raw int64 (NOT datetime). Contrast with the BOOLEAN
        # arm above, which DOES collapse to ``bool`` because ``bool``
        # is a wire primitive per the module charter. UNIXTIME's
        # primitive is the epoch-seconds int; the ``datetime``
        # semantic is driver-layer (see ``dqlitedbapi``).
        #
        # Higher-level clients (``dqlitedbapi``) turn this into a
        # UTC-aware ``datetime``; Go's ``Rows.Next()`` does the
        # conversion at the ``database/sql`` driver layer with
        # ``time.Unix(...)``. Bare wire-layer consumers must convert
        # themselves; the wire layer's parity bar is "raw bytes →
        # primitive", not "raw bytes → semantic".
        #
        # NOTE: This arm is intentionally context-blind. UNIXTIME is
        # legitimate in row cells (the C row-encode side at
        # ``dqlite-upstream/src/query.c`` does emit it) but malformed
        # in params (``dqlite-upstream/src/tuple.c::tuple_decoder__next``
        # has no DQLITE_UNIXTIME case in the params switch; the
        # default returns DQLITE_PARSE). The params-vs-row rejection
        # lives at the caller's type-code switch in
        # ``tuples.py::decode_params_tuple``, NOT here — a refactor
        # that moved the rejection into this arm would also break the
        # row decoder, which legitimately needs to accept UNIXTIME.
        return decode_int64(data, label="UNIXTIME cell"), 8
    elif value_type == ValueType.FLOAT:
        return decode_double(data, label="FLOAT cell"), 8
    elif value_type in (ValueType.TEXT, ValueType.ISO8601):
        # ISO8601 is treated as text at the wire level — the C reference
        # uses text__encode / text__decode for DQLITE_ISO8601 (see dqlite
        # src/tuple.c) and Go returns the raw string from the codec.
        # Parsing to datetime belongs in the driver/DBAPI layer.
        # Forward ``value_type.name`` as the label so truncation / not-
        # null-terminated diagnostics name the actual cell type
        # (``"ISO8601 not null-terminated"``) rather than the generic
        # ``"Text ..."`` default — symmetric with ``encode_value``.
        return decode_text(data, label=value_type.name, errors=text_errors)
    elif value_type == ValueType.BLOB:
        return decode_blob(data)
    elif value_type == ValueType.NULL:
        # Permissive decode matching Go's
        # ``r.message.getUint64()`` discard
        # (``internal/protocol/message.go:539``) and C's
        # ``uint64__decode(... &value->null)`` (``tuple.c:147``) where
        # the value is read into a union member that is never
        # inspected. Delegating to ``decode_uint64`` (instead of
        # rolling an inline length check) gives the same wording
        # discipline as the sibling INTEGER / UNIXTIME / FLOAT /
        # BOOLEAN arms — operators grepping logs for "wire decode
        # short-read" see a uniform ``"Need 8 bytes for <type> cell"``
        # phrasing across every short-read case. Our encoder still
        # writes 8 zero bytes
        # (``encode_null``), but a peer / future server / mock that
        # emits non-zero NULL bytes is wire-conformant per upstream's
        # tuple_decoder semantics. Strict-rejection provided no
        # defensive value (the field is documented as unused) and
        # was an interop hazard.
        #
        # Round-trip identity caveat: ``encode_value(None)`` writes 8
        # zero bytes regardless of what the decoder consumed, so a
        # mock-server / proxy / fuzzer that captured a non-zero NULL
        # payload cannot recover byte-identity through the public
        # encode path. The public encoder is uniformly zero-emitting;
        # callers needing byte-identity must emit the captured 8 bytes
        # plus the NULL type tag directly. The asymmetry mirrors the
        # sibling permissive-read patterns at
        # :meth:`DbResponse.decode_body` and
        # :meth:`EmptyResponse.decode_body`, which document the same
        # round-trip lossiness for their reserved fields.
        decode_uint64(data, label="NULL cell")
        return None, 8
    else:
        raise DecodeError(f"Unknown value type: {value_type}")
