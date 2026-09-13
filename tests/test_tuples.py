"""Tests for tuple encoding/decoding and row-header markers."""

from __future__ import annotations

import struct
from typing import cast
from unittest import mock

import pytest

from dqlitewire import tuples as tuples_mod
from dqlitewire.constants import (
    ROW_DONE_BYTE,
    ROW_DONE_MARKER,
    ROW_PART_BYTE,
    ROW_PART_MARKER,
    ValueType,
)
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.tuples import (
    _ROW_DONE_MARKER,
    _ROW_PART_MARKER,
    RowMarker,
    decode_params_tuple,
    decode_row_header,
    decode_row_values,
    encode_params_tuple,
    encode_row_header,
    encode_row_values,
)
from dqlitewire.types import NULL_CELL_WIDTH, WireInput, decode_value, encode_value


class TestParamsTuple:
    def test_encode_empty(self) -> None:
        """Empty params encode to nothing, matching Go."""
        encoded = encode_params_tuple([])
        assert encoded == b""

    def test_params_tuple_rejects_unixtime_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reject a UNIXTIME tag on an outgoing param: the C server can't decode it inbound."""
        from dqlitewire import tuples as tuples_mod

        def fake_encode(value: object, value_type: object = None) -> tuple[bytes, ValueType]:
            return b"\x00" * 8, ValueType.UNIXTIME

        monkeypatch.setattr(tuples_mod, "encode_value", fake_encode)
        with pytest.raises(EncodeError, match="UNIXTIME"):
            encode_params_tuple([1_700_000_000])

    def test_decode_params_tuple_rejects_unixtime_tag(self) -> None:
        """Reject a params-tuple body with type tag 9 (UNIXTIME), matching the C
        server's ``tuple_decoder__next`` default → DQLITE_PARSE; otherwise a Python
        mock-server accepts what a real C gateway rejects."""
        import struct

        body = (
            b"\x01"  # count
            + bytes([ValueType.UNIXTIME])  # type tag = 9
            + b"\x00" * 6
            + struct.pack("<q", 1_700_000_000)
        )
        with pytest.raises(DecodeError, match="(?i)UNIXTIME"):
            decode_params_tuple(body)

    def test_encode_single_integer(self) -> None:
        encoded = encode_params_tuple([42])
        assert len(encoded) == 16
        assert encoded[0] == 1  # count
        assert encoded[1] == ValueType.INTEGER  # type code
        import struct

        assert struct.unpack("<q", encoded[8:16])[0] == 42

    def test_encode_multiple_integers(self) -> None:
        encoded = encode_params_tuple([1, 2, 3])
        assert len(encoded) == 32
        assert encoded[0] == 3  # count
        assert encoded[1] == ValueType.INTEGER
        assert encoded[2] == ValueType.INTEGER
        assert encoded[3] == ValueType.INTEGER
        decoded, _ = decode_params_tuple(encoded)
        assert decoded == [1, 2, 3]

    def test_encode_mixed_types(self) -> None:
        params: list[WireInput] = [42, "hello", 3.14, None, b"blob"]
        encoded = encode_params_tuple(params)
        assert encoded[0] == 5  # count
        assert encoded[1] == ValueType.INTEGER
        assert encoded[2] == ValueType.TEXT
        assert encoded[3] == ValueType.FLOAT
        assert encoded[4] == ValueType.NULL
        assert encoded[5] == ValueType.BLOB
        decoded, _ = decode_params_tuple(encoded)
        assert decoded[0] == 42
        assert decoded[1] == "hello"
        assert abs(cast(float, decoded[2]) - 3.14) < 0.0001
        assert decoded[3] is None
        assert decoded[4] == b"blob"

    def test_decode_empty(self) -> None:
        values, consumed = decode_params_tuple(b"")
        assert values == []
        assert consumed == 0

    def test_decode_rejects_short_data_for_nonzero_count(self) -> None:
        """count>0 with too few bytes must raise, not return [] (drops params on a torn read)."""
        from dqlitewire.exceptions import DecodeError

        with pytest.raises(DecodeError, match="Not enough data"):
            decode_params_tuple(b"\x00\x00\x00\x00", count=1)

    def test_decode_short_data_with_zero_count_returns_empty(self) -> None:
        """count=0 with short data is legitimate (upstream bound count to zero); returns []."""
        values, consumed = decode_params_tuple(b"\x00\x00\x00\x00", count=0)
        assert values == []
        assert consumed == 0

    def test_roundtrip_integers(self) -> None:
        params = [1, 2, 3, 100, -50]
        encoded = encode_params_tuple(params)
        decoded, _ = decode_params_tuple(encoded)
        assert decoded == params

    def test_roundtrip_mixed(self) -> None:
        params: list[WireInput] = [42, "hello", 3.14, None]
        encoded = encode_params_tuple(params)
        decoded, _ = decode_params_tuple(encoded)
        assert decoded[0] == 42
        assert decoded[1] == "hello"
        assert abs(cast(float, decoded[2]) - 3.14) < 0.0001
        assert decoded[3] is None

    def test_encode_v1_schema(self) -> None:
        """V1 encoding uses uint32 count instead of uint8."""
        params = [1, 2, 3]
        encoded = encode_params_tuple(params, schema=1)
        # V1 header: count(4) + 3 types + padding(1) = 8, 3 values = 24, total = 32
        assert len(encoded) == 32
        import struct

        count = struct.unpack("<I", encoded[:4])[0]
        assert count == 3

    def test_roundtrip_v1(self) -> None:
        params: list[WireInput] = [42, "hello"]
        encoded = encode_params_tuple(params, schema=1)
        decoded, _ = decode_params_tuple(encoded, schema=1)
        assert decoded[0] == 42
        assert decoded[1] == "hello"

    def test_decode_zero_count_v0_consumes_header(self) -> None:
        """Decoding a V0 count=0 tuple consumes the 8-byte header word."""
        data = b"\x00" * 8 + b"\xaa" * 8  # header + trailing data
        values, consumed = decode_params_tuple(data, schema=0)
        assert values == []
        assert consumed == 8

    def test_decode_zero_count_v1_consumes_header(self) -> None:
        """Decoding a V1 count=0 tuple consumes the 8-byte header word."""
        import struct

        data = struct.pack("<I", 0) + b"\x00" * 4 + b"\xbb" * 8
        values, consumed = decode_params_tuple(data, schema=1)
        assert values == []
        assert consumed == 8

    def test_v0_rejects_more_than_255_params(self) -> None:
        """V0 uses a uint8 count, so > 255 params must raise EncodeError."""
        import pytest

        from dqlitewire.exceptions import EncodeError

        params = list(range(256))
        with pytest.raises(EncodeError, match="255"):
            encode_params_tuple(params, schema=0)

    def test_v0_255_params_boundary(self) -> None:
        """Exactly 255 params is the V0 maximum (count byte 0xFF)."""
        params = list(range(255))
        encoded = encode_params_tuple(params, schema=0)
        assert encoded[0] == 255

        decoded, consumed = decode_params_tuple(encoded, schema=0)
        assert len(decoded) == 255
        assert decoded == params
        assert consumed == len(encoded)


class TestParamsTupleExternalCount:
    """External count: data has no count field (types start at data[0]); caller
    passes buffer_offset for the externally-consumed count (1 byte V0, 4 bytes V1)
    so padding aligns."""

    def test_decode_with_external_count_v0(self) -> None:
        """External count reads types from data[0]; buffer_offset=1 accounts for the count byte."""
        params: list[WireInput] = [42, "hello"]
        encoded = encode_params_tuple(params, schema=0)
        data_without_count = encoded[1:]
        decoded, consumed = decode_params_tuple(
            data_without_count, count=2, schema=0, buffer_offset=1
        )
        assert decoded[0] == 42
        assert decoded[1] == "hello"

    def test_decode_with_external_count_v1(self) -> None:
        """External count with V1 skips the 4-byte count prefix; buffer_offset=4."""
        params: list[WireInput] = [42, "hello"]
        encoded = encode_params_tuple(params, schema=1)
        data_without_count = encoded[4:]
        decoded, consumed = decode_params_tuple(
            data_without_count, count=2, schema=1, buffer_offset=4
        )
        assert decoded[0] == 42
        assert decoded[1] == "hello"

    def test_decode_with_external_count_mixed_types(self) -> None:
        """External count with mixed types should decode correctly."""
        params: list[WireInput] = [42, 3.14, "text"]
        encoded = encode_params_tuple(params, schema=0)
        data_without_count = encoded[1:]
        decoded, consumed = decode_params_tuple(
            data_without_count, count=3, schema=0, buffer_offset=1
        )
        assert decoded[0] == 42
        assert abs(cast(float, decoded[1]) - 3.14) < 1e-10
        assert decoded[2] == "text"

    def test_decode_with_external_count_zero(self) -> None:
        """External count=0 should return empty list immediately."""
        decoded, consumed = decode_params_tuple(b"\x00" * 8, count=0, schema=0)
        assert decoded == []
        assert consumed == 0

    @pytest.mark.parametrize(
        ("data", "count", "schema", "buffer_offset"),
        [
            # count=10 V0, buffer_offset=0 → padded header is 16 bytes
            (b"\x01" * 8, 10, 0, 0),  # short-by-eight
            (b"\x01" * 15, 10, 0, 0),  # short-by-one
            # count=14 V1, buffer_offset=2 → padded header is 14 bytes
            (b"\x01" * 8, 14, 1, 2),
            (b"\x01" * 13, 14, 1, 2),
            # count=24 V0, buffer_offset=1 → padded = 24 + pad_to_word(25) = 31
            (b"\x01" * 30, 24, 0, 1),
        ],
    )
    def test_decode_with_external_count_short_data(
        self, data: bytes, count: int, schema: int, buffer_offset: int
    ) -> None:
        """External-count branch must raise DecodeError, not IndexError,
        when ``data`` is shorter than the padded type-code header.

        The explicit ``len(data) < padded_header_len`` guard protects the
        type-code read at ``data[count_size + i]``; a regression that
        reordered the check to run after the read would raise
        ``IndexError`` on this path. Every case here has
        ``len(data) >= 8`` (the earlier truncation check passes) so the
        test exercises only the padded-header-length branch.
        """
        with pytest.raises(DecodeError, match="Not enough data for param types"):
            decode_params_tuple(data, count=count, schema=schema, buffer_offset=buffer_offset)


class TestParamsTupleBufferOffset:
    def test_aligned_offset_matches_no_offset(self) -> None:
        """With word-aligned buffer_offset, padding is same as offset=0."""
        params = [42]
        encoded_default = encode_params_tuple(params, buffer_offset=0)
        encoded_aligned = encode_params_tuple(params, buffer_offset=8)
        assert encoded_default == encoded_aligned

    def test_non_aligned_offset_changes_padding(self) -> None:
        """With non-word-aligned buffer_offset, padding differs from offset=0."""
        # 1 param: count(1) + type(1) = 2 header bytes
        # At buffer_offset=0: absolute=2, pad to 8 -> 6 padding bytes
        # At buffer_offset=2: absolute=4, pad to 8 -> 4 padding bytes
        params = [42]
        encoded_at_0 = encode_params_tuple(params, buffer_offset=0)
        encoded_at_2 = encode_params_tuple(params, buffer_offset=2)
        # Header+padding at offset 0: 2 + 6 = 8 bytes + 8 value = 16
        assert len(encoded_at_0) == 16
        # Header+padding at offset 2: 2 + 4 = 6 bytes + 8 value = 14
        assert len(encoded_at_2) == 14

    def test_roundtrip_with_non_aligned_offset(self) -> None:
        """Encode and decode with non-aligned offset must roundtrip."""
        params: list[WireInput] = [42, "hello"]
        encoded = encode_params_tuple(params, buffer_offset=4)
        decoded, _ = decode_params_tuple(encoded, buffer_offset=4)
        assert decoded == [42, "hello"]

    def test_v1_with_non_aligned_offset(self) -> None:
        """V1 schema with non-aligned offset must roundtrip."""
        params = [1, 2, 3]
        encoded = encode_params_tuple(params, schema=1, buffer_offset=4)
        decoded, _ = decode_params_tuple(encoded, schema=1, buffer_offset=4)
        assert decoded == [1, 2, 3]


class TestRowHeader:
    def test_encode_empty(self) -> None:
        encoded = encode_row_header([])
        assert encoded == b""

    def test_encode_single(self) -> None:
        encoded = encode_row_header([ValueType.INTEGER])
        # 4-bit codes: 1 type needs 1 byte (half used) + 7 padding = 8 bytes
        assert len(encoded) == 8
        # Lower nibble should have INTEGER (1)
        assert (encoded[0] & 0x0F) == ValueType.INTEGER

    def test_encode_two_types(self) -> None:
        types = [ValueType.INTEGER, ValueType.TEXT]
        encoded = encode_row_header(types)
        # 2 types fit in 1 byte + 7 padding = 8 bytes
        assert len(encoded) == 8
        # Lower nibble: INTEGER (1), upper nibble: TEXT (3)
        assert (encoded[0] & 0x0F) == ValueType.INTEGER
        assert ((encoded[0] >> 4) & 0x0F) == ValueType.TEXT

    def test_encode_multiple(self) -> None:
        types = [ValueType.INTEGER, ValueType.TEXT, ValueType.FLOAT]
        encoded = encode_row_header(types)
        # 3 types need 2 bytes (4 slots, 1 unused) + 6 padding = 8 bytes
        assert len(encoded) == 8

    def test_encode_rejects_type_exceeding_nibble(self) -> None:
        """ValueType codes >= 16 cannot fit in 4-bit nibble and must be rejected."""
        import pytest

        from dqlitewire.exceptions import EncodeError

        # Create a fake type with value 16 (doesn't fit in 4 bits)
        fake_type = 16
        with pytest.raises(EncodeError, match="nibble"):
            encode_row_header([fake_type])  # type: ignore[list-item]

    def test_decode_empty(self) -> None:
        types, consumed = decode_row_header(b"", 0)
        assert types == []
        assert consumed == 0

    def test_decode_single(self) -> None:
        # 4-bit encoding: INTEGER (1) in lower nibble
        data = bytes([ValueType.INTEGER]) + b"\x00" * 7
        types, consumed = decode_row_header(data, 1)
        assert types == [ValueType.INTEGER]
        assert consumed == 8

    def test_decode_two_types(self) -> None:
        # Pack INTEGER (1) in lower nibble, TEXT (3) in upper nibble
        packed_byte = ValueType.INTEGER | (ValueType.TEXT << 4)
        data = bytes([packed_byte]) + b"\x00" * 7
        types, consumed = decode_row_header(data, 2)
        assert types == [ValueType.INTEGER, ValueType.TEXT]
        assert consumed == 8

    def test_roundtrip(self) -> None:
        types = [ValueType.INTEGER, ValueType.TEXT, ValueType.BLOB, ValueType.NULL]
        encoded = encode_row_header(types)
        decoded, _ = decode_row_header(encoded, len(types))
        assert decoded == types

    def test_decode_done_marker(self) -> None:
        """decode_row_header should detect 0xFF marker byte (done) like Go does."""
        data = b"\xff" * 8
        result = decode_row_header(data, 1)
        assert result == (RowMarker.DONE, 8)

    def test_decode_part_marker(self) -> None:
        """decode_row_header should detect 0xEE marker byte (more rows) like Go does."""
        data = b"\xee" * 8
        result = decode_row_header(data, 1)
        assert result == (RowMarker.PART, 8)

    def test_decode_done_marker_multi_column(self) -> None:
        """Marker detection should work regardless of column count."""
        data = b"\xff" * 8
        result = decode_row_header(data, 4)
        assert result == (RowMarker.DONE, 8)

    def test_decode_done_marker_17_columns(self) -> None:
        """Marker must be detected even when header_size would exceed 8 bytes.

        With 17 columns, the type header needs 9 bytes (padded to 16), but
        the marker is always exactly 8 bytes. Marker detection must happen
        before the header size validation.
        """
        data = b"\xff" * 8
        result = decode_row_header(data, 17)
        assert result == (RowMarker.DONE, 8)

    def test_decode_part_marker_20_columns(self) -> None:
        """PART marker must also work with large column counts."""
        data = b"\xee" * 8
        result = decode_row_header(data, 20)
        assert result == (RowMarker.PART, 8)

    def test_decode_done_marker_33_columns(self) -> None:
        """Marker must work even with very large column counts (header_size=24)."""
        data = b"\xff" * 8
        result = decode_row_header(data, 33)
        assert result == (RowMarker.DONE, 8)

    def test_full_uint64_marker_comparison(self) -> None:
        """Full uniform markers must be detected."""
        assert decode_row_header(b"\xff" * 8, 1) == (RowMarker.DONE, 8)
        assert decode_row_header(b"\xee" * 8, 1) == (RowMarker.PART, 8)

    def test_non_uniform_marker_rejected(self) -> None:
        """Non-uniform markers are rejected as corrupt.

        Upstream C uses the full uint64 sentinel (DQLITE_RESPONSE_ROWS_DONE
        = 0xff..ff, _PART = 0xee..ee). Go's reference client accepts any
        8 bytes starting with 0xff/0xee as a marker; we validate all 8
        bytes so torn/corrupt frames are rejected rather than silently
        truncating results.

        ValueType max is 11 (0xb) — a real row-header byte can never be
        0xff or 0xee — so the "strictly C-aligned" check and the
        "ValueType rejection on nibble 0xf/0xe" arrive at the same
        outcome from different angles.
        """
        # First byte 0xff, remaining zero → falls through to ValueType
        # nibble decode, which rejects 0xf.
        with pytest.raises(DecodeError, match="Invalid value type code"):
            decode_row_header(b"\xff\x00\x00\x00\x00\x00\x00\x00", 1)

        with pytest.raises(DecodeError, match="Invalid value type code"):
            decode_row_header(b"\xee\x00\x00\x00\x00\x00\x00\x00", 1)

    def test_marker_sentinel_bytes_match_full_constants(self) -> None:
        """ROW_DONE_BYTE/ROW_PART_BYTE must match the first byte of the full marker words.

        The full marker words (ROW_DONE_MARKER, ROW_PART_MARKER) are written as
        uint64 on the wire, but detection uses only the first byte. The sentinel
        byte constants must be consistent with the full words.
        """
        from dqlitewire.constants import (
            ROW_DONE_BYTE,
            ROW_DONE_MARKER,
            ROW_PART_BYTE,
            ROW_PART_MARKER,
        )
        from dqlitewire.types import encode_uint64

        # Sentinel bytes must match the first byte of encoded marker words
        done_wire = encode_uint64(ROW_DONE_MARKER)
        assert done_wire[0] == ROW_DONE_BYTE

        part_wire = encode_uint64(ROW_PART_MARKER)
        assert part_wire[0] == ROW_PART_BYTE

    def test_decode_invalid_type_code_raises_decode_error(self) -> None:
        """Invalid nibble value (0) in row header must raise DecodeError, not ValueError."""
        import pytest

        from dqlitewire.exceptions import DecodeError

        # Byte 0x00 means both nibbles are 0, which is not a valid ValueType
        data = b"\x00" * 8
        with pytest.raises(DecodeError, match="Invalid value type") as exc_info:
            decode_row_header(data, 1)
        # Chain preserves the underlying ValueError so log backends show
        # "the above exception was the direct cause" with the enum name.
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestParamsTupleInvalidType:
    def test_decode_invalid_type_code_raises_decode_error(self) -> None:
        """Invalid type byte (0) in params tuple must raise DecodeError, not ValueError."""
        import pytest

        from dqlitewire.exceptions import DecodeError

        # count=1, type=0 (invalid), padding to 8 bytes
        data = b"\x01\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 8
        with pytest.raises(DecodeError, match="Invalid value type") as exc_info:
            decode_params_tuple(data)
        # Chain preserves the underlying ValueError so log backends show
        # "the above exception was the direct cause" with the enum name.
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestParamsTupleSchemaValidation:
    def test_encode_rejects_schema_2(self) -> None:
        import pytest

        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="Unsupported params tuple schema"):
            encode_params_tuple([42], schema=2)

    def test_encode_rejects_negative_schema(self) -> None:
        import pytest

        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="Unsupported params tuple schema"):
            encode_params_tuple([42], schema=-1)

    def test_decode_rejects_schema_2(self) -> None:
        import pytest

        from dqlitewire.exceptions import DecodeError

        data = encode_params_tuple([42], schema=0)
        with pytest.raises(DecodeError, match="Unsupported params tuple schema"):
            decode_params_tuple(data, schema=2)

    def test_decode_rejects_negative_schema(self) -> None:
        import pytest

        from dqlitewire.exceptions import DecodeError

        data = encode_params_tuple([42], schema=0)
        with pytest.raises(DecodeError, match="Unsupported params tuple schema"):
            decode_params_tuple(data, schema=-1)

    def test_encode_accepts_schema_0_and_1(self) -> None:
        """Schema 0 and 1 should work normally."""
        encode_params_tuple([42], schema=0)
        encode_params_tuple([42], schema=1)


class TestParamsTupleErrors:
    def test_decode_insufficient_data_for_header(self) -> None:
        """Data shorter than 8 bytes should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError

        with pytest.raises(DecodeError, match="Not enough data"):
            decode_params_tuple(b"\x01\x02\x03")

    def test_decode_empty_data_with_nonzero_count_raises(self) -> None:
        """Empty data with count > 0 should raise DecodeError, not return []."""
        import pytest

        from dqlitewire.exceptions import DecodeError

        with pytest.raises(DecodeError, match="Expected data for 5 parameters"):
            decode_params_tuple(b"", count=5)

    def test_decode_empty_data_with_count_none_returns_empty(self) -> None:
        """Empty data with count=None is legitimate (Go writes nothing for empty params)."""
        result, consumed = decode_params_tuple(b"", count=None)
        assert result == []
        assert consumed == 0

    def test_decode_empty_data_with_count_zero_returns_empty(self) -> None:
        """Empty data with count=0 is legitimate."""
        result, consumed = decode_params_tuple(b"", count=0)
        assert result == []
        assert consumed == 0

    def test_decode_insufficient_data_for_types(self) -> None:
        """Data too short for declared type count should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError

        # count=100, but only 8 bytes total (not enough for 100 type codes)
        data = b"\x64" + b"\x00" * 7
        with pytest.raises(DecodeError, match="Not enough data for param types"):
            decode_params_tuple(data)


class TestRowHeaderErrors:
    def test_decode_insufficient_data(self) -> None:
        """Row header with insufficient data should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError

        # Need 8 bytes for 1 column header, but only have 4
        with pytest.raises(DecodeError, match="Not enough data for row header"):
            decode_row_header(b"\x01\x02\x03\x04", 1)

    def test_encode_rejects_type_code_zero(self) -> None:
        """encode_row_header should reject type code 0, which is not a valid ValueType."""
        import pytest

        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="[Ii]nvalid.*type"):
            encode_row_header([0])  # type: ignore[list-item]

    def test_encode_rejects_undefined_type_codes(self) -> None:
        """encode_row_header should reject type codes not defined in ValueType."""
        import pytest

        from dqlitewire.exceptions import EncodeError

        # Type codes 6, 7, 8 are undefined
        for code in [6, 7, 8, 12, 13, 14, 15]:
            with pytest.raises(EncodeError, match="[Ii]nvalid.*type"):
                encode_row_header([code])  # type: ignore[list-item]


class TestRowValuesBlob:
    def test_roundtrip_with_blob(self) -> None:
        """Blob values in rows should roundtrip correctly."""
        values = [b"\x01\x02\x03"]
        types = [ValueType.BLOB]
        encoded = encode_row_values(values, types)
        decoded, _ = decode_row_values(encoded, types)
        assert decoded == values


class TestRowValues:
    def test_encode_single_integer(self) -> None:
        values = [42]
        types = [ValueType.INTEGER]
        encoded = encode_row_values(values, types)
        assert len(encoded) == 8

    def test_roundtrip_integers(self) -> None:
        values = [1, 2, 3]
        types = [ValueType.INTEGER, ValueType.INTEGER, ValueType.INTEGER]
        encoded = encode_row_values(values, types)
        decoded, _ = decode_row_values(encoded, types)
        assert decoded == values

    def test_roundtrip_mixed(self) -> None:
        values: list[WireInput] = [42, "hello", 3.14]
        types = [ValueType.INTEGER, ValueType.TEXT, ValueType.FLOAT]
        encoded = encode_row_values(values, types)
        decoded, _ = decode_row_values(encoded, types)
        assert decoded[0] == 42
        assert decoded[1] == "hello"
        assert abs(cast(float, decoded[2]) - 3.14) < 0.0001

    def test_encode_mismatched_lengths_raises_encode_error(self) -> None:
        """Mismatched values/types lengths should raise EncodeError, not ValueError."""
        import pytest

        from dqlitewire.exceptions import EncodeError

        with pytest.raises(EncodeError, match="does not match"):
            encode_row_values([1, 2, 3], [ValueType.INTEGER, ValueType.INTEGER])

        with pytest.raises(EncodeError, match="does not match"):
            encode_row_values([1], [ValueType.INTEGER, ValueType.INTEGER])

    def test_roundtrip_with_null(self) -> None:
        values = [42, None, "test"]
        types = [ValueType.INTEGER, ValueType.NULL, ValueType.TEXT]
        encoded = encode_row_values(values, types)
        decoded, _ = decode_row_values(encoded, types)
        assert decoded == values


class TestParamsTupleMaxCount:
    """170: decode_params_tuple should reject excessive parameter counts."""

    def test_v1_count_exceeding_cap_raises(self) -> None:
        """A V1 params tuple claiming 200_000 params should be rejected."""
        # Craft a V1 header with count = 200_000
        count = 200_000
        header = struct.pack("<I", count)
        # Pad to a full 8-byte word
        data = header + b"\x00" * 4
        with pytest.raises(DecodeError, match="exceeds maximum"):
            decode_params_tuple(data, schema=1)

    def test_v0_count_within_cap_succeeds(self) -> None:
        """V0 counts (max 255) should never hit the cap."""
        # Encode a valid empty-params V0 tuple (count=0)
        data = b"\x00" + b"\x00" * 7  # count=0, padded to 8 bytes
        result, _ = decode_params_tuple(data, schema=0)
        assert result == []


class TestParamsTupleEncoderMaxCount:
    """encode_params_tuple should mirror decode_params_tuple's cap."""

    def test_encoder_rejects_excess_count(self) -> None:
        from dqlitewire.limits import MAX_PARAM_COUNT

        params = [0] * (MAX_PARAM_COUNT + 1)
        with pytest.raises(EncodeError, match="exceeds maximum"):
            encode_params_tuple(params, schema=1)

    def test_exec_request_rejects_excess_params(self) -> None:
        from dqlitewire.limits import MAX_PARAM_COUNT
        from dqlitewire.messages.requests import ExecRequest

        req = ExecRequest(db_id=1, stmt_id=1, params=[0] * (MAX_PARAM_COUNT + 1))
        with pytest.raises(EncodeError, match="exceeds maximum"):
            req.encode()


# ---- merged from test_decode_row_header_docstring.py ----
# ``decode_row_header`` validates all 8 marker bytes (like the upstream C
# sentinel), not just the first byte (Go). Its docstring must say so, so a
# contributor doesn't relax the check and accept torn markers like ``0xff 0x00...``.


def test_decode_row_header_torn_marker_rejected_not_silently_consumed() -> None:
    """A torn marker (first byte 0xFF, trailing bytes diverge) must NOT be
    treated as DONE — it falls through to the type-header decode path."""
    # Go's first-byte check would accept this as DONE; we must not.
    torn = b"\xff\x00\x00\x00\x00\x00\x00\x00"
    try:
        result = decode_row_header(torn, column_count=1)
    except DecodeError:
        # Narrow catch so a refactor-introduced non-decode error propagates.
        return
    types_or_marker, _ = result
    from dqlitewire.tuples import RowMarker

    assert types_or_marker is not RowMarker.DONE, (
        "Torn marker (first byte 0xFF, rest zeros) must not be accepted as DONE"
    )


# ---- merged from test_decode_row_header_marker_no_per_row_bytes_alloc.py ----
# ``decode_row_header`` and the zero-column fast-path in
# ``RowsResponse.decode_body`` detect DONE / PART markers correctly from both
# memoryview- and bytes-backed input, and reject torn markers.


def test_decode_row_header_marker_detection_works_with_memoryview_input() -> None:
    """DONE and PART markers are identified from memoryview-backed bytes."""
    from dqlitewire.constants import ROW_DONE_BYTE, ROW_PART_BYTE
    from dqlitewire.tuples import RowMarker, decode_row_header

    done_buf = memoryview(bytes([ROW_DONE_BYTE]) * 8 + b"\x00" * 8)
    part_buf = memoryview(bytes([ROW_PART_BYTE]) * 8 + b"\x00" * 8)

    result, consumed = decode_row_header(done_buf, column_count=1)
    assert result is RowMarker.DONE
    assert consumed == 8

    result, consumed = decode_row_header(part_buf, column_count=1)
    assert result is RowMarker.PART
    assert consumed == 8


def test_decode_row_header_marker_detection_works_with_bytes_input() -> None:
    """bytes input still works (the rewrite collapses both paths into one
    direct equality)."""
    from dqlitewire.constants import ROW_DONE_BYTE, ROW_PART_BYTE
    from dqlitewire.tuples import RowMarker, decode_row_header

    done_buf = bytes([ROW_DONE_BYTE]) * 8
    part_buf = bytes([ROW_PART_BYTE]) * 8

    result, consumed = decode_row_header(done_buf, column_count=1)
    assert result is RowMarker.DONE
    assert consumed == 8

    result, consumed = decode_row_header(part_buf, column_count=1)
    assert result is RowMarker.PART
    assert consumed == 8


def test_decode_row_header_rejects_torn_marker_under_memoryview_input() -> None:
    """A torn marker must NOT be accepted as DONE — it falls through to the
    type-decode arm (Go checks only the first byte; we validate all 8)."""
    from dqlitewire.exceptions import DecodeError
    from dqlitewire.tuples import RowMarker, decode_row_header

    # First byte's low nibble 0x0f is an invalid type code -> DecodeError.
    torn = memoryview(b"\xff" + b"\x00" * 15)
    try:
        result, _ = decode_row_header(torn, column_count=1)
    except DecodeError:
        return
    assert result is not RowMarker.DONE, (
        "torn marker incorrectly accepted as DONE under memoryview input"
    )


# ---- merged from test_decode_row_header_precomputed_nibble_lookup.py ----
# ``decode_row_header`` resolves nibble -> ValueType via a module-level
# precomputed ``_NIBBLE_TO_VALUETYPE`` tuple rather than calling the IntEnum
# constructor per cell (a per-row hot loop).


def test_nibble_to_valuetype_lookup_table_exists_and_is_well_formed() -> None:
    """16 entries: a ValueType at each known type code, None elsewhere."""
    table = tuples_mod._NIBBLE_TO_VALUETYPE
    assert len(table) == 16, "table must cover all 4-bit nibble values"

    valid_codes = {int(v) for v in ValueType}
    for nibble in range(16):
        if nibble in valid_codes:
            assert table[nibble] is not None, f"nibble {nibble} should map to a ValueType"
            assert int(table[nibble]) == nibble  # type: ignore[arg-type]
            assert isinstance(table[nibble], ValueType)
        else:
            assert table[nibble] is None, (
                f"nibble {nibble} is not a known ValueType; should map to None"
            )


def test_decode_row_header_does_not_call_valuetype_constructor_per_cell() -> None:
    """Decoding a header must not invoke the ValueType constructor; the
    precomputed table is consulted instead."""
    # 16-column header: each byte packs two nibbles (low, high), 8 bytes total.
    valid_codes = sorted(int(v) for v in ValueType)
    nibbles = [valid_codes[i % len(valid_codes)] for i in range(16)]
    header_bytes = bytearray(8)
    for i in range(0, 16, 2):
        low = nibbles[i]
        high = nibbles[i + 1]
        header_bytes[i // 2] = (high << 4) | low

    call_count = 0
    original_call = type(ValueType).__call__

    def counting_call(cls, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_call(cls, *args, **kwargs)

    with mock.patch.object(type(ValueType), "__call__", counting_call):
        types, consumed = tuples_mod.decode_row_header(bytes(header_bytes), 16)

    assert consumed == 8
    assert isinstance(types, list)
    assert len(types) == 16
    for t, expected_nibble in zip(types, nibbles, strict=True):
        assert int(t) == expected_nibble

    # The table is built at import time; only per-cell calls reach the patch.
    assert call_count == 0, (
        f"decode_row_header called ValueType() {call_count} times; "
        "expected zero (precomputed _NIBBLE_TO_VALUETYPE table should "
        "replace the per-cell constructor)"
    )


def test_decode_row_header_invalid_nibble_preserves_existing_error_phrasing() -> None:
    """An invalid nibble must still raise with the "Invalid value type code"
    phrasing pinned by test_tuples.py."""
    valid_codes = {int(v) for v in ValueType}
    invalid_nibbles = [n for n in range(16) if n not in valid_codes]
    assert invalid_nibbles, "test setup requires at least one invalid nibble"
    invalid = invalid_nibbles[0]

    header_bytes = bytearray(8)
    header_bytes[0] = invalid

    with pytest.raises(DecodeError, match="Invalid value type code"):
        tuples_mod.decode_row_header(bytes(header_bytes), 1)


# ---- merged from test_null_cell_width_constant.py ----
# Pin: NULL wire cell width is centralised in ``NULL_CELL_WIDTH``
# rather than scattered as a hard-coded ``8`` across the encode arm, the
# decode arm, and the short-read diagnostic.
#
# Anchored to four ``/* TODO: allow null to be encoded with 0 bytes */``
# sites in ``dqlite-upstream/src/tuple.c`` (lines 71-73, 146, 262, 298).
# If upstream lands the 0-byte NULL change, this constant flips and
# both encode_value and decode_value NULL branches pick it up in
# lockstep. The single-point-of-change discipline lets a future
# maintainer answering "how wide is NULL on the wire?" find one site
# instead of three.


def test_null_cell_width_is_eight() -> None:
    """Pinned to the current upstream tuple.c contract: NULL is 8
    bytes wide on the wire. Flips iff upstream lands the long-standing
    TODO at tuple.c:71-73, 146, 262, 298."""
    assert NULL_CELL_WIDTH == 8


def test_encode_value_null_emits_exactly_null_cell_width_zero_bytes() -> None:
    """The encode arm emits ``NULL_CELL_WIDTH`` zero bytes, not a
    hard-coded 8."""
    encoded, vtype = encode_value(None)
    assert vtype == ValueType.NULL
    assert encoded == b"\x00" * NULL_CELL_WIDTH
    assert len(encoded) == NULL_CELL_WIDTH


def test_decode_value_null_consumes_exactly_null_cell_width_bytes() -> None:
    """The decode arm consumes ``NULL_CELL_WIDTH`` bytes."""
    value, consumed = decode_value(b"\x00" * NULL_CELL_WIDTH, ValueType.NULL)
    assert value is None
    assert consumed == NULL_CELL_WIDTH


# ---- merged from test_row_marker_constants_cross_check.py ----
# Pin: the four row-marker constants form a consistent pair.
#
# The wire format identifies the end of a ``RowsResponse`` body with one
# of two 8-byte sentinels: ``0xFF * 8`` (DONE) or ``0xEE * 8`` (PART).
# The package spells the sentinel in two representations:
#
# - ``ROW_DONE_BYTE`` / ``ROW_PART_BYTE`` — single-byte ints (source of
#   truth).
# - ``ROW_DONE_MARKER`` / ``ROW_PART_MARKER`` — uint64 ints used by the
#   encode path.
# - ``_ROW_DONE_MARKER`` / ``_ROW_PART_MARKER`` — 8-byte ``bytes``
#   constants (in ``tuples.py``) used by the decode path.
#
# These four must agree byte-for-byte. Without a cross-check pin, a typo
# in any single-byte constant propagates only to the bytes form (which
# is derived from it) and not to the uint64 int form (or vice versa),
# silently breaking the encode/decode round-trip.
#
# The ``test_row_*_marker_canonical_hex_value`` pins below are also the
# runtime enforcement for the ``if __debug__:`` invariant block in
# ``constants.py``: under ``python -O`` the in-module assertions are
# stripped, but these tests run regardless of optimisation level (the
# pytest runner does not pass ``-O`` to interpreter startup) and would
# fail loudly on any derivation regression.


def test_row_done_marker_uint64_matches_bytes_form() -> None:
    assert ROW_DONE_MARKER.to_bytes(8, "little") == _ROW_DONE_MARKER


def test_row_part_marker_uint64_matches_bytes_form() -> None:
    assert ROW_PART_MARKER.to_bytes(8, "little") == _ROW_PART_MARKER


def test_row_done_marker_bytes_derived_from_byte_constant() -> None:
    assert bytes([ROW_DONE_BYTE]) * 8 == _ROW_DONE_MARKER


def test_row_part_marker_bytes_derived_from_byte_constant() -> None:
    assert bytes([ROW_PART_BYTE]) * 8 == _ROW_PART_MARKER


def test_row_done_marker_canonical_hex_value() -> None:
    """Pin the wire byte sequence to the C upstream macro value."""
    assert ROW_DONE_MARKER == 0xFFFFFFFFFFFFFFFF
    assert _ROW_DONE_MARKER == b"\xff" * 8


def test_row_part_marker_canonical_hex_value() -> None:
    assert ROW_PART_MARKER == 0xEEEEEEEEEEEEEEEE
    assert _ROW_PART_MARKER == b"\xee" * 8


# ---- merged from test_tuples_16_column_round_trip.py ----
# Pin: at column count n=16, no valid ``ValueType`` packing collides
# with the row-marker sentinels (``DONE = 0xFF..FF``, ``PART = 0xEE..EE``).
#
# For n=16 the row-header is exactly one 8-byte word — the same shape as
# a marker. The decoder applies a full-uint64 marker check (strictly
# tighter than Go's first-byte-only check) before the type-nibble
# decode. The safety property: no ``ValueType`` is 14 or 15, so neither
# ``0xEE`` nor ``0xFF`` can arise from packing two valid type nibbles.
# That property holds today by construction; this test fixture pins it
# so a future change — adding a ``ValueType`` with code 14 or 15,
# changing the marker pattern, or changing the row-header layout — can
# not silently violate it.

_VALID_TYPES = list(ValueType)


@pytest.mark.parametrize("type_uniform", _VALID_TYPES)
def test_16_column_uniform_type_does_not_collide_with_row_marker(
    type_uniform: ValueType,
) -> None:
    """A 16-column row header with all-same type nibbles must not pack
    to either marker sentinel. Round-trip through ``decode_row_header``
    must yield the original types, never a ``RowMarker``."""
    types = [type_uniform] * 16
    header_bytes = encode_row_header(types)
    assert len(header_bytes) == 8
    assert header_bytes != b"\xff" * 8, (
        f"ValueType {type_uniform!r} packs to the DONE marker — would be "
        "indistinguishable from end-of-rows on the wire."
    )
    assert header_bytes != b"\xee" * 8, (
        f"ValueType {type_uniform!r} packs to the PART marker — would be "
        "indistinguishable from end-of-batch on the wire."
    )

    decoded, consumed = decode_row_header(header_bytes, column_count=16)
    assert consumed == 8
    assert decoded == types


def test_16_column_no_valid_type_pair_packs_to_marker_byte() -> None:
    """Stronger property: no pair of valid ``ValueType`` codes packs
    to ``0xEE`` or ``0xFF``. Pinning the property exhaustively across
    every (low, high) pair, not just the uniform-type slice. A new
    ``ValueType`` with code 14 or 15 would break this immediately."""
    for low in _VALID_TYPES:
        for high in _VALID_TYPES:
            packed = (int(high) << 4) | int(low)
            assert packed != 0xFF, (
                f"({low!r}, {high!r}) packs to 0xFF — would collide with the "
                "DONE marker if repeated 8 times."
            )
            assert packed != 0xEE, (
                f"({low!r}, {high!r}) packs to 0xEE — would collide with the "
                "PART marker if repeated 8 times."
            )


def test_16_column_done_marker_bytes_decode_as_marker_not_null_row() -> None:
    """A raw 8-byte ``0xFF * 8`` payload must classify as
    ``RowMarker.DONE``, not as a 16-column row of repeated type
    nibbles. Pin the marker check runs before the type-nibble decode
    so the strict-validation contract is preserved at the n=16
    boundary where header_size == marker_size."""
    payload = b"\xff" * 8
    decoded, consumed = decode_row_header(payload, column_count=16)
    assert decoded is RowMarker.DONE
    assert consumed == 8


def test_16_column_part_marker_bytes_decode_as_marker_not_null_row() -> None:
    """Symmetric pin for the PART marker."""
    payload = b"\xee" * 8
    decoded, consumed = decode_row_header(payload, column_count=16)
    assert decoded is RowMarker.PART
    assert consumed == 8
