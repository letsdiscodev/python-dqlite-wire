"""Tests for tuple encoding/decoding."""

from dqlitewire.constants import ValueType
from dqlitewire.tuples import (
    RowMarker,
    decode_params_tuple,
    decode_row_header,
    decode_row_values,
    encode_params_tuple,
    encode_row_header,
    encode_row_values,
)


class TestParamsTuple:
    def test_encode_empty(self) -> None:
        """Empty params should encode to nothing, matching Go behavior."""
        encoded = encode_params_tuple([])
        assert encoded == b""

    def test_encode_single_integer(self) -> None:
        encoded = encode_params_tuple([42])
        assert len(encoded) == 16
        assert encoded[0] == 1  # count
        assert encoded[1] == ValueType.INTEGER  # type code
        # Verify value: 42 as little-endian int64 at offset 8
        import struct

        assert struct.unpack("<q", encoded[8:16])[0] == 42

    def test_encode_multiple_integers(self) -> None:
        encoded = encode_params_tuple([1, 2, 3])
        assert len(encoded) == 32
        assert encoded[0] == 3  # count
        assert encoded[1] == ValueType.INTEGER
        assert encoded[2] == ValueType.INTEGER
        assert encoded[3] == ValueType.INTEGER
        # Roundtrip to verify values
        decoded, _ = decode_params_tuple(encoded)
        assert decoded == [1, 2, 3]

    def test_encode_mixed_types(self) -> None:
        params = [42, "hello", 3.14, None, b"blob"]
        encoded = encode_params_tuple(params)
        assert encoded[0] == 5  # count
        assert encoded[1] == ValueType.INTEGER
        assert encoded[2] == ValueType.TEXT
        assert encoded[3] == ValueType.FLOAT
        assert encoded[4] == ValueType.NULL
        assert encoded[5] == ValueType.BLOB
        # Verify full roundtrip
        decoded, _ = decode_params_tuple(encoded)
        assert decoded[0] == 42
        assert decoded[1] == "hello"
        assert abs(decoded[2] - 3.14) < 0.0001
        assert decoded[3] is None
        assert decoded[4] == b"blob"

    def test_decode_empty(self) -> None:
        """Empty params data should decode to empty list."""
        values, consumed = decode_params_tuple(b"")
        assert values == []
        assert consumed == 0

    def test_roundtrip_integers(self) -> None:
        params = [1, 2, 3, 100, -50]
        encoded = encode_params_tuple(params)
        decoded, _ = decode_params_tuple(encoded)  # count read from data
        assert decoded == params

    def test_roundtrip_mixed(self) -> None:
        params = [42, "hello", 3.14, None]
        encoded = encode_params_tuple(params)
        decoded, _ = decode_params_tuple(encoded)  # count read from data
        assert decoded[0] == 42
        assert decoded[1] == "hello"
        assert abs(decoded[2] - 3.14) < 0.0001
        assert decoded[3] is None

    def test_encode_v1_schema(self) -> None:
        """V1 encoding uses uint32 count instead of uint8."""
        params = [1, 2, 3]
        encoded = encode_params_tuple(params, schema=1)
        # V1 header: count(4) + 3 types + padding(1) = 8, 3 values = 24, total = 32
        assert len(encoded) == 32
        # uint32 count in first 4 bytes (little-endian)
        import struct

        count = struct.unpack("<I", encoded[:4])[0]
        assert count == 3

    def test_roundtrip_v1(self) -> None:
        """V1 params should roundtrip correctly."""
        params = [42, "hello"]
        encoded = encode_params_tuple(params, schema=1)
        decoded, _ = decode_params_tuple(encoded, schema=1)
        assert decoded[0] == 42
        assert decoded[1] == "hello"

    def test_decode_zero_count_v0_consumes_header(self) -> None:
        """Decoding a V0 tuple with count=0 should consume the header word."""
        # V0: count=0 at byte 0, rest padding = 8 bytes total
        data = b"\x00" * 8 + b"\xaa" * 8  # header + trailing data
        values, consumed = decode_params_tuple(data, schema=0)
        assert values == []
        assert consumed == 8  # one word consumed for the header

    def test_decode_zero_count_v1_consumes_header(self) -> None:
        """Decoding a V1 tuple with count=0 should consume the header word."""
        import struct

        # V1: uint32 count=0 at bytes 0-3, rest padding = 8 bytes total
        data = struct.pack("<I", 0) + b"\x00" * 4 + b"\xbb" * 8
        values, consumed = decode_params_tuple(data, schema=1)
        assert values == []
        assert consumed == 8  # one word consumed for the header

    def test_v0_rejects_more_than_255_params(self) -> None:
        """V0 schema uses uint8 count, so > 255 params must raise EncodeError."""
        import pytest

        from dqlitewire.exceptions import EncodeError

        params = list(range(256))
        with pytest.raises(EncodeError, match="255"):
            encode_params_tuple(params, schema=0)


class TestParamsTupleExternalCount:
    """Tests for decode_params_tuple with externally provided count.

    When count is externally provided, the data does NOT contain a count
    field — type codes start at data[0]. The caller must pass buffer_offset
    accounting for the count field that was consumed externally (1 byte for
    V0, 4 bytes for V1) so that padding aligns correctly.
    """

    def test_decode_with_external_count_v0(self) -> None:
        """External count should read types from data[0], not data[1]."""
        params = [42, "hello"]
        encoded = encode_params_tuple(params, schema=0)
        # Strip the 1-byte count prefix — external count means no count in data.
        # Pass buffer_offset=1 because the count byte was consumed externally.
        data_without_count = encoded[1:]
        decoded, consumed = decode_params_tuple(
            data_without_count, count=2, schema=0, buffer_offset=1
        )
        assert decoded[0] == 42
        assert decoded[1] == "hello"

    def test_decode_with_external_count_v1(self) -> None:
        """External count with V1 schema should skip 4-byte count prefix."""
        params = [42, "hello"]
        encoded = encode_params_tuple(params, schema=1)
        # Strip the 4-byte count prefix.
        # Pass buffer_offset=4 because the count field was consumed externally.
        data_without_count = encoded[4:]
        decoded, consumed = decode_params_tuple(
            data_without_count, count=2, schema=1, buffer_offset=4
        )
        assert decoded[0] == 42
        assert decoded[1] == "hello"

    def test_decode_with_external_count_mixed_types(self) -> None:
        """External count with mixed types should decode correctly."""
        params = [42, 3.14, "text"]
        encoded = encode_params_tuple(params, schema=0)
        data_without_count = encoded[1:]
        decoded, consumed = decode_params_tuple(
            data_without_count, count=3, schema=0, buffer_offset=1
        )
        assert decoded[0] == 42
        assert abs(decoded[1] - 3.14) < 1e-10
        assert decoded[2] == "text"

    def test_decode_with_external_count_zero(self) -> None:
        """External count=0 should return empty list immediately."""
        decoded, consumed = decode_params_tuple(b"\x00" * 8, count=0, schema=0)
        assert decoded == []
        assert consumed == 0


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
        params = [42, "hello"]
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

    def test_first_byte_marker_detection(self) -> None:
        """Marker detection uses first byte, matching Go's byte-by-byte check.

        Go checks the first byte (0xFF -> DONE, 0xEE -> PART). A non-uniform
        marker where only the first byte matches must still be detected.
        """
        # Non-uniform marker: first byte 0xFF, rest different
        data = b"\xff\x00\x00\x00\x00\x00\x00\x00"
        assert decode_row_header(data, 1) == (RowMarker.DONE, 8)

        data = b"\xee\x00\x00\x00\x00\x00\x00\x00"
        assert decode_row_header(data, 1) == (RowMarker.PART, 8)

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
        with pytest.raises(DecodeError, match="Invalid value type"):
            decode_row_header(data, 1)


class TestParamsTupleInvalidType:
    def test_decode_invalid_type_code_raises_decode_error(self) -> None:
        """Invalid type byte (0) in params tuple must raise DecodeError, not ValueError."""
        import pytest

        from dqlitewire.exceptions import DecodeError

        # count=1, type=0 (invalid), padding to 8 bytes
        data = b"\x01\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 8
        with pytest.raises(DecodeError, match="Invalid value type"):
            decode_params_tuple(data)


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
            encode_row_header([0])

    def test_encode_rejects_undefined_type_codes(self) -> None:
        """encode_row_header should reject type codes not defined in ValueType."""
        import pytest

        from dqlitewire.exceptions import EncodeError

        # Type codes 6, 7, 8 are undefined
        for code in [6, 7, 8, 12, 13, 14, 15]:
            with pytest.raises(EncodeError, match="[Ii]nvalid.*type"):
                encode_row_header([code])


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
        values = [42, "hello", 3.14]
        types = [ValueType.INTEGER, ValueType.TEXT, ValueType.FLOAT]
        encoded = encode_row_values(values, types)
        decoded, _ = decode_row_values(encoded, types)
        assert decoded[0] == 42
        assert decoded[1] == "hello"
        assert abs(decoded[2] - 3.14) < 0.0001

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
