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
        data = b"\x00" * 8 + b"\xAA" * 8  # header + trailing data
        values, consumed = decode_params_tuple(data, schema=0)
        assert values == []
        assert consumed == 8  # one word consumed for the header

    def test_decode_zero_count_v1_consumes_header(self) -> None:
        """Decoding a V1 tuple with count=0 should consume the header word."""
        import struct

        # V1: uint32 count=0 at bytes 0-3, rest padding = 8 bytes total
        data = struct.pack("<I", 0) + b"\x00" * 4 + b"\xBB" * 8
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


class TestParamsTupleErrors:
    def test_decode_insufficient_data_for_header(self) -> None:
        """Data shorter than 8 bytes should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError

        with pytest.raises(DecodeError, match="Not enough data"):
            decode_params_tuple(b"\x01\x02\x03")

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

    def test_roundtrip_with_null(self) -> None:
        values = [42, None, "test"]
        types = [ValueType.INTEGER, ValueType.NULL, ValueType.TEXT]
        encoded = encode_row_values(values, types)
        decoded, _ = decode_row_values(encoded, types)
        assert decoded == values
