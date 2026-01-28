"""Tests for tuple encoding/decoding."""

from dqlitewire.constants import ValueType
from dqlitewire.tuples import (
    decode_params_tuple,
    decode_row_header,
    decode_row_values,
    encode_params_tuple,
    encode_row_header,
    encode_row_values,
)


class TestParamsTuple:
    def test_encode_empty(self) -> None:
        encoded = encode_params_tuple([])
        # Empty params: count=0 + 7 bytes padding
        assert encoded == b"\x00" * 8

    def test_encode_single_integer(self) -> None:
        encoded = encode_params_tuple([42])
        # Header: count(1) + type(1) + padding(6) = 8, value = 8, total = 16
        assert len(encoded) == 16
        assert encoded[0] == 1  # count

    def test_encode_multiple_integers(self) -> None:
        encoded = encode_params_tuple([1, 2, 3])
        # Header: count(1) + 3 types + padding(4) = 8, 3 values = 24, total = 32
        assert len(encoded) == 32
        assert encoded[0] == 3  # count

    def test_encode_mixed_types(self) -> None:
        params = [42, "hello", 3.14, None, b"blob"]
        encoded = encode_params_tuple(params)
        assert len(encoded) > 0
        assert encoded[0] == 5  # count

    def test_decode_empty(self) -> None:
        # Empty params with proper header
        data = b"\x00" * 8
        values, consumed = decode_params_tuple(data)
        assert values == []
        assert consumed == 8

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
