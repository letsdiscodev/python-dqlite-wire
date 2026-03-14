"""Tests for primitive type encoding/decoding."""

from datetime import UTC

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.types import (
    decode_blob,
    decode_double,
    decode_int64,
    decode_text,
    decode_uint32,
    decode_uint64,
    decode_value,
    encode_blob,
    encode_double,
    encode_int64,
    encode_text,
    encode_uint32,
    encode_uint64,
    encode_value,
    pad_to_word,
)


class TestUint64:
    def test_encode_zero(self) -> None:
        assert encode_uint64(0) == b"\x00" * 8

    def test_encode_one(self) -> None:
        assert encode_uint64(1) == b"\x01\x00\x00\x00\x00\x00\x00\x00"

    def test_encode_max(self) -> None:
        max_val = 2**64 - 1
        assert encode_uint64(max_val) == b"\xff" * 8

    def test_decode_zero(self) -> None:
        assert decode_uint64(b"\x00" * 8) == 0

    def test_decode_one(self) -> None:
        assert decode_uint64(b"\x01\x00\x00\x00\x00\x00\x00\x00") == 1

    def test_roundtrip(self) -> None:
        values = [0, 1, 255, 256, 65535, 2**32, 2**64 - 1]
        for val in values:
            assert decode_uint64(encode_uint64(val)) == val

    def test_encode_negative_fails(self) -> None:
        with pytest.raises(EncodeError):
            encode_uint64(-1)

    def test_encode_overflow_fails(self) -> None:
        with pytest.raises(EncodeError):
            encode_uint64(2**64)

    def test_decode_too_short_fails(self) -> None:
        with pytest.raises(DecodeError):
            decode_uint64(b"\x00" * 7)


class TestInt64:
    def test_encode_zero(self) -> None:
        assert encode_int64(0) == b"\x00" * 8

    def test_encode_positive(self) -> None:
        assert encode_int64(1) == b"\x01\x00\x00\x00\x00\x00\x00\x00"

    def test_encode_negative(self) -> None:
        assert encode_int64(-1) == b"\xff" * 8

    def test_roundtrip(self) -> None:
        values = [0, 1, -1, 127, -128, 2**63 - 1, -(2**63)]
        for val in values:
            assert decode_int64(encode_int64(val)) == val

    def test_encode_overflow_fails(self) -> None:
        with pytest.raises(EncodeError):
            encode_int64(2**63)

    def test_encode_underflow_fails(self) -> None:
        with pytest.raises(EncodeError):
            encode_int64(-(2**63) - 1)


class TestUint32:
    def test_encode_zero(self) -> None:
        assert encode_uint32(0) == b"\x00\x00\x00\x00"

    def test_encode_one(self) -> None:
        assert encode_uint32(1) == b"\x01\x00\x00\x00"

    def test_roundtrip(self) -> None:
        values = [0, 1, 255, 65535, 2**32 - 1]
        for val in values:
            assert decode_uint32(encode_uint32(val)) == val

    def test_encode_overflow_fails(self) -> None:
        with pytest.raises(EncodeError):
            encode_uint32(2**32)


class TestDouble:
    def test_encode_zero(self) -> None:
        encoded = encode_double(0.0)
        assert len(encoded) == 8

    def test_roundtrip(self) -> None:
        values = [0.0, 1.0, -1.0, 3.14159, 1e100, -1e-100]
        for val in values:
            assert decode_double(encode_double(val)) == val


class TestPadding:
    def test_pad_to_word_aligned(self) -> None:
        assert pad_to_word(0) == 0
        assert pad_to_word(8) == 0
        assert pad_to_word(16) == 0

    def test_pad_to_word_unaligned(self) -> None:
        assert pad_to_word(1) == 7
        assert pad_to_word(2) == 6
        assert pad_to_word(7) == 1
        assert pad_to_word(9) == 7


class TestText:
    def test_encode_empty(self) -> None:
        encoded = encode_text("")
        assert encoded == b"\x00" + b"\x00" * 7  # null + 7 padding

    def test_encode_short(self) -> None:
        encoded = encode_text("hi")
        # "hi" + null = 3 bytes, padded to 8
        assert encoded == b"hi\x00" + b"\x00" * 5

    def test_encode_exact_word(self) -> None:
        # 7 chars + null = 8 bytes exactly
        encoded = encode_text("1234567")
        assert encoded == b"1234567\x00"
        assert len(encoded) == 8

    def test_encode_multi_word(self) -> None:
        # 8 chars + null = 9 bytes, padded to 16
        encoded = encode_text("12345678")
        assert encoded == b"12345678\x00" + b"\x00" * 7
        assert len(encoded) == 16

    def test_decode_empty(self) -> None:
        text, consumed = decode_text(b"\x00" + b"\x00" * 7)
        assert text == ""
        assert consumed == 8

    def test_decode_short(self) -> None:
        text, consumed = decode_text(b"hi\x00" + b"\x00" * 5)
        assert text == "hi"
        assert consumed == 8

    def test_roundtrip(self) -> None:
        strings = ["", "a", "hello", "1234567", "12345678", "hello world", "unicode: \u00e9\u00e8"]
        for s in strings:
            text, _ = decode_text(encode_text(s))
            assert text == s

    def test_roundtrip_unicode_comprehensive(self) -> None:
        """Test various Unicode edge cases."""
        unicode_strings = [
            # Emojis (4-byte UTF-8)
            "Hello 🎉 World",
            "🎉🎊🎁🎂",
            "👨‍👩‍👧‍👦",  # Family emoji (ZWJ sequence)
            "🇫🇷🇺🇸🇯🇵",  # Flag emojis
            # CJK characters
            "中文测试",
            "日本語テスト",
            "한국어 테스트",
            # RTL languages
            "العربية",
            "עברית",
            # Mixed scripts
            "Hello 世界 🌍 مرحبا",
            # Combining characters
            "café",  # Using combining acute accent
            "e\u0301",  # e + combining acute = é
            # Special characters
            "line1\nline2\ttab",
            # Long unicode
            "日" * 100,
            "🎉" * 50,
            # Edge cases
            "\uffff",  # Max BMP
            "\U0001f600",  # Grinning face (outside BMP)
        ]
        for s in unicode_strings:
            text, _ = decode_text(encode_text(s))
            assert text == s, f"Failed for: {repr(s)}"

    def test_null_character_limitation(self) -> None:
        """Null characters are truncated due to null-terminated string format."""
        # Single null becomes empty string
        text, _ = decode_text(encode_text("\x00"))
        assert text == ""

        # Embedded null truncates string
        text, _ = decode_text(encode_text("hello\x00world"))
        assert text == "hello"

    def test_decode_not_terminated_fails(self) -> None:
        with pytest.raises(DecodeError):
            decode_text(b"hello")

    def test_decode_invalid_utf8_raises_decode_error(self) -> None:
        """Invalid UTF-8 bytes should raise DecodeError, not UnicodeDecodeError."""
        # 0xFF 0xFE are invalid UTF-8 start bytes
        data = b"\xff\xfe\x00\x00\x00\x00\x00\x00"
        with pytest.raises(DecodeError, match="UTF-8"):
            decode_text(data)


class TestBlob:
    def test_encode_empty(self) -> None:
        encoded = encode_blob(b"")
        assert len(encoded) == 8  # just the length

    def test_encode_short(self) -> None:
        encoded = encode_blob(b"hi")
        # 8 (length) + 2 (data) + 6 (padding) = 16
        assert len(encoded) == 16
        assert encoded[:8] == b"\x02\x00\x00\x00\x00\x00\x00\x00"
        assert encoded[8:10] == b"hi"

    def test_roundtrip(self) -> None:
        blobs = [b"", b"x", b"hello", b"\x00\x01\x02", b"a" * 100]
        for blob in blobs:
            decoded, _ = decode_blob(encode_blob(blob))
            assert decoded == blob


class TestValue:
    def test_encode_integer(self) -> None:
        encoded, vtype = encode_value(42)
        assert vtype == ValueType.INTEGER
        assert decode_int64(encoded) == 42

    def test_encode_float(self) -> None:
        encoded, vtype = encode_value(3.14)
        assert vtype == ValueType.FLOAT
        assert decode_double(encoded) == 3.14

    def test_encode_text(self) -> None:
        encoded, vtype = encode_value("hello")
        assert vtype == ValueType.TEXT
        text, _ = decode_text(encoded)
        assert text == "hello"

    def test_encode_blob(self) -> None:
        encoded, vtype = encode_value(b"binary")
        assert vtype == ValueType.BLOB
        blob, _ = decode_blob(encoded)
        assert blob == b"binary"

    def test_encode_none(self) -> None:
        encoded, vtype = encode_value(None)
        assert vtype == ValueType.NULL
        assert len(encoded) == 8

    def test_encode_bool_true(self) -> None:
        encoded, vtype = encode_value(True)
        assert vtype == ValueType.BOOLEAN
        assert decode_uint64(encoded) == 1

    def test_encode_bool_false(self) -> None:
        encoded, vtype = encode_value(False)
        assert vtype == ValueType.BOOLEAN
        assert decode_uint64(encoded) == 0

    def test_boolean_uses_uint64_encoding(self) -> None:
        """Go reference encodes BOOLEAN as uint64, not int64."""
        encoded_true, _ = encode_value(True, ValueType.BOOLEAN)
        encoded_false, _ = encode_value(False, ValueType.BOOLEAN)
        # Must be decodable as uint64 (unsigned)
        assert decode_uint64(encoded_true) == 1
        assert decode_uint64(encoded_false) == 0

    def test_decode_integer(self) -> None:
        value, consumed = decode_value(encode_int64(42), ValueType.INTEGER)
        assert value == 42
        assert consumed == 8

    def test_decode_boolean(self) -> None:
        value, consumed = decode_value(encode_uint64(1), ValueType.BOOLEAN)
        assert value is True
        assert consumed == 8

    def test_encode_decode_iso8601(self) -> None:
        """Test ISO8601 datetime string encoding."""
        iso_string = "2024-01-15T10:30:45"
        encoded, vtype = encode_value(iso_string, ValueType.ISO8601)
        assert vtype == ValueType.ISO8601
        decoded, _ = decode_value(encoded, ValueType.ISO8601)
        assert decoded == iso_string

    def test_encode_decode_unixtime(self) -> None:
        """Test Unix timestamp encoding."""
        timestamp = 1705312245  # 2024-01-15 10:30:45 UTC
        encoded, vtype = encode_value(timestamp, ValueType.UNIXTIME)
        assert vtype == ValueType.UNIXTIME
        decoded, _ = decode_value(encoded, ValueType.UNIXTIME)
        assert decoded == timestamp

    def test_boolean_roundtrip(self) -> None:
        """Test boolean encoding/decoding."""
        for val in [True, False]:
            encoded, vtype = encode_value(val)
            assert vtype == ValueType.BOOLEAN
            decoded, _ = decode_value(encoded, ValueType.BOOLEAN)
            assert decoded == val

    def test_float_edge_cases(self) -> None:
        """Test float edge cases."""
        floats = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            3.14159265358979,
            1e10,
            1e-10,
            float("inf"),
            float("-inf"),
        ]
        for val in floats:
            encoded, vtype = encode_value(val)
            assert vtype == ValueType.FLOAT
            decoded, _ = decode_value(encoded, ValueType.FLOAT)
            assert decoded == val

    def test_nan_rejected(self) -> None:
        """NaN should be rejected since SQLite and dqlite don't support it."""
        with pytest.raises(EncodeError, match="NaN"):
            encode_double(float("nan"))

        with pytest.raises(EncodeError, match="NaN"):
            encode_value(float("nan"))

    def test_integer_edge_cases(self) -> None:
        """Test integer edge cases."""
        integers = [
            0,
            1,
            -1,
            127,
            128,
            255,
            256,
            32767,
            32768,
            65535,
            65536,
            2147483647,  # Max 32-bit signed
            2147483648,
            -2147483648,  # Min 32-bit signed
            9223372036854775807,  # Max 64-bit signed
            -9223372036854775808,  # Min 64-bit signed
        ]
        for val in integers:
            encoded, vtype = encode_value(val)
            assert vtype == ValueType.INTEGER
            decoded, _ = decode_value(encoded, ValueType.INTEGER)
            assert decoded == val

    def test_datetime_auto_detection(self) -> None:
        """datetime.datetime should auto-detect as ISO8601."""
        from datetime import datetime

        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        encoded, vtype = encode_value(dt)
        assert vtype == ValueType.ISO8601
        # Should encode as a text string in Go-compatible format
        decoded, _ = decode_value(encoded, ValueType.ISO8601)
        assert "2024-01-15" in decoded
        assert "10:30:45" in decoded

    def test_naive_datetime_includes_utc_offset(self) -> None:
        """Naive datetime should include +00:00 offset to match Go's format."""
        from datetime import datetime

        dt = datetime(2024, 1, 15, 10, 30, 45)  # naive, no tzinfo
        encoded, vtype = encode_value(dt)
        assert vtype == ValueType.ISO8601
        decoded, _ = decode_value(encoded, ValueType.ISO8601)
        assert decoded.endswith("+00:00")

    def test_datetime_with_microseconds(self) -> None:
        """Datetime with microseconds should include them in the format."""
        from datetime import datetime

        dt = datetime(2024, 1, 15, 10, 30, 45, 123456, tzinfo=UTC)
        encoded, vtype = encode_value(dt)
        decoded, _ = decode_value(encoded, ValueType.ISO8601)
        assert ".123456" in decoded

    def test_encode_value_unsupported_type_raises(self) -> None:
        """Unsupported Python types should raise EncodeError."""
        with pytest.raises(EncodeError, match="Cannot infer type"):
            encode_value({"key": "value"})

    def test_decode_value_unknown_type_raises(self) -> None:
        """Unknown ValueType should raise DecodeError."""
        with pytest.raises(DecodeError, match="Unknown value type"):
            decode_value(b"\x00" * 8, 99)  # type: ignore[arg-type]

    def test_encode_value_unknown_type_raises(self) -> None:
        """Unknown explicit ValueType should raise EncodeError."""
        with pytest.raises(EncodeError, match="Unknown value type"):
            encode_value(42, 99)  # type: ignore[arg-type]

    def test_decode_int64_short_data(self) -> None:
        with pytest.raises(DecodeError):
            decode_int64(b"\x00" * 7)

    def test_decode_uint32_short_data(self) -> None:
        with pytest.raises(DecodeError):
            decode_uint32(b"\x00" * 3)

    def test_decode_double_short_data(self) -> None:
        with pytest.raises(DecodeError):
            decode_double(b"\x00" * 7)

    def test_decode_null(self) -> None:
        value, consumed = decode_value(b"\x00" * 8, ValueType.NULL)
        assert value is None
        assert consumed == 8
