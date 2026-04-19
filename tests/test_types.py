"""Tests for primitive type encoding/decoding."""

from datetime import UTC

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.types import (
    _MAX_BLOB_SIZE,
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

    def test_nan_roundtrip(self) -> None:
        """NaN should roundtrip through encode/decode, matching Go behavior.

        The Go reference implementation and C dqlite server pass NaN through
        without checks. A wire protocol codec should faithfully transport all
        IEEE 754 values.
        """
        import math

        encoded = encode_double(float("nan"))
        assert len(encoded) == 8
        result = decode_double(encoded)
        assert math.isnan(result)

    def test_positive_infinity_roundtrip(self) -> None:
        """Positive infinity should roundtrip, matching Go behavior."""
        import math

        encoded = encode_double(float("inf"))
        result = decode_double(encoded)
        assert math.isinf(result) and result > 0

    def test_negative_infinity_roundtrip(self) -> None:
        """Negative infinity should roundtrip, matching Go behavior."""
        import math

        encoded = encode_double(float("-inf"))
        result = decode_double(encoded)
        assert math.isinf(result) and result < 0


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

    def test_embedded_null_raises_encode_error(self) -> None:
        """Strings with embedded null bytes should be rejected by encode_text."""
        with pytest.raises(EncodeError, match="embedded null byte"):
            encode_text("hello\x00world")

        with pytest.raises(EncodeError, match="embedded null byte"):
            encode_text("\x00")

    @pytest.mark.parametrize(
        "bad",
        [None, 42, b"x", 3.14, ["x"], bytearray(b"x"), memoryview(b"x")],
    )
    def test_non_str_raises_encode_error(self, bad: object) -> None:
        """234: Non-string inputs must raise EncodeError, not raw TypeError.

        Callers following the library's documented exception contract
        (``except EncodeError``) would otherwise miss these failures.
        """
        with pytest.raises(EncodeError, match="expected str"):
            encode_text(bad)  # type: ignore[arg-type]

    def test_lone_surrogate_raises_encode_error(self) -> None:
        """234: Lone UTF-16 surrogates are legal Python str but not UTF-8-encodable.

        Must raise EncodeError (not the stdlib's UnicodeEncodeError).
        """
        with pytest.raises(EncodeError, match="invalid UTF-8"):
            encode_text("\ud800")

    def test_decode_not_terminated_fails(self) -> None:
        with pytest.raises(DecodeError):
            decode_text(b"hello")

    def test_decode_truncated_padding_raises(self) -> None:
        """decode_text with null terminator found but insufficient padding must raise."""
        # "hello" + null = 6 bytes, needs 2 padding bytes to reach 8
        # But we only provide the 6 bytes (null found, but padding truncated)
        data = b"hello\x00"
        with pytest.raises(DecodeError, match="[Nn]ot enough data for text"):
            decode_text(data)

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

    def test_decode_blob_too_short_for_length(self) -> None:
        """decode_blob with < 8 bytes should raise DecodeError."""
        with pytest.raises(DecodeError, match="Not enough data for blob length"):
            decode_blob(b"\x00" * 7)

    def test_decode_blob_truncated_data(self) -> None:
        """decode_blob with valid length prefix but insufficient data."""
        data = encode_uint64(100) + b"\x00" * 10  # claims 100 bytes, only has 10
        with pytest.raises(DecodeError, match="Not enough data for blob"):
            decode_blob(data)

    def test_decode_blob_rejects_length_beyond_cap(self) -> None:
        """Crafted buffer claiming a length beyond the per-field cap must be
        rejected before the decoder allocates or does total-size arithmetic
        with the attacker-controlled length."""
        oversized = _MAX_BLOB_SIZE + 1
        data = encode_uint64(oversized)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            decode_blob(data)

    def test_decode_blob_accepts_length_at_cap(self) -> None:
        """Length exactly at the cap is still accepted; the body check is
        what fires on a truncated buffer."""
        # Claim exactly the cap but provide a short buffer. The cap check
        # must pass; the later "not enough data" check is what rejects.
        data = encode_uint64(_MAX_BLOB_SIZE) + b"\x00" * 8
        with pytest.raises(DecodeError, match="Not enough data for blob"):
            decode_blob(data)

    def test_encode_blob_rejects_oversize(self) -> None:
        """encode_blob mirrors the decode cap so callers fail fast on an
        accidental giant bytes input instead of burning allocations."""
        with pytest.raises(EncodeError, match="exceeds maximum"):
            encode_blob(b"\x00" * (_MAX_BLOB_SIZE + 1))


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
        """BOOLEAN encodes as uint64 (0 or 1), matching C server's uint64_t."""
        encoded_true, _ = encode_value(True, ValueType.BOOLEAN)
        encoded_false, _ = encode_value(False, ValueType.BOOLEAN)
        assert decode_uint64(encoded_true) == 1
        assert decode_uint64(encoded_false) == 0

    def test_boolean_explicit_rejects_non_bool_non_int(self) -> None:
        """encode_value with explicit BOOLEAN should reject non-bool/int types."""
        import pytest

        with pytest.raises(EncodeError, match="BOOLEAN"):
            encode_value("hello", ValueType.BOOLEAN)
        with pytest.raises(EncodeError, match="BOOLEAN"):
            encode_value([1, 2], ValueType.BOOLEAN)
        with pytest.raises(EncodeError, match="BOOLEAN"):
            encode_value({"key": "val"}, ValueType.BOOLEAN)

    @pytest.mark.parametrize(
        "payload",
        [b"hello", bytearray(b"hello"), memoryview(b"hello")],
    )
    def test_blob_roundtrip_bytes_like(self, payload: object) -> None:
        """All three bytes-like inputs encode identically and decode
        back to ``bytes``."""
        encoded, vtype = encode_value(payload, ValueType.BLOB)
        assert vtype == ValueType.BLOB
        decoded, consumed = decode_value(encoded, ValueType.BLOB)
        assert decoded == b"hello"
        assert isinstance(decoded, bytes)
        assert consumed == len(encoded)

    def test_boolean_rejects_arbitrary_int(self) -> None:
        """BOOLEAN requires exact bool or 0/1 — arbitrary ints are rejected.

        Previously any truthy int was silently coerced to True via
        ``1 if value else 0``. That made the round-trip lossy: encode(5,
        BOOLEAN) → wire 1 → decode → True. Callers that mean "store the
        integer 5" should use INTEGER; callers that mean "bool" should
        pass bool.
        """
        import pytest

        with pytest.raises(EncodeError, match="BOOLEAN"):
            encode_value(5, ValueType.BOOLEAN)
        with pytest.raises(EncodeError, match="BOOLEAN"):
            encode_value(-1, ValueType.BOOLEAN)
        with pytest.raises(EncodeError, match="BOOLEAN"):
            encode_value(2, ValueType.BOOLEAN)

        # 0 and 1 remain acceptable as a pragmatic escape for callers
        # working with raw column values.
        encoded_zero, _ = encode_value(0, ValueType.BOOLEAN)
        encoded_one, _ = encode_value(1, ValueType.BOOLEAN)
        assert decode_uint64(encoded_zero) == 0
        assert decode_uint64(encoded_one) == 1

    def test_decode_integer(self) -> None:
        value, consumed = decode_value(encode_int64(42), ValueType.INTEGER)
        assert value == 42
        assert consumed == 8

    def test_decode_boolean(self) -> None:
        value, consumed = decode_value(encode_int64(1), ValueType.BOOLEAN)
        assert value is True
        assert consumed == 8

    def test_decode_boolean_zero(self) -> None:
        value, consumed = decode_value(encode_int64(0), ValueType.BOOLEAN)
        assert value is False
        assert consumed == 8

    def test_decode_boolean_rejects_non_bit(self) -> None:
        """Decoder is symmetric with encoder: only {0, 1} are valid BOOLEAN
        wire values. Silently coercing any uint64 via ``bool(...)`` would
        make round-trips lossy (e.g., uint64=2 → True → re-encodes as 1)
        and would mask protocol violations that we want to surface.
        """
        from dqlitewire.exceptions import DecodeError

        for bad in (2, 3, 255, 2**63 - 1):
            with pytest.raises(DecodeError, match="BOOLEAN"):
                decode_value(encode_int64(bad), ValueType.BOOLEAN)

    def test_encode_decode_iso8601_roundtrips_as_text(self) -> None:
        """ISO8601 is stored as text at the wire level — datetime conversion
        lives in the driver/DBAPI layer, matching C and Go's split.
        """
        for iso in (
            "2024-01-15 10:30:45+00:00",
            "2024-01-15T10:30:45+00:00",
            "2024-01-15 10:30:45.123456+00:00",
            "2024-01-15 10:30:45",
            "2024-01-15",
            "",
        ):
            encoded, vtype = encode_value(iso, ValueType.ISO8601)
            assert vtype == ValueType.ISO8601
            decoded, _ = decode_value(encoded, ValueType.ISO8601)
            assert decoded == iso

    def test_encode_decode_unixtime(self) -> None:
        """UNIXTIME should decode to raw int, matching Go's getInt64()."""
        timestamp = 1705312245
        encoded, vtype = encode_value(timestamp, ValueType.UNIXTIME)
        assert vtype == ValueType.UNIXTIME
        decoded, _ = decode_value(encoded, ValueType.UNIXTIME)
        assert isinstance(decoded, int)
        assert decoded == timestamp

    def test_encode_decode_unixtime_zero(self) -> None:
        """Unix timestamp 0 should decode to integer 0."""
        encoded, _ = encode_value(0, ValueType.UNIXTIME)
        decoded, _ = decode_value(encoded, ValueType.UNIXTIME)
        assert isinstance(decoded, int)
        assert decoded == 0

    def test_encode_decode_unixtime_negative(self) -> None:
        """Negative Unix timestamps should decode to negative int."""
        encoded, _ = encode_value(-86400, ValueType.UNIXTIME)
        decoded, _ = decode_value(encoded, ValueType.UNIXTIME)
        assert isinstance(decoded, int)
        assert decoded == -86400

    def test_unixtime_extreme_values_roundtrip(self) -> None:
        """Extreme UNIXTIME values should roundtrip as raw ints (no datetime conversion)."""
        for timestamp in [2**63 - 1, -(2**63)]:
            data = encode_int64(timestamp)
            decoded, consumed = decode_value(data, ValueType.UNIXTIME)
            assert isinstance(decoded, int)
            assert decoded == timestamp
            assert consumed == 8

    def test_unixtime_roundtrip_preserves_type(self) -> None:
        """UNIXTIME encode→decode should return an int, matching Go's getInt64().

        Previously, decode returned a datetime.datetime, which when re-encoded
        would auto-infer ISO8601 — changing the value type and wire format.
        """
        timestamp = 1710000000
        encoded, vtype = encode_value(timestamp, ValueType.UNIXTIME)
        assert vtype == ValueType.UNIXTIME

        decoded, consumed = decode_value(encoded, ValueType.UNIXTIME)
        assert consumed == 8
        assert isinstance(decoded, int)
        assert decoded == timestamp

        # Re-encoding with explicit UNIXTIME should produce identical bytes
        re_encoded, re_vtype = encode_value(decoded, ValueType.UNIXTIME)
        assert re_vtype == ValueType.UNIXTIME
        assert re_encoded == encoded

    def test_encode_datetime_with_explicit_unixtime_raises_encode_error(self) -> None:
        """Passing a datetime with ValueType.UNIXTIME should raise EncodeError, not TypeError."""
        import datetime

        dt = datetime.datetime(2024, 1, 15, 12, 30, 0, tzinfo=datetime.UTC)
        with pytest.raises(EncodeError, match="Expected int for UNIXTIME"):
            encode_value(dt, ValueType.UNIXTIME)

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
        ]
        for val in floats:
            encoded, vtype = encode_value(val)
            assert vtype == ValueType.FLOAT
            decoded, _ = decode_value(encoded, ValueType.FLOAT)
            assert decoded == val

    def test_nan_accepted(self) -> None:
        """NaN should be accepted, matching Go reference implementation."""
        import math

        encoded, vtype = encode_value(float("nan"))
        assert vtype == ValueType.FLOAT
        decoded, _ = decode_value(encoded, ValueType.FLOAT)
        assert math.isnan(decoded)

    def test_infinity_accepted(self) -> None:
        """Infinity should be accepted, matching Go reference implementation."""
        import math

        for val in (float("inf"), float("-inf")):
            encoded, vtype = encode_value(val)
            assert vtype == ValueType.FLOAT
            decoded, _ = decode_value(encoded, ValueType.FLOAT)
            assert math.isinf(decoded)
            assert (decoded > 0) == (val > 0)

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

    def test_encode_value_rejects_datetime(self) -> None:
        """Wire codec no longer auto-infers datetime — driver/DBAPI must
        convert to str (for ISO8601) or int (for UNIXTIME)."""
        from datetime import datetime

        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        with pytest.raises(EncodeError, match="Cannot infer wire type"):
            encode_value(dt)

    def test_encode_value_unsupported_type_raises(self) -> None:
        """Unsupported Python types should raise EncodeError."""
        with pytest.raises(EncodeError, match="Cannot infer wire type"):
            encode_value({"key": "value"})

    def test_decode_value_unknown_type_raises(self) -> None:
        """Unknown ValueType should raise DecodeError."""
        with pytest.raises(DecodeError, match="Unknown value type"):
            decode_value(b"\x00" * 8, 99)  # type: ignore[arg-type]

    def test_encode_value_unknown_type_raises(self) -> None:
        """Unknown explicit ValueType should raise EncodeError."""
        with pytest.raises(EncodeError, match="Unknown value type"):
            encode_value(42, 99)  # type: ignore[arg-type]

    def test_encode_value_float_rejects_string(self) -> None:
        with pytest.raises(EncodeError, match="FLOAT"):
            encode_value("hello", ValueType.FLOAT)

    def test_encode_value_float_rejects_bool(self) -> None:
        with pytest.raises(EncodeError, match="FLOAT"):
            encode_value(True, ValueType.FLOAT)

    def test_encode_value_float_accepts_int(self) -> None:
        encoded, vtype = encode_value(42, ValueType.FLOAT)
        assert vtype == ValueType.FLOAT
        decoded, _ = decode_value(encoded, ValueType.FLOAT)
        assert decoded == 42.0

    def test_encode_value_text_rejects_int(self) -> None:
        with pytest.raises(EncodeError, match="TEXT"):
            encode_value(42, ValueType.TEXT)

    def test_encode_value_iso8601_rejects_int(self) -> None:
        with pytest.raises(EncodeError, match="ISO8601"):
            encode_value(42, ValueType.ISO8601)

    def test_encode_value_blob_rejects_string(self) -> None:
        with pytest.raises(EncodeError, match="BLOB"):
            encode_value("hello", ValueType.BLOB)

    def test_encode_value_blob_accepts_bytearray(self) -> None:
        encoded, vtype = encode_value(bytearray(b"\x01\x02"), ValueType.BLOB)
        assert vtype == ValueType.BLOB
        decoded, _ = decode_value(encoded, ValueType.BLOB)
        assert decoded == b"\x01\x02"

    def test_encode_value_blob_accepts_memoryview(self) -> None:
        encoded, vtype = encode_value(memoryview(b"\x03\x04"), ValueType.BLOB)
        assert vtype == ValueType.BLOB
        decoded, _ = decode_value(encoded, ValueType.BLOB)
        assert decoded == b"\x03\x04"

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

    def test_decode_null_truncated(self) -> None:
        with pytest.raises(DecodeError, match="8 bytes"):
            decode_value(b"\x00" * 4, ValueType.NULL)

    def test_decode_null_empty(self) -> None:
        with pytest.raises(DecodeError, match="8 bytes"):
            decode_value(b"", ValueType.NULL)

    def test_encode_value_null_type_with_non_none_raises(self) -> None:
        """Explicit ValueType.NULL with a non-None value should raise EncodeError."""
        with pytest.raises(EncodeError, match="Cannot encode non-None value"):
            encode_value(42, ValueType.NULL)

    def test_encode_value_null_type_with_string_raises(self) -> None:
        with pytest.raises(EncodeError, match="Cannot encode non-None value"):
            encode_value("hello", ValueType.NULL)

    def test_encode_value_none_with_explicit_integer_raises(self) -> None:
        """None with explicit ValueType.INTEGER should raise EncodeError."""
        with pytest.raises(EncodeError, match="Cannot encode None with explicit type"):
            encode_value(None, ValueType.INTEGER)

    def test_encode_value_none_with_explicit_text_raises(self) -> None:
        """None with explicit ValueType.TEXT should raise EncodeError."""
        with pytest.raises(EncodeError, match="Cannot encode None with explicit type"):
            encode_value(None, ValueType.TEXT)

    def test_encode_value_none_with_explicit_null_ok(self) -> None:
        """None with explicit ValueType.NULL should succeed."""
        encoded, vtype = encode_value(None, ValueType.NULL)
        assert vtype == ValueType.NULL
        assert encoded == b"\x00" * 8

    def test_encode_value_none_without_type_ok(self) -> None:
        """None without explicit type should succeed (infer NULL)."""
        encoded, vtype = encode_value(None)
        assert vtype == ValueType.NULL
        assert encoded == b"\x00" * 8

    def test_encode_value_bool_as_explicit_integer(self) -> None:
        """Bool with explicit ValueType.INTEGER should coerce to int."""
        encoded, vtype = encode_value(True, ValueType.INTEGER)
        assert vtype == ValueType.INTEGER
        assert decode_int64(encoded) == 1

    def test_encode_value_false_as_explicit_integer(self) -> None:
        encoded, vtype = encode_value(False, ValueType.INTEGER)
        assert vtype == ValueType.INTEGER
        assert decode_int64(encoded) == 0


class TestEncodeValueUnsupportedTypes:
    """Pin the rejection contract for Python types the wire codec does
    not natively understand. The codec accepts only bool / int / float /
    str / bytes / None; anything else must surface as ``EncodeError`` so
    callers see a clear driver-level error rather than a nonsense wire
    frame. These tests lock the current behaviour so a future refactor
    that adds implicit Decimal / numpy support is an intentional
    change — not a silent one.
    """

    def test_decimal_rejected(self) -> None:
        from decimal import Decimal

        with pytest.raises(EncodeError, match="Cannot infer wire type"):
            encode_value(Decimal("3.14"))

    def test_fraction_rejected(self) -> None:
        from fractions import Fraction

        with pytest.raises(EncodeError, match="Cannot infer wire type"):
            encode_value(Fraction(1, 3))

    def test_complex_rejected(self) -> None:
        with pytest.raises(EncodeError, match="Cannot infer wire type"):
            encode_value(complex(1, 2))

    def test_plain_object_rejected(self) -> None:
        with pytest.raises(EncodeError, match="Cannot infer wire type"):
            encode_value(object())

    def test_index_only_class_rejected(self) -> None:
        """An object with ``__index__`` but no ``__int__`` or bool-ness
        is NOT treated as int by ``isinstance(value, int)``; confirm
        rejection so the contract stays tight.
        """

        class OnlyIndex:
            def __index__(self) -> int:
                return 42

        with pytest.raises(EncodeError, match="Cannot infer wire type"):
            encode_value(OnlyIndex())
