"""Fixed-length response decoders must reject trailing bytes.

Strict-parse peers in this module (``LeaderRequest.decode_body``,
``StmtResponse.decode_body``) assert exact body lengths. The three
messages audited here previously accepted ``len >= expected``,
silently ignoring trailing bytes — asymmetric with peers and
permissive in a way that masks frame-corruption.

Peers of ISSUE-298 / ISSUE-299.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import DbResponse, EmptyResponse, ResultResponse


class TestEmptyResponseStrictLength:
    def test_short_body_rejected(self) -> None:
        with pytest.raises(DecodeError, match="EmptyResponse body must be exactly 8 bytes"):
            EmptyResponse.decode_body(b"\x00" * 7)

    def test_exact_length_accepted(self) -> None:
        msg = EmptyResponse.decode_body(b"\x00" * 8)
        assert isinstance(msg, EmptyResponse)

    def test_trailing_bytes_rejected(self) -> None:
        """16-byte body: decoders used to silently discard the
        trailing 8 bytes. Must now raise.
        """
        with pytest.raises(DecodeError, match="EmptyResponse body must be exactly 8 bytes"):
            EmptyResponse.decode_body(b"\x00" * 16)

    def test_trailing_zero_bytes_still_rejected(self) -> None:
        """Trailing zeros look innocuous but must still fail — the
        decoder should not need to introspect the padding."""
        with pytest.raises(DecodeError, match="EmptyResponse body must be exactly 8 bytes"):
            EmptyResponse.decode_body(b"\x00" * 9)

    def test_reserved_nonzero_still_rejected(self) -> None:
        """Length check comes first; reserved-field check must still
        apply on exactly-8-byte bodies."""
        with pytest.raises(DecodeError, match="reserved field must be 0"):
            EmptyResponse.decode_body(b"\x01" + b"\x00" * 7)


class TestDbResponseStrictLength:
    def test_short_body_rejected(self) -> None:
        with pytest.raises(DecodeError, match="DbResponse body must be exactly 8 bytes"):
            DbResponse.decode_body(b"\x00" * 7)

    def test_exact_length_accepted(self) -> None:
        msg = DbResponse.decode_body(b"\x05\x00\x00\x00" + b"\x00" * 4)
        assert isinstance(msg, DbResponse)
        assert msg.db_id == 5

    def test_trailing_bytes_rejected(self) -> None:
        with pytest.raises(DecodeError, match="DbResponse body must be exactly 8 bytes"):
            DbResponse.decode_body(b"\x00" * 16)


class TestResultResponseStrictLength:
    def test_short_body_rejected_with_type_name(self) -> None:
        """Error message names ``ResultResponse``, not just
        ``uint64`` — so operators reading logs can trace the
        framed message."""
        with pytest.raises(DecodeError, match="ResultResponse body must be exactly 16 bytes"):
            ResultResponse.decode_body(b"\x00" * 15)

    def test_exact_length_accepted(self) -> None:
        body = (42).to_bytes(8, "little") + (7).to_bytes(8, "little")
        msg = ResultResponse.decode_body(body)
        assert msg.last_insert_id == 42
        assert msg.rows_affected == 7

    def test_trailing_bytes_rejected(self) -> None:
        with pytest.raises(DecodeError, match="ResultResponse body must be exactly 16 bytes"):
            ResultResponse.decode_body(b"\x00" * 17)
