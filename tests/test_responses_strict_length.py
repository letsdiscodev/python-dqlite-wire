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
from dqlitewire.messages.responses import DbResponse, EmptyResponse, ResultResponse, StmtResponse


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


class TestStmtResponseStrictLength:
    """StmtResponse carries db_id + stmt_id + num_params (+ optional
    tail_offset for schema>=1). Body is exactly 16 bytes (schema 0) or
    24 bytes (schema 1). Trailing bytes had been silently accepted;
    sibling responses reject them.
    """

    @staticmethod
    def _body_schema0(db_id: int = 1, stmt_id: int = 42, num_params: int = 3) -> bytes:
        return (
            db_id.to_bytes(4, "little")
            + stmt_id.to_bytes(4, "little")
            + num_params.to_bytes(8, "little")
        )

    @staticmethod
    def _body_schema1(
        db_id: int = 1, stmt_id: int = 42, num_params: int = 3, tail_offset: int = 0
    ) -> bytes:
        return TestStmtResponseStrictLength._body_schema0(
            db_id, stmt_id, num_params
        ) + tail_offset.to_bytes(8, "little")

    def test_short_body_rejected_schema0(self) -> None:
        with pytest.raises(DecodeError, match=r"StmtResponse schema=0 body must be exactly 16"):
            StmtResponse.decode_body(b"\x00" * 15, schema=0)

    def test_exact_length_accepted_schema0(self) -> None:
        msg = StmtResponse.decode_body(self._body_schema0(), schema=0)
        assert msg.db_id == 1
        assert msg.stmt_id == 42
        assert msg.num_params == 3
        assert msg.tail_offset is None

    def test_trailing_bytes_rejected_schema0(self) -> None:
        """Previous decoder silently accepted extra bytes; must now
        raise so a conforming StmtResponse round-trips exactly."""
        body = self._body_schema0() + b"\x01"
        with pytest.raises(DecodeError, match=r"StmtResponse schema=0 body must be exactly 16"):
            StmtResponse.decode_body(body, schema=0)

    def test_short_body_rejected_schema1(self) -> None:
        with pytest.raises(DecodeError, match=r"StmtResponse schema=1 body must be exactly 24"):
            StmtResponse.decode_body(b"\x00" * 23, schema=1)

    def test_exact_length_accepted_schema1(self) -> None:
        msg = StmtResponse.decode_body(self._body_schema1(tail_offset=7), schema=1)
        assert msg.db_id == 1
        assert msg.stmt_id == 42
        assert msg.num_params == 3
        assert msg.tail_offset == 7

    def test_trailing_bytes_rejected_schema1(self) -> None:
        body = self._body_schema1() + b"\x02"
        with pytest.raises(DecodeError, match=r"StmtResponse schema=1 body must be exactly 24"):
            StmtResponse.decode_body(body, schema=1)


class TestWelcomeResponseStrictLength:
    """Body is uint64(heartbeat_timeout) — exactly 8 bytes."""

    def test_short_body_rejected(self) -> None:
        from dqlitewire.messages.responses import WelcomeResponse

        with pytest.raises(DecodeError, match=r"WelcomeResponse body must be exactly 8"):
            WelcomeResponse.decode_body(b"\x00" * 7)

    def test_exact_length_accepted(self) -> None:
        from dqlitewire.messages.responses import WelcomeResponse

        msg = WelcomeResponse.decode_body((42).to_bytes(8, "little"))
        assert msg.heartbeat_timeout == 42

    def test_trailing_bytes_rejected(self) -> None:
        from dqlitewire.messages.responses import WelcomeResponse

        with pytest.raises(DecodeError, match=r"WelcomeResponse body must be exactly 8"):
            WelcomeResponse.decode_body(b"\x00" * 9)


class TestServersResponseStrictLength:
    """Variable-length body: ``uint64 count`` then ``count`` × (id, text
    address, role). Trailing bytes after the last node had been silently
    dropped.
    """

    @staticmethod
    def _single_node_body(addr: str = "1.2.3.4:9001") -> bytes:
        from dqlitewire.types import encode_text, encode_uint64

        return (
            encode_uint64(1)  # count
            + encode_uint64(1)  # node_id
            + encode_text(addr)  # address (padded)
            + encode_uint64(0)  # role = voter
        )

    def test_empty_list_exact_round_trip(self) -> None:
        from dqlitewire.messages.responses import ServersResponse
        from dqlitewire.types import encode_uint64

        msg = ServersResponse.decode_body(encode_uint64(0))
        assert msg.nodes == []

    def test_single_node_exact_round_trip(self) -> None:
        from dqlitewire.messages.responses import ServersResponse

        msg = ServersResponse.decode_body(self._single_node_body())
        assert len(msg.nodes) == 1
        assert msg.nodes[0].address == "1.2.3.4:9001"

    def test_multi_node_exact_round_trip(self) -> None:
        """Offset accumulation through multiple iterations — exercises
        the per-node loop boundary condition that the single-node test
        cannot reach."""
        from dqlitewire.messages.responses import ServersResponse
        from dqlitewire.types import encode_text, encode_uint64

        body = encode_uint64(3)
        for i in range(3):
            body += encode_uint64(i)
            body += encode_text(f"node{i}.example:900{i}")
            body += encode_uint64(0)
        msg = ServersResponse.decode_body(body)
        assert [n.address for n in msg.nodes] == [
            "node0.example:9000",
            "node1.example:9001",
            "node2.example:9002",
        ]

    def test_trailing_bytes_rejected(self) -> None:
        from dqlitewire.messages.responses import ServersResponse

        body = self._single_node_body() + b"\x01"
        with pytest.raises(DecodeError, match=r"ServersResponse has 1 trailing byte"):
            ServersResponse.decode_body(body)

    def test_trailing_word_rejected(self) -> None:
        from dqlitewire.messages.responses import ServersResponse

        body = self._single_node_body() + b"\x00" * 8
        with pytest.raises(DecodeError, match=r"ServersResponse has 8 trailing byte"):
            ServersResponse.decode_body(body)

    def test_empty_list_trailing_bytes_rejected(self) -> None:
        """Count=0 with trailing bytes must still raise — otherwise a
        server could hide bytes behind an empty list."""
        from dqlitewire.messages.responses import ServersResponse
        from dqlitewire.types import encode_uint64

        body = encode_uint64(0) + b"\x00" * 8
        with pytest.raises(DecodeError, match=r"ServersResponse has 8 trailing byte"):
            ServersResponse.decode_body(body)


class TestMetadataResponseStrictLength:
    """Body is two uint64s (failure_domain + weight) — exactly 16 bytes."""

    def test_short_body_rejected(self) -> None:
        from dqlitewire.messages.responses import MetadataResponse

        with pytest.raises(DecodeError, match=r"MetadataResponse body must be exactly 16"):
            MetadataResponse.decode_body(b"\x00" * 15)

    def test_exact_length_accepted(self) -> None:
        from dqlitewire.messages.responses import MetadataResponse

        body = (3).to_bytes(8, "little") + (7).to_bytes(8, "little")
        msg = MetadataResponse.decode_body(body)
        assert msg.failure_domain == 3
        assert msg.weight == 7

    def test_trailing_bytes_rejected(self) -> None:
        from dqlitewire.messages.responses import MetadataResponse

        with pytest.raises(DecodeError, match=r"MetadataResponse body must be exactly 16"):
            MetadataResponse.decode_body(b"\x00" * 17)
