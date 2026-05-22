"""Fixed-length response decoders must reject trailing bytes.

Strict-parse peers in this module (``LeaderRequest.decode_body``,
``StmtResponse.decode_body``) assert exact body lengths. The three
messages audited here previously accepted ``len >= expected``,
silently ignoring trailing bytes — asymmetric with peers and
permissive in a way that masks frame-corruption.
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

    def test_reserved_nonzero_accepted(self) -> None:
        """Match Go's ``response.getUint64()`` discard
        (``internal/protocol/response.go:186``). The reserved field
        is documented as unused; a future server reusing it must not
        break Python clients while Go clients continue working.
        """
        msg = EmptyResponse.decode_body(b"\x01" + b"\x00" * 7)
        assert isinstance(msg, EmptyResponse)


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
        cannot reach. node_id starts at 1 because raft reserves id=0
        for "no node" and ``ServersResponse`` enforces the
        ``(node_id, address)`` atomicity invariant — a 0-id entry
        with a non-empty address is rejected at the wire boundary."""
        from dqlitewire.messages.responses import ServersResponse
        from dqlitewire.types import encode_text, encode_uint64

        body = encode_uint64(3)
        for i in range(1, 4):
            body += encode_uint64(i)
            body += encode_text(f"node{i}.example:900{i}")
            body += encode_uint64(0)
        msg = ServersResponse.decode_body(body)
        assert [n.address for n in msg.nodes] == [
            "node1.example:9001",
            "node2.example:9002",
            "node3.example:9003",
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


class TestFailureResponseStrictLength:
    """Body: uint64 code + padded text message. Trailing bytes after
    the padded message had been silently accepted."""

    @staticmethod
    def _body(code: int = 5, message: str = "database is locked") -> bytes:
        from dqlitewire.types import encode_text, encode_uint64

        return encode_uint64(code) + encode_text(message)

    def test_exact_round_trip(self) -> None:
        from dqlitewire.messages.responses import FailureResponse

        msg = FailureResponse.decode_body(self._body())
        assert msg.code == 5
        assert msg.message == "database is locked"

    def test_trailing_byte_rejected(self) -> None:
        from dqlitewire.messages.responses import FailureResponse

        body = self._body() + b"\x01"
        with pytest.raises(DecodeError, match=r"FailureResponse has 1 trailing byte"):
            FailureResponse.decode_body(body)

    def test_trailing_word_rejected(self) -> None:
        from dqlitewire.messages.responses import FailureResponse

        body = self._body() + b"\x00" * 8
        with pytest.raises(DecodeError, match=r"FailureResponse has 8 trailing byte"):
            FailureResponse.decode_body(body)


class TestLeaderResponseStrictLength:
    """Modern body: uint64 node_id + padded text address. Legacy body:
    padded text address only. Trailing bytes after the address had
    been silently dropped."""

    @staticmethod
    def _modern_body(node_id: int = 7, addr: str = "1.2.3.4:9001") -> bytes:
        from dqlitewire.types import encode_text, encode_uint64

        return encode_uint64(node_id) + encode_text(addr)

    def test_modern_exact_round_trip(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse

        msg = LeaderResponse.decode_body(self._modern_body())
        assert msg.node_id == 7
        assert msg.address == "1.2.3.4:9001"

    def test_modern_trailing_byte_rejected(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse

        body = self._modern_body() + b"\x01"
        with pytest.raises(DecodeError, match=r"LeaderResponse has 1 trailing byte"):
            LeaderResponse.decode_body(body)

    def test_modern_trailing_word_rejected(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse

        body = self._modern_body() + b"\x00" * 8
        with pytest.raises(DecodeError, match=r"LeaderResponse has 8 trailing byte"):
            LeaderResponse.decode_body(body)

    def test_legacy_exact_round_trip(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse
        from dqlitewire.types import encode_text

        msg = LeaderResponse.decode_body_legacy(encode_text("10.0.0.1:9001"))
        assert msg.node_id == 0
        assert msg.address == "10.0.0.1:9001"

    def test_legacy_trailing_byte_rejected(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse
        from dqlitewire.types import encode_text

        body = encode_text("10.0.0.1:9001") + b"\x01"
        with pytest.raises(DecodeError, match=r"LeaderResponse \(legacy\) has 1 trailing byte"):
            LeaderResponse.decode_body_legacy(body)

    def test_legacy_trailing_word_rejected(self) -> None:
        from dqlitewire.messages.responses import LeaderResponse
        from dqlitewire.types import encode_text

        body = encode_text("10.0.0.1:9001") + b"\x00" * 8
        with pytest.raises(DecodeError, match=r"LeaderResponse \(legacy\) has 8 trailing byte"):
            LeaderResponse.decode_body_legacy(body)

    def test_modern_short_body_rejected_at_response_layer(self) -> None:
        # ``decode_uint64`` already rejects bodies shorter than 8 bytes,
        # but its diagnostic does not mention which response was being
        # decoded. Sibling response decoders (FailureResponse,
        # WelcomeResponse, MetadataResponse) all emit an explicit
        # response-level length guard so the diagnostic is self-
        # describing. Pin the wording so a future refactor of
        # ``decode_uint64`` cannot silently demote LeaderResponse to
        # the helper's wording.
        from dqlitewire.messages.responses import LeaderResponse

        with pytest.raises(DecodeError, match=r"LeaderResponse body too short"):
            LeaderResponse.decode_body(b"\x01" * 7)

    def test_legacy_short_body_rejected_at_response_layer(self) -> None:
        # Sibling pin for the legacy decoder: ``decode_text`` rejects
        # bodies shorter than the 8-byte length prefix, but its
        # diagnostic does not name the response. Mirror the modern
        # guard so pre-1.0 server log lines are self-describing.
        from dqlitewire.messages.responses import LeaderResponse

        with pytest.raises(DecodeError, match=r"LeaderResponse \(legacy\) body too short"):
            LeaderResponse.decode_body_legacy(b"\x01" * 7)


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
