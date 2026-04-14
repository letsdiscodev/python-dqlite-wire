"""Tests for response message encoding/decoding."""

import pytest

from dqlitewire.constants import HEADER_SIZE, ResponseType, ValueType
from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.base import Header
from dqlitewire.messages.responses import (
    DbResponse,
    EmptyResponse,
    FailureResponse,
    FilesResponse,
    LeaderResponse,
    MetadataResponse,
    NodeInfo,
    ResultResponse,
    RowsResponse,
    ServersResponse,
    StmtResponse,
    WelcomeResponse,
)


class TestFailureResponse:
    def test_encode(self) -> None:
        msg = FailureResponse(code=1, message="error")
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == ResponseType.FAILURE

    def test_roundtrip(self) -> None:
        msg = FailureResponse(code=42, message="Something went wrong")
        encoded = msg.encode()
        decoded = FailureResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.code == 42
        assert decoded.message == "Something went wrong"

    def test_empty_message(self) -> None:
        msg = FailureResponse(code=0, message="")
        encoded = msg.encode()
        decoded = FailureResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.code == 0
        assert decoded.message == ""


class TestLeaderResponse:
    def test_roundtrip(self) -> None:
        msg = LeaderResponse(node_id=1, address="192.168.1.1:9001")
        encoded = msg.encode()
        decoded = LeaderResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.node_id == 1
        assert decoded.address == "192.168.1.1:9001"

    def test_empty_address(self) -> None:
        msg = LeaderResponse(node_id=0, address="")
        encoded = msg.encode()
        decoded = LeaderResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.node_id == 0
        assert decoded.address == ""

    def test_decode_body_legacy(self) -> None:
        """Pre-1.0 servers send only a text address (no node_id) for type code 1.

        Go has DecodeNodeLegacy for this. decode_body_legacy() handles the
        legacy format, returning node_id=0.
        """
        from dqlitewire.types import encode_text

        legacy_body = encode_text("192.168.1.1:9001")
        decoded = LeaderResponse.decode_body_legacy(legacy_body)
        assert decoded.node_id == 0
        assert decoded.address == "192.168.1.1:9001"


class TestWelcomeResponse:
    def test_roundtrip(self) -> None:
        msg = WelcomeResponse(heartbeat_timeout=15000)
        encoded = msg.encode()
        decoded = WelcomeResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.heartbeat_timeout == 15000


class TestDbResponse:
    def test_roundtrip(self) -> None:
        msg = DbResponse(db_id=1)
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == ResponseType.DB
        decoded = DbResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 1


class TestStmtResponse:
    def test_roundtrip(self) -> None:
        msg = StmtResponse(db_id=1, stmt_id=5, num_params=3)
        encoded = msg.encode()
        decoded = StmtResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.db_id == 1
        assert decoded.stmt_id == 5
        assert decoded.num_params == 3

    def test_body_size(self) -> None:
        """StmtResponse body should be exactly 16 bytes per Go reference.

        Format: uint32 db_id + uint32 stmt_id + uint64 num_params = 16 bytes.
        """
        msg = StmtResponse(db_id=1, stmt_id=2, num_params=3)
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.body_size == 16

    def test_roundtrip_v1_with_tail_offset(self) -> None:
        """V1 STMT response includes tail_offset for multi-statement prepare.

        Note: tail_offset is not present in the canonical Go client
        (go-dqlite). It may be supported by the C server but the Go
        client's EncodePrepare always uses schema=0 and DecodeStmt
        never reads tail_offset.
        """
        msg = StmtResponse(db_id=1, stmt_id=5, num_params=3, tail_offset=42)
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.body_size == 24  # 16 + 8 for tail_offset
        decoded = StmtResponse.decode_body(encoded[HEADER_SIZE:], schema=header.schema)
        assert decoded.db_id == 1
        assert decoded.stmt_id == 5
        assert decoded.num_params == 3
        assert decoded.tail_offset == 42

    def test_v0_has_no_tail_offset(self) -> None:
        """V0 STMT response has no tail_offset."""
        msg = StmtResponse(db_id=1, stmt_id=5, num_params=3)
        encoded = msg.encode()
        decoded = StmtResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.tail_offset is None


class TestResultResponse:
    def test_roundtrip(self) -> None:
        msg = ResultResponse(last_insert_id=42, rows_affected=5)
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == ResponseType.RESULT
        decoded = ResultResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.last_insert_id == 42
        assert decoded.rows_affected == 5

    def test_zero_values(self) -> None:
        msg = ResultResponse(last_insert_id=0, rows_affected=0)
        encoded = msg.encode()
        decoded = ResultResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.last_insert_id == 0
        assert decoded.rows_affected == 0


class TestRowsResponseAliasing:
    """Regression tests for issue 042.

    ``RowsResponse`` used to store caller-supplied ``column_names`` and
    ``column_types`` lists by reference, creating silent aliasing:

    ``column_types`` was stored as ``all_row_types[0]`` inside
    ``decode_body`` via ``if not column_types: column_types = types``.
    The returned object then had ``r.column_types is r.row_types[0]``,
    so mutating one list silently rewrote the other.

    The fix copies both lists on construction via ``__post_init__``.
    """

    def test_column_types_is_not_aliased_to_row_types_first(self) -> None:
        """After decoding, ``column_types`` must be a distinct list
        object from ``row_types[0]`` so mutation of one does not
        propagate to the other.
        """
        msg = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[[1, "alice"], [2, "bob"]],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])

        assert decoded.row_types  # sanity: rows were decoded with types
        assert decoded.column_types is not decoded.row_types[0], (
            "column_types must not share identity with row_types[0]"
        )

        # Prove independence by mutation.
        decoded.column_types[0] = ValueType.NULL
        assert decoded.row_types[0][0] != ValueType.NULL

    def test_column_names_defensive_copy_on_construct(self) -> None:
        """A caller who supplies ``column_names`` to RowsResponse and
        later mutates the source list must not see the change
        reflected in the returned object.
        """
        names_in = ["a", "b", "c"]
        msg = RowsResponse(
            column_names=names_in,
            column_types=[ValueType.INTEGER, ValueType.TEXT, ValueType.BLOB],
            row_types=[],
            rows=[],
            has_more=False,
        )

        names_in.append("d")

        assert msg.column_names == ["a", "b", "c"], (
            "RowsResponse.column_names must not alias the caller's list"
        )

    def test_column_types_defensive_copy_on_construct(self) -> None:
        """Same for column_types passed into the constructor."""
        types_in = [ValueType.INTEGER, ValueType.TEXT]
        msg = RowsResponse(
            column_names=["a", "b"],
            column_types=types_in,
            row_types=[],
            rows=[],
            has_more=False,
        )

        types_in[0] = ValueType.BLOB

        assert msg.column_types[0] == ValueType.INTEGER, (
            "RowsResponse.column_types must not alias the caller's list"
        )

    def test_get_row_types_does_not_alias_column_types(self) -> None:
        """Regression for issue 052.

        ``__post_init__`` defensively copies ``column_types`` so that
        mutation of a caller-supplied or decoder-supplied list cannot
        rewrite the ``RowsResponse``'s private copy. ``_get_row_types``
        used to return ``self.column_types`` by reference on the
        "no per-row types" fallback path, which silently punched a
        hole in that invariant: a caller who captured the return
        value and mutated it would mutate the message's private
        list.

        This test asserts that the returned list is a distinct object
        from ``self.column_types`` and that mutating the return value
        does not affect the message.
        """
        msg = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            row_types=[],
            rows=[[1]],
            has_more=False,
        )

        row_types = msg._get_row_types(0, [1])

        assert row_types is not msg.column_types, (
            "_get_row_types must return a fresh list, not alias column_types"
        )

        row_types.append(ValueType.BLOB)
        assert msg.column_types == [ValueType.INTEGER], (
            "mutating the _get_row_types return value must not affect "
            "msg.column_types (issue 042 invariant)"
        )

    def test_get_row_types_does_not_alias_row_types(self) -> None:
        """Regression for issue 059.

        The ``row_types`` branch of ``_get_row_types`` used to return
        ``self.row_types[row_idx]`` by reference — the same aliasing
        pattern that issue 052 fixed for the ``column_types`` fallback.
        A caller who captured the return value and mutated it would
        silently rewrite the message's internal ``row_types[i]`` list.
        """
        msg = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[1]],
            has_more=False,
        )

        row_types = msg._get_row_types(0, [1])

        assert row_types is not msg.row_types[0], (
            "_get_row_types must return a fresh list, not alias row_types[i]"
        )

        row_types.append(ValueType.BLOB)
        assert msg.row_types[0] == [ValueType.INTEGER], (
            "mutating the _get_row_types return value must not affect "
            "msg.row_types[i] (issue 042 invariant)"
        )


class TestRowsResponse:
    def test_empty_result(self) -> None:
        msg = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.column_names == ["id", "name"]
        assert decoded.rows == []
        assert decoded.has_more is False

    def test_single_row(self) -> None:
        msg = RowsResponse(
            column_names=["id"],
            column_types=[ValueType.INTEGER],
            rows=[[42]],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.column_names == ["id"]
        assert len(decoded.rows) == 1
        assert decoded.rows[0][0] == 42

    def test_multiple_rows(self) -> None:
        msg = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[[1, "Alice"], [2, "Bob"], [3, "Charlie"]],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert len(decoded.rows) == 3
        assert decoded.rows[0] == [1, "Alice"]
        assert decoded.rows[1] == [2, "Bob"]
        assert decoded.rows[2] == [3, "Charlie"]

    def test_has_more(self) -> None:
        msg = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            has_more=True,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.has_more is True

    def test_heterogeneous_types_roundtrip(self) -> None:
        """Rows with different types per column must encode/decode correctly.

        SQLite allows different storage types for the same column across rows
        (type affinity). Each row has its own type header in the wire format.
        """
        msg = RowsResponse(
            column_names=["x"],
            row_types=[
                [ValueType.INTEGER],
                [ValueType.TEXT],
                [ValueType.NULL],
            ],
            rows=[[42], ["hello"], [None]],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.rows[0] == [42]
        assert decoded.rows[1] == ["hello"]
        assert decoded.rows[2] == [None]
        assert decoded.row_types[0] == [ValueType.INTEGER]
        assert decoded.row_types[1] == [ValueType.TEXT]
        assert decoded.row_types[2] == [ValueType.NULL]

    def test_column_types_populated_on_decode(self) -> None:
        """column_types should be populated from the first row's types after decode."""
        msg = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[[1, "Alice"], [2, "Bob"]],
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.column_types == [ValueType.INTEGER, ValueType.TEXT]

    def test_column_types_reflects_first_row_only(self) -> None:
        """column_types reflects first row's types; use row_types for per-row types."""
        msg = RowsResponse(
            column_names=["x"],
            row_types=[
                [ValueType.INTEGER],
                [ValueType.TEXT],
                [ValueType.NULL],
            ],
            rows=[[42], ["hello"], [None]],
            has_more=False,
        )
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        # column_types reflects first row only
        assert decoded.column_types == [ValueType.INTEGER]
        # row_types has accurate per-row types
        assert decoded.row_types[0] == [ValueType.INTEGER]
        assert decoded.row_types[1] == [ValueType.TEXT]
        assert decoded.row_types[2] == [ValueType.NULL]

    def test_zero_columns_with_done_marker(self) -> None:
        """Zero-column result with DONE marker should decode to empty rows."""
        from dqlitewire.types import encode_uint64

        # column_count=0, followed by DONE marker
        data = encode_uint64(0) + encode_uint64(0xFFFFFFFFFFFFFFFF)
        decoded = RowsResponse.decode_body(data)
        assert decoded.column_names == []
        assert decoded.rows == []
        assert decoded.has_more is False

    def test_zero_columns_with_part_marker(self) -> None:
        """Zero-column result with PART marker should decode with has_more=True."""
        from dqlitewire.types import encode_uint64

        # column_count=0, followed by PART marker
        data = encode_uint64(0) + encode_uint64(0xEEEEEEEEEEEEEEEE)
        decoded = RowsResponse.decode_body(data)
        assert decoded.column_names == []
        assert decoded.rows == []
        assert decoded.has_more is True

    def test_zero_columns_roundtrip(self) -> None:
        """Zero-column RowsResponse should roundtrip correctly."""
        msg = RowsResponse(column_names=[], rows=[], has_more=False)
        encoded = msg.encode()
        decoded = RowsResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.column_names == []
        assert decoded.rows == []
        assert decoded.has_more is False

    def test_zero_columns_malformed_no_infinite_loop(self) -> None:
        """Zero-column result with non-marker data must raise a clear error."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # column_count=0, followed by non-marker bytes
        data = encode_uint64(0) + b"\x01\x02\x03\x04\x05\x06\x07\x08"
        with pytest.raises(DecodeError, match="Expected DONE or PART marker"):
            RowsResponse.decode_body(data)

    def test_zero_columns_missing_marker_raises(self) -> None:
        """Zero-column result with no end marker should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # column_count=0, but no marker follows
        data = encode_uint64(0)
        with pytest.raises(DecodeError, match="end marker"):
            RowsResponse.decode_body(data)

    def test_decode_body_rejects_non_list_row_header(self) -> None:
        """decode_body must raise DecodeError if decode_row_header returns unexpected type."""
        from unittest.mock import patch

        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        # Build a valid body
        body = encode_uint64(1)  # column_count=1
        body += encode_text("id")
        body += encode_row_header([ValueType.INTEGER])
        body += encode_row_values([42], [ValueType.INTEGER])
        body += encode_uint64(0xFFFFFFFFFFFFFFFF)  # DONE

        # Patch decode_row_header to return a string (simulating unexpected type)
        with (
            patch("dqlitewire.messages.responses.decode_row_header", return_value=("bad", 8)),
            pytest.raises(DecodeError, match="Expected column types list"),
        ):
            RowsResponse.decode_body(body)

    def test_truncated_body_without_marker_raises(self) -> None:
        """Body exhausted without DONE/PART marker must raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        # Build a body with rows but NO end marker
        body = encode_uint64(1)  # column_count=1
        body += encode_text("id")
        body += encode_row_header([ValueType.INTEGER])
        body += encode_row_values([42], [ValueType.INTEGER])
        # No marker at the end!

        with pytest.raises(DecodeError, match="end marker"):
            RowsResponse.decode_body(body)

    def test_bogus_column_count_raises(self) -> None:
        """A column_count larger than remaining data should raise DecodeError early."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # column_count = 1 billion, but only 8 bytes of data after count
        body = encode_uint64(1_000_000_000) + b"\x00" * 8
        with pytest.raises(DecodeError, match="exceeds maximum"):
            RowsResponse.decode_body(body)

    def test_column_count_exceeds_hard_limit(self) -> None:
        """Column count exceeding the hard limit should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # 20_000 columns, with enough data to pass the data-size check
        body = encode_uint64(20_000) + b"\x00" * (20_000 * 8 + 8)
        with pytest.raises(DecodeError, match="Column count.*exceeds maximum"):
            RowsResponse.decode_body(body)

    def test_max_rows_limit_decode_body(self) -> None:
        """decode_body should reject messages exceeding the max_rows limit."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        # Build a body with 5 rows
        body = encode_uint64(1)  # column_count=1
        body += encode_text("x")
        for i in range(5):
            body += encode_row_header([ValueType.INTEGER])
            body += encode_row_values([i], [ValueType.INTEGER])
        body += encode_uint64(0xFFFFFFFFFFFFFFFF)  # DONE

        # Should succeed with default limit
        decoded = RowsResponse.decode_body(body)
        assert len(decoded.rows) == 5

        # Should fail with max_rows=3
        with pytest.raises(DecodeError, match="Row count.*exceeds maximum"):
            RowsResponse.decode_body(body, max_rows=3)

    def test_max_rows_exact_boundary_rejects_at_limit(self) -> None:
        """max_rows=3 with exactly 3 rows should raise DecodeError.

        The max_rows parameter is a strict upper bound: at most max_rows - 1
        rows should be decoded without error. When the number of rows
        reaches max_rows, the limit has been exceeded and DecodeError
        should fire immediately — without decoding another row first.
        """
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        def build_body(n_rows: int) -> bytes:
            body = encode_uint64(1)  # column_count=1
            body += encode_text("x")
            for i in range(n_rows):
                body += encode_row_header([ValueType.INTEGER])
                body += encode_row_values([i], [ValueType.INTEGER])
            body += encode_uint64(0xFFFFFFFFFFFFFFFF)  # DONE
            return body

        # Exactly max_rows rows should raise
        with pytest.raises(DecodeError, match="exceeds maximum"):
            RowsResponse.decode_body(build_body(3), max_rows=3)

        # One fewer than max_rows should succeed
        decoded = RowsResponse.decode_body(build_body(2), max_rows=3)
        assert len(decoded.rows) == 2


class TestRowsResponseEncodeValidation:
    """Encode-time validation of column count consistency."""

    def test_mismatched_column_types_length(self) -> None:
        resp = RowsResponse(
            column_names=["a", "b", "c"],
            column_types=[ValueType.INTEGER],
            rows=[],
        )
        with pytest.raises(EncodeError, match="column_types length"):
            resp.encode_body()

    def test_mismatched_row_values_length(self) -> None:
        resp = RowsResponse(
            column_names=["a", "b"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            row_types=[[ValueType.INTEGER, ValueType.TEXT]],
            rows=[[1]],  # only 1 value, expected 2
        )
        with pytest.raises(EncodeError, match="Row 0 has 1 values, expected 2"):
            resp.encode_body()

    def test_mismatched_row_types_length(self) -> None:
        resp = RowsResponse(
            column_names=["a", "b"],
            row_types=[[ValueType.INTEGER]],  # only 1 type, expected 2
            rows=[[1, 2]],
        )
        with pytest.raises(EncodeError, match="row_types\\[0\\] has 1 types, expected 2"):
            resp.encode_body()

    def test_empty_column_types_with_rows_ok(self) -> None:
        """Empty column_types is valid — types are inferred from values."""
        resp = RowsResponse(
            column_names=["a"],
            rows=[[42]],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.rows == [[42]]


class TestRowsResponseNullInTypedColumn:
    """137: None values in rows with explicit column types must encode correctly."""

    def test_none_with_explicit_column_types(self) -> None:
        """None in a TEXT column must encode as NULL, not crash."""
        resp = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            row_types=[[ValueType.INTEGER, ValueType.TEXT]],
            rows=[[1, None]],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.rows == [[1, None]]
        # The type nibble for the None column should be NULL (5), not TEXT (3)
        assert decoded.row_types[0][1] == ValueType.NULL

    def test_none_with_explicit_row_types_integer(self) -> None:
        """None in an INTEGER column with explicit row_types."""
        resp = RowsResponse(
            column_names=["val"],
            column_types=[ValueType.INTEGER],
            row_types=[[ValueType.INTEGER]],
            rows=[[None]],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.rows == [[None]]
        assert decoded.row_types[0][0] == ValueType.NULL

    def test_mixed_none_and_values(self) -> None:
        """Row with mix of None and real values."""
        resp = RowsResponse(
            column_names=["a", "b", "c"],
            column_types=[ValueType.INTEGER, ValueType.TEXT, ValueType.FLOAT],
            row_types=[[ValueType.INTEGER, ValueType.TEXT, ValueType.FLOAT]],
            rows=[[None, "hello", None]],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.rows == [[None, "hello", None]]
        assert decoded.row_types[0][0] == ValueType.NULL
        assert decoded.row_types[0][1] == ValueType.TEXT
        assert decoded.row_types[0][2] == ValueType.NULL


class TestRowsResponseWordBoundary:
    """115: row header padding crosses word boundary at 17+ columns."""

    def test_16_columns_exact_word(self) -> None:
        """16 columns = 8 nibble bytes = exactly 1 word, no padding."""
        n = 16
        types = [ValueType.INTEGER] * n
        names = [f"c{i}" for i in range(n)]
        values = list(range(n))
        resp = RowsResponse(
            column_names=names,
            column_types=types,
            row_types=[types],
            rows=[values],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.column_names == names
        assert decoded.rows == [values]
        assert decoded.row_types[0] == types

    def test_17_columns_crosses_word_boundary(self) -> None:
        """17 columns = 9 nibble bytes, padded to 16 = 2 words."""
        n = 17
        types = [ValueType.INTEGER] * n
        names = [f"c{i}" for i in range(n)]
        values = list(range(n))
        resp = RowsResponse(
            column_names=names,
            column_types=types,
            row_types=[types],
            rows=[values],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.column_names == names
        assert decoded.rows == [values]
        assert decoded.row_types[0] == types

    def test_33_columns_third_word_boundary(self) -> None:
        """33 columns = 17 nibble bytes, padded to 24 = 3 words."""
        n = 33
        types = [ValueType.INTEGER] * n
        names = [f"c{i}" for i in range(n)]
        values = list(range(n))
        resp = RowsResponse(
            column_names=names,
            column_types=types,
            row_types=[types],
            rows=[values],
        )
        encoded = resp.encode_body()
        decoded = RowsResponse.decode_body(encoded)
        assert decoded.column_names == names
        assert decoded.rows == [values]
        assert decoded.row_types[0] == types


class TestRowsResponseValueTypes:
    """Full RowsResponse round-trips with BOOLEAN, UNIXTIME, ISO8601, and BLOB."""

    def test_boolean_column(self) -> None:
        """BOOLEAN (code 11) uses all 4 nibble bits — catches truncation bugs."""
        resp = RowsResponse(
            column_names=["flag"],
            column_types=[ValueType.BOOLEAN],
            rows=[[True], [False]],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        assert decoded.rows == [[True], [False]]
        assert decoded.column_types == [ValueType.BOOLEAN]

    def test_unixtime_column(self) -> None:
        """UNIXTIME (code 9) round-trips as raw int, not datetime."""
        resp = RowsResponse(
            column_names=["created_at"],
            column_types=[ValueType.UNIXTIME],
            row_types=[[ValueType.UNIXTIME], [ValueType.UNIXTIME]],
            rows=[[1700000000], [0]],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        assert decoded.rows == [[1700000000], [0]]
        assert decoded.column_types == [ValueType.UNIXTIME]

    def test_iso8601_column(self) -> None:
        """ISO8601 values go through _parse_iso8601 on decode."""
        from datetime import UTC, datetime

        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        resp = RowsResponse(
            column_names=["ts"],
            column_types=[ValueType.ISO8601],
            rows=[[dt]],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        assert decoded.rows[0][0] == dt

    def test_blob_column(self) -> None:
        """BLOB has variable-length encoding with padding in row context."""
        resp = RowsResponse(
            column_names=["data"],
            column_types=[ValueType.BLOB],
            rows=[[b"\xde\xad\xbe\xef"], [b""]],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        assert decoded.rows == [[b"\xde\xad\xbe\xef"], [b""]]

    def test_mixed_all_types(self) -> None:
        """Every value type in a single row exercises full encoding path."""
        from datetime import UTC, datetime

        dt = datetime(2024, 1, 1, tzinfo=UTC)
        resp = RowsResponse(
            column_names=["i", "f", "t", "b", "n", "ut", "iso", "bo"],
            column_types=[
                ValueType.INTEGER,
                ValueType.FLOAT,
                ValueType.TEXT,
                ValueType.BLOB,
                ValueType.NULL,
                ValueType.UNIXTIME,
                ValueType.ISO8601,
                ValueType.BOOLEAN,
            ],
            row_types=[
                [
                    ValueType.INTEGER,
                    ValueType.FLOAT,
                    ValueType.TEXT,
                    ValueType.BLOB,
                    ValueType.NULL,
                    ValueType.UNIXTIME,
                    ValueType.ISO8601,
                    ValueType.BOOLEAN,
                ]
            ],
            rows=[
                [42, 3.14, "hello", b"\x00\x01", None, 1700000000, dt, True],
            ],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        row = decoded.rows[0]
        assert row[0] == 42
        assert row[1] == 3.14
        assert row[2] == "hello"
        assert row[3] == b"\x00\x01"
        assert row[4] is None
        assert row[5] == 1700000000
        assert row[6] == dt
        assert row[7] is True

    def test_blob_multiple_sizes(self) -> None:
        """Multiple BLOB rows with different sizes verify padding/offset interplay."""
        resp = RowsResponse(
            column_names=["data"],
            column_types=[ValueType.BLOB],
            rows=[
                [b""],  # empty
                [b"x"],  # 1 byte, needs 7 pad
                [b"12345678"],  # exactly 8 bytes, no pad
                [b"123456789"],  # 9 bytes, needs 7 pad
            ],
        )
        data = resp.encode()
        decoded = RowsResponse.decode_body(data[HEADER_SIZE:])
        assert decoded.rows[0] == [b""]
        assert decoded.rows[1] == [b"x"]
        assert decoded.rows[2] == [b"12345678"]
        assert decoded.rows[3] == [b"123456789"]


class TestEmptyResponse:
    def test_encode(self) -> None:
        msg = EmptyResponse()
        encoded = msg.encode()
        header = Header.decode(encoded[:HEADER_SIZE])
        assert header.msg_type == ResponseType.EMPTY
        assert header.size_words == 1  # Reserved uint64 per Go spec

    def test_body_has_reserved_field(self) -> None:
        """EmptyResponse body must contain a reserved uint64 per Go spec."""
        msg = EmptyResponse()
        body = msg.encode_body()
        assert len(body) == 8

    def test_roundtrip(self) -> None:
        msg = EmptyResponse()
        encoded = msg.encode()
        decoded = EmptyResponse.decode_body(encoded[HEADER_SIZE:])
        assert isinstance(decoded, EmptyResponse)


class TestFilesResponse:
    def test_empty(self) -> None:
        msg = FilesResponse(files={})
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files == {}

    def test_empty_wire_format(self) -> None:
        """Empty files should encode as just uint64 count=0."""
        msg = FilesResponse(files={})
        body = msg.encode_body()
        # Should be exactly 8 bytes: uint64 count = 0
        assert len(body) == 8
        assert body == b"\x00" * 8

    def test_wire_format_starts_with_count(self) -> None:
        """Body must start with uint64 file count per Go wire protocol."""
        from dqlitewire.types import decode_uint64

        msg = FilesResponse(files={"test.db": b"data"})
        body = msg.encode_body()
        count = decode_uint64(body[:8])
        assert count == 1

    def test_wire_format_has_size_field(self) -> None:
        """Each file entry has text name, uint64 size, then raw content bytes."""
        from dqlitewire.types import decode_text, decode_uint64

        content = b"abcdefgh"
        msg = FilesResponse(files={"test.db": content})
        body = msg.encode_body()
        offset = 8  # skip count
        _name, consumed = decode_text(body[offset:])
        offset += consumed
        # Next should be uint64 size of content
        size = decode_uint64(body[offset:])
        assert size == len(content)

    def test_roundtrip(self) -> None:
        msg = FilesResponse(files={"db.sqlite": b"database content", "wal": b"wal data"})
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files["db.sqlite"] == b"database content"
        assert decoded.files["wal"] == b"wal data"

    def test_roundtrip_single_file(self) -> None:
        msg = FilesResponse(files={"main.db": b"\x00\x01\x02\x03"})
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files["main.db"] == b"\x00\x01\x02\x03"

    def test_roundtrip_aligned_content(self) -> None:
        """Real dqlite content is always word-aligned (SQLite pages are multiples of 512)."""
        page = b"\x00" * 512  # Realistic SQLite page size
        msg = FilesResponse(
            files={
                "main.db": page,
                "wal.db": page + page,
            }
        )
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files["main.db"] == page
        assert decoded.files["wal.db"] == page + page

    def test_roundtrip_non_aligned_content(self) -> None:
        """Non-aligned content roundtrips correctly without padding.

        The C server asserts content is always word-aligned. Neither Go nor
        Python adds padding after file content. This test verifies that
        non-aligned content still works for encode/decode symmetry.
        """
        msg = FilesResponse(
            files={
                "file1.db": b"\x01\x02\x03",  # 3 bytes, needs 5 padding
                "file2.db": b"\x04\x05\x06\x07\x08\x09\x0a",  # 7 bytes, needs 1 padding
            }
        )
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files["file1.db"] == b"\x01\x02\x03"
        assert decoded.files["file2.db"] == b"\x04\x05\x06\x07\x08\x09\x0a"

    def test_roundtrip_empty_content(self) -> None:
        """116: zero-length file content must round-trip correctly."""
        msg = FilesResponse(files={"empty.db": b""})
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files == {"empty.db": b""}

    def test_roundtrip_mixed_empty_and_nonempty(self) -> None:
        """116: empty and non-empty files in the same response."""
        msg = FilesResponse(
            files={
                "main.db": b"data",
                "empty.db": b"",
                "wal.db": b"more data",
            }
        )
        encoded = msg.encode()
        decoded = FilesResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.files == msg.files

    def test_aligned_content_has_no_padding(self) -> None:
        """Word-aligned content must not produce any extra padding bytes."""
        content = b"\x00" * 16  # exactly 2 words
        msg = FilesResponse(files={"a.db": content})
        body = msg.encode_body()
        # count(8) + name "a.db\0"(8) + size(8) + content(16) = 40
        assert len(body) == 40

    def test_no_padding_after_content_matches_go(self) -> None:
        """Go reads file content without padding; encoder must not add padding."""
        from dqlitewire.types import encode_text, encode_uint64

        # Manually build Go-format body: no padding after non-aligned content
        body = encode_uint64(2)  # count=2
        body += encode_text("f1")  # filename
        body += encode_uint64(3)  # size=3
        body += b"\x01\x02\x03"  # content (not word-aligned, no padding!)
        body += encode_text("f2")  # next filename immediately after
        body += encode_uint64(1)  # size=1
        body += b"\xff"  # content
        decoded = FilesResponse.decode_body(body)
        assert decoded.files["f1"] == b"\x01\x02\x03"
        assert decoded.files["f2"] == b"\xff"

    def test_bogus_file_count_raises(self) -> None:
        """A file count larger than remaining data should raise DecodeError early."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        body = encode_uint64(1_000_000_000) + b"\x00" * 8
        with pytest.raises(DecodeError, match="exceeds maximum"):
            FilesResponse.decode_body(body)

    def test_file_count_exceeds_hard_limit(self) -> None:
        """File count exceeding the hard limit should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # 200 files, enough data to pass data-size check
        body = encode_uint64(200) + b"\x00" * (200 * 16 + 8)
        with pytest.raises(DecodeError, match="File count.*exceeds maximum"):
            FilesResponse.decode_body(body)

    def test_truncated_file_content_raises(self) -> None:
        """Declared file size larger than available data should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_text, encode_uint64

        body = encode_uint64(1)  # count=1
        body += encode_text("test.db")  # filename
        body += encode_uint64(4096)  # claims 4096 bytes
        body += b"\x00" * 100  # but only 100 bytes available
        with pytest.raises(DecodeError, match="truncated"):
            FilesResponse.decode_body(body)


class TestServersResponse:
    def test_empty(self) -> None:
        msg = ServersResponse(nodes=[])
        encoded = msg.encode()
        decoded = ServersResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.nodes == []

    def test_roundtrip(self) -> None:
        nodes = [
            NodeInfo(node_id=1, address="node1:9001", role=1),
            NodeInfo(node_id=2, address="node2:9002", role=2),
            NodeInfo(node_id=3, address="node3:9003", role=2),
        ]
        msg = ServersResponse(nodes=nodes)
        encoded = msg.encode()
        decoded = ServersResponse.decode_body(encoded[HEADER_SIZE:])
        assert len(decoded.nodes) == 3
        for i, expected in enumerate(nodes):
            assert decoded.nodes[i].node_id == expected.node_id
            assert decoded.nodes[i].address == expected.address
            assert decoded.nodes[i].role == expected.role

    def test_wire_format_starts_with_count(self) -> None:
        """Body must start with uint64 node count per Go reference."""
        from dqlitewire.types import decode_uint64

        nodes = [
            NodeInfo(node_id=1, address="node1:9001", role=1),
            NodeInfo(node_id=2, address="node2:9002", role=2),
        ]
        msg = ServersResponse(nodes=nodes)
        body = msg.encode_body()
        count = decode_uint64(body[:8])
        assert count == 2

    def test_empty_wire_format(self) -> None:
        """Empty nodes should encode as just uint64 count=0."""
        msg = ServersResponse(nodes=[])
        body = msg.encode_body()
        assert len(body) == 8
        assert body == b"\x00" * 8

    def test_bogus_node_count_raises(self) -> None:
        """A node count larger than remaining data should raise DecodeError early."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        body = encode_uint64(1_000_000_000) + b"\x00" * 8
        with pytest.raises(DecodeError, match="exceeds maximum"):
            ServersResponse.decode_body(body)

    def test_node_count_exceeds_hard_limit(self) -> None:
        """Node count exceeding the hard limit should raise DecodeError."""
        import pytest

        from dqlitewire.exceptions import DecodeError
        from dqlitewire.types import encode_uint64

        # 20_000 nodes, enough data to pass data-size check
        body = encode_uint64(20_000) + b"\x00" * (20_000 * 24 + 8)
        with pytest.raises(DecodeError, match="Node count.*exceeds maximum"):
            ServersResponse.decode_body(body)


class TestMetadataResponse:
    def test_roundtrip(self) -> None:
        msg = MetadataResponse(failure_domain=1, weight=50)
        encoded = msg.encode()
        decoded = MetadataResponse.decode_body(encoded[HEADER_SIZE:])
        assert decoded.failure_domain == 1
        assert decoded.weight == 50
