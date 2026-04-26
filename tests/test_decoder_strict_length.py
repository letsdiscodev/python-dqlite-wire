"""Request / response decoders must reject trailing bytes.

Upstream C drives every decode through a ``struct cursor`` with an
explicit ``cap``, so extra bytes past the declared fields are
detected (``DQLITE_PARSE``). Prior cycles applied the strict-length
pattern to ``EmptyResponse`` / ``DbResponse`` / ``ResultResponse``
and ``LeaderRequest``. This module fences the same pattern across
the rest of the request decoders and the two response decoders that
still accepted trailing bytes (``FilesResponse`` last-file tail;
``RowsResponse`` zero-column fast path).

Covers:
- ISSUE-319 (DescribeRequest format membership)
- ISSUE-320 (fixed-size + variable-size request decoders strict)
- ISSUE-321 (AssignRequest equality, not threshold)
- ISSUE-322 (FilesResponse trailing bytes past last file)
- ISSUE-323 (RowsResponse zero-column fast path trailing bytes)
"""

from __future__ import annotations

from typing import Any

import pytest

from dqlitewire.constants import ROW_DONE_MARKER, WORD_SIZE
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages import FilesResponse, RowsResponse
from dqlitewire.messages.requests import (
    AddRequest,
    AssignRequest,
    ClientRequest,
    ClusterRequest,
    DescribeRequest,
    DumpRequest,
    ExecRequest,
    ExecSqlRequest,
    FinalizeRequest,
    InterruptRequest,
    OpenRequest,
    PrepareRequest,
    QueryRequest,
    QuerySqlRequest,
    RemoveRequest,
    TransferRequest,
    WeightRequest,
    _ConnectRequest,
    _HeartbeatRequest,
)
from dqlitewire.types import encode_uint64


class TestFixedSizeRequestStrictLength:
    """Each fixed-size request decoder must reject body sizes other than 8."""

    @pytest.mark.parametrize(
        "cls",
        [
            ClientRequest,
            _HeartbeatRequest,
            InterruptRequest,
            RemoveRequest,
            TransferRequest,
            WeightRequest,
            FinalizeRequest,
        ],
    )
    def test_too_long_body_rejected(self, cls: Any) -> None:
        # Exactly 8 bytes should succeed; extra 8 should fail.
        body = encode_uint64(1) + encode_uint64(0)
        with pytest.raises(DecodeError, match="bytes"):
            cls.decode_body(body)

    @pytest.mark.parametrize(
        "cls",
        [
            ClientRequest,
            _HeartbeatRequest,
            InterruptRequest,
            RemoveRequest,
            TransferRequest,
            WeightRequest,
        ],
    )
    def test_too_short_body_rejected(self, cls: Any) -> None:
        with pytest.raises(DecodeError, match="bytes"):
            cls.decode_body(b"\x00" * 4)


class TestAssignRequestStrictLength:
    """ISSUE-321: `len(data) >= 16` was a threshold; must be equality."""

    def test_eight_byte_body_is_promote(self) -> None:
        body = encode_uint64(1)
        assert AssignRequest.decode_body(body) == AssignRequest(1, None)

    def test_sixteen_byte_body_is_assign(self) -> None:
        body = encode_uint64(1) + encode_uint64(0)
        assert AssignRequest.decode_body(body) == AssignRequest(1, 0)

    def test_twenty_four_byte_body_rejected(self) -> None:
        # Previously silently dropped the trailing 8 bytes.
        body = encode_uint64(1) + encode_uint64(0) + encode_uint64(42)
        with pytest.raises(DecodeError, match="8 .* or 16 .* bytes"):
            AssignRequest.decode_body(body)


class TestDescribeRequestFormatMembership:
    """ISSUE-319: upstream rejects DescribeRequest.format != 0 with
    SQLITE_PROTOCOL. Reject client-side so callers get a local ValueError.
    """

    def test_construct_with_nonzero_format_rejected(self) -> None:
        with pytest.raises(ValueError, match="format must be 0"):
            DescribeRequest(format=1)

    def test_decode_nonzero_format_rejected(self) -> None:
        with pytest.raises(DecodeError, match="format must be 0"):
            DescribeRequest.decode_body(encode_uint64(1))

    def test_construct_with_zero_format_allowed(self) -> None:
        req = DescribeRequest(format=0)
        assert req.format == 0


class TestVariableSizeRequestStrictLength:
    """Variable-size request decoders must reject trailing bytes."""

    def test_open_request_trailing_bytes_rejected(self) -> None:
        valid = OpenRequest("db").encode_body()
        with pytest.raises(DecodeError, match="trailing bytes"):
            OpenRequest.decode_body(valid + b"\x00" * 8)

    def test_prepare_request_trailing_bytes_rejected(self) -> None:
        valid = PrepareRequest(1, "SELECT 1").encode_body()
        with pytest.raises(DecodeError, match="trailing bytes"):
            PrepareRequest.decode_body(valid + b"\x00" * 8)

    def test_exec_request_trailing_bytes_rejected(self) -> None:
        # Use params so decode_params_tuple consumes a non-trivial window;
        # trailing bytes must then be caught by the offset-exhaustion check.
        valid = ExecRequest(1, 2, [42]).encode_body()
        with pytest.raises(DecodeError, match="trailing bytes"):
            ExecRequest.decode_body(valid + b"\x00" * 8)

    def test_query_request_trailing_bytes_rejected(self) -> None:
        valid = QueryRequest(1, 2, [42]).encode_body()
        with pytest.raises(DecodeError, match="trailing bytes"):
            QueryRequest.decode_body(valid + b"\x00" * 8)

    def test_exec_sql_request_trailing_bytes_rejected(self) -> None:
        valid = ExecSqlRequest(1, "SELECT 1", [42]).encode_body()
        with pytest.raises(DecodeError, match="trailing bytes"):
            ExecSqlRequest.decode_body(valid + b"\x00" * 8)

    def test_query_sql_request_trailing_bytes_rejected(self) -> None:
        valid = QuerySqlRequest(1, "SELECT 1", [42]).encode_body()
        with pytest.raises(DecodeError, match="trailing bytes"):
            QuerySqlRequest.decode_body(valid + b"\x00" * 8)

    def test_connect_request_trailing_bytes_rejected(self) -> None:
        valid = _ConnectRequest(1, "127.0.0.1:9001").encode_body()
        with pytest.raises(DecodeError, match="trailing bytes"):
            _ConnectRequest.decode_body(valid + b"\x00" * 8)

    def test_add_request_trailing_bytes_rejected(self) -> None:
        valid = AddRequest(1, "127.0.0.1:9001").encode_body()
        with pytest.raises(DecodeError, match="trailing bytes"):
            AddRequest.decode_body(valid + b"\x00" * 8)

    def test_dump_request_trailing_bytes_rejected(self) -> None:
        valid = DumpRequest("db").encode_body()
        with pytest.raises(DecodeError, match="trailing bytes"):
            DumpRequest.decode_body(valid + b"\x00" * 8)

    def test_cluster_request_trailing_bytes_rejected(self) -> None:
        valid = ClusterRequest(format=1).encode_body()
        with pytest.raises(DecodeError, match="bytes"):
            ClusterRequest.decode_body(valid + b"\x00" * 8)


class TestFilesResponseTrailingBytesRejected:
    """ISSUE-322: FilesResponse must reject bytes past the last file."""

    def test_trailing_bytes_after_last_file_rejected(self) -> None:
        msg = FilesResponse(files={"a.db": b"\x00" * 8})
        body = msg.encode_body()
        with pytest.raises(DecodeError, match="trailing bytes"):
            FilesResponse.decode_body(body + b"\x00" * 8)


class TestRowsResponseZeroColumnTrailingBytesRejected:
    """ISSUE-323: zero-column fast path must enforce buffer exhaustion."""

    def test_trailing_bytes_after_zero_column_marker_rejected(self) -> None:
        # Body: column_count=0 (uint64), DONE marker (uint64) — all 16 bytes.
        body = encode_uint64(0) + ROW_DONE_MARKER.to_bytes(WORD_SIZE, "little")
        # Valid case should decode cleanly.
        RowsResponse.decode_body(body)
        # Appending a trailing aligned word must now fail.
        with pytest.raises(DecodeError, match="trailing bytes"):
            RowsResponse.decode_body(body + b"\x00" * 8)
