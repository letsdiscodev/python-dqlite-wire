"""Frame-size caps: reject a count under the cap that needs more bytes than the body holds."""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.messages import FilesResponse, RowsResponse, ServersResponse
from dqlitewire.tuples import encode_row_values
from dqlitewire.types import encode_uint64


class TestFrameSizeCaps:
    def test_servers_response_count_exceeds_remaining_bytes(self) -> None:
        """count=1000 passes the absolute cap (max 10_000) but can't fit in an 8-byte body."""
        body = encode_uint64(1000)
        with pytest.raises(DecodeError, match="exceeds maximum possible"):
            ServersResponse.decode_body(body)

    def test_files_response_count_exceeds_remaining_bytes(self) -> None:
        """count=50 passes the cap (100) but can't fit an 8-byte body (each file >=16 bytes)."""
        body = encode_uint64(50)
        with pytest.raises(DecodeError, match="exceeds maximum possible"):
            FilesResponse.decode_body(body)

    def test_rows_response_count_exceeds_remaining_bytes(self) -> None:
        """column_count=200 passes the cap (255) but can't fit a body under count * 8 bytes."""
        body = encode_uint64(200)
        with pytest.raises(DecodeError, match="exceeds maximum possible"):
            RowsResponse.decode_body(body)


class TestEncodeRowValuesLengthMismatch:
    def test_more_values_than_types_rejected(self) -> None:
        with pytest.raises(EncodeError, match="Row values count"):
            encode_row_values([1, 2], [ValueType.INTEGER])

    def test_fewer_values_than_types_rejected(self) -> None:
        with pytest.raises(EncodeError, match="Row values count"):
            encode_row_values([1], [ValueType.INTEGER, ValueType.INTEGER])
