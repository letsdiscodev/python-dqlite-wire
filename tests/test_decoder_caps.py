"""Fence the secondary "exceeds maximum possible in N bytes" caps.

Each response decoder with a count field has a two-stage check:
1. An absolute cap (``count > _MAX_FOO_COUNT``) — tested elsewhere.
2. A frame-size cap (``count * SLOT_SIZE > remaining``) — rejects
   a count that is small enough to pass stage 1 but would require
   more bytes than the body holds (allocation-amplification attack).

Stage 2's raises are tested here. A regression that removes the
frame-size check or reorders it with the absolute cap would ship
silently otherwise.

Also fences ``encode_row_values``'s length-mismatch EncodeError.
"""

from __future__ import annotations

import pytest

from dqlitewire.constants import ValueType
from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.messages import FilesResponse, RowsResponse, ServersResponse
from dqlitewire.tuples import encode_row_values
from dqlitewire.types import encode_uint64


class TestFrameSizeCaps:
    def test_servers_response_count_exceeds_remaining_bytes(self) -> None:
        """count=1000 passes the absolute cap (max 10_000) but can't
        fit in a body of 8 bytes (just the count field itself).
        """
        body = encode_uint64(1000)
        with pytest.raises(DecodeError, match="exceeds maximum possible"):
            ServersResponse.decode_body(body)

    def test_files_response_count_exceeds_remaining_bytes(self) -> None:
        """count=50 passes the absolute cap (max 100) but can't fit in
        a body of 8 bytes (each file requires ≥16 bytes).
        """
        body = encode_uint64(50)
        with pytest.raises(DecodeError, match="exceeds maximum possible"):
            FilesResponse.decode_body(body)

    def test_rows_response_count_exceeds_remaining_bytes(self) -> None:
        """column_count=200 passes the absolute cap (255) but cannot
        fit in a body smaller than column_count * 8 bytes.
        """
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
