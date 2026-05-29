"""A V0 (``format=0``) ``ClusterRequest`` obtained via ``decode_body`` re-encodes
byte-identically (the ``_decoded=True`` sentinel), but a fresh ``ClusterRequest(format=0)``
is still rejected at construction."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.requests import ClusterRequest


def test_cluster_request_v0_decoded_round_trips_byte_identical() -> None:
    body = bytes(8)
    req = ClusterRequest.decode_body(body)
    assert req.format == 0
    assert req.encode_body() == body


def test_cluster_request_v1_round_trip_unchanged() -> None:
    body = b"\x01" + b"\x00" * 7
    req = ClusterRequest.decode_body(body)
    assert req.format == 1
    assert req.encode_body() == body


def test_cluster_request_fresh_v0_still_rejected_at_construction() -> None:
    with pytest.raises(EncodeError, match="V0"):
        ClusterRequest(format=0)
