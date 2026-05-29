"""Admin-request decoders reject a non-zero schema kwarg on the direct-caller
path (the wire dispatcher already gates schema before decode_body). Mirrors
upstream C's INIT_V0 macro (gateway.c) rejecting req->schema != 0."""

from __future__ import annotations

from typing import Any

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.requests import (
    AddRequest,
    AssignRequest,
    ClientRequest,
    ClusterRequest,
    DescribeRequest,
    DumpRequest,
    FinalizeRequest,
    InterruptRequest,
    LeaderRequest,
    OpenRequest,
    RemoveRequest,
    TransferRequest,
    WeightRequest,
    _ConnectRequest,
    _HeartbeatRequest,
)
from dqlitewire.types import encode_text, encode_uint32, encode_uint64

# Each body is a valid V0 frame so the only thing causing a decode failure
# is the schema kwarg; the schema gate must fire before length/content checks.
_ADMIN_CASES = [
    (LeaderRequest, encode_uint64(0)),
    (ClientRequest, encode_uint64(0)),
    (_HeartbeatRequest, encode_uint64(0)),
    (OpenRequest, encode_text("db") + encode_uint64(0) + encode_text("vfs")),
    (FinalizeRequest, encode_uint32(0) + encode_uint32(0)),
    (InterruptRequest, encode_uint64(0)),
    (_ConnectRequest, encode_uint64(1) + encode_text("a:1")),
    (AddRequest, encode_uint64(1) + encode_text("a:1")),
    (AssignRequest, encode_uint64(1) + encode_uint64(0)),
    (RemoveRequest, encode_uint64(1)),
    (DumpRequest, encode_text("db")),
    (ClusterRequest, encode_uint64(1)),
    (TransferRequest, encode_uint64(1)),
    (DescribeRequest, encode_uint64(0)),
    (WeightRequest, encode_uint64(0)),
]


@pytest.mark.parametrize(("cls", "body"), _ADMIN_CASES, ids=[c.__name__ for c, _ in _ADMIN_CASES])
def test_admin_decoder_rejects_nonzero_schema(cls: Any, body: bytes) -> None:
    with pytest.raises(DecodeError, match="unsupported schema version"):
        cls.decode_body(body, schema=1)


@pytest.mark.parametrize(("cls", "body"), _ADMIN_CASES, ids=[c.__name__ for c, _ in _ADMIN_CASES])
def test_admin_decoder_rejects_garbage_schema(cls: Any, body: bytes) -> None:
    with pytest.raises(DecodeError, match="unsupported schema version"):
        cls.decode_body(body, schema=99)


@pytest.mark.parametrize(("cls", "body"), _ADMIN_CASES, ids=[c.__name__ for c, _ in _ADMIN_CASES])
def test_admin_decoder_accepts_default_schema(cls: Any, body: bytes) -> None:
    cls.decode_body(body)
    cls.decode_body(body, schema=0)
