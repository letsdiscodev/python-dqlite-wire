"""Pin: admin-request decoders reject non-zero schema kwarg.

The wire dispatcher (``codec.py``) consults ``_REQUEST_MAX_SCHEMA`` to
decide whether a given ``msg_type + header.schema`` combination is
admissible before dispatching to ``decode_body``. That table only
contains the five params-schema-aware codes (PrepareRequest,
ExecRequest, QueryRequest, ExecSqlRequest, QuerySqlRequest); admin
request types fall through to the default ``0``. So the wire path is
already safe.

The gap pinned here is on the **direct-caller** path: a negative-path
test, mock-server fixture, captured-traffic replay tool, or fuzz
harness that calls ``<Class>.decode_body(body, schema=N)`` directly
must get a ``DecodeError`` for ``N != 0``, mirroring upstream C's
``INIT_V0`` macro (``gateway.c:109-122``) which rejects
``req->schema != 0`` for the canonical-INIT_V0 handlers with
``DQLITE_PARSE`` before calling ``request_##REQ##__decode``.
"""

from __future__ import annotations

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

# (class, body) — each body is a *valid* V0 frame for its class so the
# only mutation that should cause the decode to fail is the schema
# kwarg. If the schema gate fires, the length / content checks must
# never have a chance to fire.
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
def test_admin_decoder_rejects_nonzero_schema(cls: type, body: bytes) -> None:
    with pytest.raises(DecodeError, match="unsupported schema version"):
        cls.decode_body(body, schema=1)


@pytest.mark.parametrize(("cls", "body"), _ADMIN_CASES, ids=[c.__name__ for c, _ in _ADMIN_CASES])
def test_admin_decoder_rejects_garbage_schema(cls: type, body: bytes) -> None:
    with pytest.raises(DecodeError, match="unsupported schema version"):
        cls.decode_body(body, schema=99)


@pytest.mark.parametrize(("cls", "body"), _ADMIN_CASES, ids=[c.__name__ for c, _ in _ADMIN_CASES])
def test_admin_decoder_accepts_default_schema(cls: type, body: bytes) -> None:
    cls.decode_body(body)
    cls.decode_body(body, schema=0)
