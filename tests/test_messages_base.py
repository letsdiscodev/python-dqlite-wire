"""Tests for dqlitewire.messages.base: Header, schema tables and construction validators."""

from __future__ import annotations

import struct
from typing import ClassVar

import pytest

from dqlitewire.codec import _REQUEST_MAX_SCHEMA, _RESPONSE_MAX_SCHEMA
from dqlitewire.constants import RequestType, ResponseType
from dqlitewire.exceptions import EncodeError, ProtocolError
from dqlitewire.messages.base import Header, Message
from dqlitewire.messages.requests import (
    AddRequest,
    AssignRequest,
    ClusterRequest,
    DescribeRequest,
    DumpRequest,
    ExecRequest,
    ExecSqlRequest,
    OpenRequest,
    PrepareRequest,
    QueryRequest,
    QuerySqlRequest,
    _ConnectRequest,
)
from dqlitewire.messages.responses import NodeInfo, StmtResponse


class TestMessageEncodeAlignment:
    def test_encode_rejects_body_with_unaligned_length(self) -> None:
        """A custom ``Message`` subclass whose ``encode_body()``
        returns a length not divisible by ``WORD_SIZE`` (8 bytes)
        must raise ``EncodeError`` — silently emitting a misaligned
        frame would be rejected by the C peer's strict-decode and
        is exactly the kind of regression a custom subclass might
        introduce."""

        class _BadMessage(Message):
            MSG_TYPE: ClassVar[int] = 99

            def encode_body(self) -> bytes:
                return b"abc"  # 3 bytes — not 8-aligned

            @classmethod
            def decode_body(cls, data: bytes, schema: int = 0) -> _BadMessage:  # pragma: no cover
                raise NotImplementedError

        with pytest.raises(EncodeError, match=r"must be \d+-aligned"):
            _BadMessage().encode()

    def test_encode_accepts_body_with_aligned_length(self) -> None:
        """Sanity: an 8-byte body is accepted (negative test for the
        guard above — confirms the alignment check is the only thing
        that would have rejected the bad case)."""

        class _GoodMessage(Message):
            MSG_TYPE: ClassVar[int] = 99

            def encode_body(self) -> bytes:
                return b"\x00" * 8

            @classmethod
            def decode_body(cls, data: bytes, schema: int = 0) -> _GoodMessage:  # pragma: no cover
                raise NotImplementedError

        encoded = _GoodMessage().encode()
        assert isinstance(encoded, bytes)
        assert len(encoded) > 8  # header + body


class TestHeaderEncodeStructErrorWrap:
    """Pin ``Header.encode()``'s ``struct.error`` → ``EncodeError``
    wrap at ``messages/base.py:80-81``.

    Through normal construction, ``Header.__post_init__`` rejects any
    value that would overflow the ``<IBBH`` pack format, so the encode
    arm is a belt-and-braces defensive guard. The wire's caller layer
    catches ``EncodeError`` — not ``struct.error`` — so a regression
    that dropped the wrap (or its ``from e`` chain, or the
    ``"Failed to encode header"`` message prefix) would leak the
    lower-level exception out of the wire boundary and erode forensic
    detail. The bypass below mutates a frozen+slots dataclass via
    ``object.__setattr__`` to reach the otherwise-unreachable arm.
    """

    def _bypass_size_words(self, value: int) -> Header:
        """Construct a valid ``Header`` and then overwrite
        ``size_words`` post-construction, bypassing ``__post_init__``
        so the encode-time ``struct.pack`` raises."""
        h = Header(size_words=1, msg_type=1, schema=0, reserved=0)
        object.__setattr__(h, "size_words", value)
        return h

    def test_encode_wraps_struct_error_as_encode_error(self) -> None:
        hdr = self._bypass_size_words(2**32)
        with pytest.raises(EncodeError, match="Failed to encode header"):
            hdr.encode()

    def test_encode_preserves_struct_error_as_cause(self) -> None:
        """``EncodeError.__cause__`` carries the underlying
        ``struct.error`` so operators see the original overflow signal
        (e.g. the ``'I' format requires ...`` text from CPython)."""
        hdr = self._bypass_size_words(2**32)
        try:
            hdr.encode()
        except EncodeError as exc:
            assert isinstance(exc.__cause__, struct.error)
        else:
            pytest.fail("expected EncodeError")


# ---- merged from test_header_schema_construction_range.py ----
# Pin: ``Header.__post_init__`` accepts the full uint8 range for
# ``schema`` (0..255), not just the per-message-type ceiling.
#
# The docstring previously said "schema (0 or 1)" matching the
# per-message-type ceiling, but ``__post_init__`` only range-validates
# ``0 <= schema < 2**8``. The narrower per-type ceiling is enforced at
# decode-dispatch in ``codec.py``, not at construction time. The
# docstring now clarifies this layering; pin against a future
# re-tightening that would slip a circular import back into ``base.py``.


def test_header_construction_accepts_schema_above_per_type_ceiling() -> None:
    """A user constructing Header(schema=42) must succeed at the
    base.py layer — narrower ceilings are enforced by codec.py
    dispatch, not __post_init__."""
    h = Header(size_words=1, msg_type=0, schema=42, reserved=0)
    assert h.schema == 42


def test_header_construction_rejects_schema_above_uint8() -> None:
    with pytest.raises(EncodeError, match="schema"):
        Header(size_words=1, msg_type=0, schema=256, reserved=0)


def test_header_construction_rejects_negative_schema() -> None:
    with pytest.raises(EncodeError, match="schema"):
        Header(size_words=1, msg_type=0, schema=-1, reserved=0)


# ---- merged from test_schema_version_table_exhaustive.py ----
# Every Request/Response type has an explicit entry in the schema-
# version table OR is documented as schema-0-only.
#
# ``codec.py`` defines ``_REQUEST_MAX_SCHEMA`` and
# ``_RESPONSE_MAX_SCHEMA`` mapping each known message code to the
# maximum supported schema version. Codes not in the map default to
# ``max_schema=0`` — meaning the decoder rejects schema=1 frames the
# server might send for a future request type.
#
# Pin the contract: every value of ``RequestType`` / ``ResponseType``
# must either appear in the corresponding ``_*_MAX_SCHEMA`` dict OR
# appear in the explicit "schema-0-only" allow-list below. A future
# maintainer who adds a new message code without updating one of the
# two structures will fail this test loudly — exactly the scenario
# that's currently invisible to the suite.
#
# The schema-0-only allow-lists are authored against the C reference
# at ``dqlite-upstream/src/`` (request.h, command.h) — every entry has
# an authority citation by being declared there with no schema-1
# variant.

# Request types that are deliberately schema-0-only (no schema-1
# variant in the C reference). If a future schema-1 variant is added
# upstream, move the entry from here to ``_REQUEST_MAX_SCHEMA`` and
# the test will continue to pass.
_REQUEST_SCHEMA_ZERO_ONLY: frozenset[int] = frozenset(
    {
        RequestType.LEADER,
        RequestType.CLIENT,
        RequestType.HEARTBEAT,
        RequestType.OPEN,
        RequestType.FINALIZE,
        RequestType.INTERRUPT,
        RequestType.CONNECT,
        RequestType.ADD,
        RequestType.ASSIGN,
        RequestType.REMOVE,
        RequestType.DUMP,
        RequestType.CLUSTER,
        RequestType.TRANSFER,
        RequestType.DESCRIBE,
        RequestType.WEIGHT,
    }
)

# Response types that are deliberately schema-0-only.
_RESPONSE_SCHEMA_ZERO_ONLY: frozenset[int] = frozenset(
    {
        ResponseType.FAILURE,
        ResponseType.LEADER,
        ResponseType.WELCOME,
        ResponseType.SERVERS,
        ResponseType.DB,
        ResponseType.RESULT,
        ResponseType.ROWS,
        ResponseType.EMPTY,
        ResponseType.FILES,
        ResponseType.METADATA,
    }
)


def test_every_request_type_has_schema_entry_or_explicit_zero() -> None:
    missing = []
    for req in RequestType:
        if req in _REQUEST_MAX_SCHEMA:
            continue
        if req in _REQUEST_SCHEMA_ZERO_ONLY:
            continue
        missing.append(req)
    assert not missing, (
        f"RequestType values missing from both _REQUEST_MAX_SCHEMA and the "
        f"schema-0-only allow-list: {missing!r}. Add the entry to "
        f"_REQUEST_MAX_SCHEMA in codec.py if a schema-1 variant exists "
        f"upstream, or extend _REQUEST_SCHEMA_ZERO_ONLY in this test file "
        f"with the authority citation."
    )


def test_every_response_type_has_schema_entry_or_explicit_zero() -> None:
    missing = []
    for resp in ResponseType:
        if resp in _RESPONSE_MAX_SCHEMA:
            continue
        if resp in _RESPONSE_SCHEMA_ZERO_ONLY:
            continue
        missing.append(resp)
    assert not missing, (
        f"ResponseType values missing from both _RESPONSE_MAX_SCHEMA and the "
        f"schema-0-only allow-list: {missing!r}. Add the entry to "
        f"_RESPONSE_MAX_SCHEMA in codec.py if a schema-1 variant exists "
        f"upstream, or extend _RESPONSE_SCHEMA_ZERO_ONLY in this test file "
        f"with the authority citation."
    )


def test_request_max_schema_does_not_overlap_zero_only_allowlist() -> None:
    """Sanity: a code listed as schema-0-only must NOT also appear in
    ``_REQUEST_MAX_SCHEMA`` with a value > 0. The split lists are
    intentionally disjoint."""
    overlap = set(_REQUEST_MAX_SCHEMA) & _REQUEST_SCHEMA_ZERO_ONLY
    assert not overlap, (
        f"RequestType codes appear in both _REQUEST_MAX_SCHEMA and the "
        f"schema-0-only allow-list: {overlap!r}. Pick one."
    )


def test_response_max_schema_does_not_overlap_zero_only_allowlist() -> None:
    overlap = set(_RESPONSE_MAX_SCHEMA) & _RESPONSE_SCHEMA_ZERO_ONLY
    assert not overlap, (
        f"ResponseType codes appear in both _RESPONSE_MAX_SCHEMA and the "
        f"schema-0-only allow-list: {overlap!r}. Pick one."
    )


# ---- merged from test_construction_validators_raise_encode_error.py ----
# Construction-time wire-format validators must raise ``EncodeError``, not
# plain ``ValueError``. Pure Python argument validators (e.g.
# ``ReadBuffer.__init__``'s ``max_message_size < 1``) stay ``ValueError`` —
# caller bugs, not wire violations.


def test_encode_error_is_protocol_error_subclass() -> None:
    assert issubclass(EncodeError, ProtocolError)


class TestDecodedSchemaHintRaisesEncodeError:
    @pytest.mark.parametrize(
        "cls_name",
        ["ExecRequest", "QueryRequest", "ExecSqlRequest", "QuerySqlRequest"],
    )
    def test_unknown_decoded_schema_value(self, cls_name: str) -> None:
        cls = {
            "ExecRequest": ExecRequest,
            "QueryRequest": QueryRequest,
            "ExecSqlRequest": ExecSqlRequest,
            "QuerySqlRequest": QuerySqlRequest,
        }[cls_name]
        kwargs: dict[str, object] = {"db_id": 1, "_decoded_schema": 2}
        if cls_name in ("ExecRequest", "QueryRequest"):
            kwargs["stmt_id"] = 1
        else:
            kwargs["sql"] = "SELECT 1"
        with pytest.raises(EncodeError, match="_decoded_schema"):
            cls(**kwargs)


class TestPrepareRequestSchemaRaisesEncodeError:
    def test_unknown_schema_byte(self) -> None:
        with pytest.raises(EncodeError, match="schema must be 0 or 1"):
            PrepareRequest(db_id=1, sql="SELECT 1", schema=2)


class TestAssignRequestRoleRaisesEncodeError:
    def test_unknown_role_int(self) -> None:
        with pytest.raises(EncodeError, match="unknown role 999"):
            AssignRequest(node_id=1, role=999)


class TestClusterRequestFormatRaisesEncodeError:
    def test_format_v0_rejected(self) -> None:
        with pytest.raises(EncodeError, match="format=0.*not implemented"):
            ClusterRequest(format=0)

    @pytest.mark.parametrize("fmt", [2, 3, 255])
    def test_unknown_format(self, fmt: int) -> None:
        with pytest.raises(EncodeError, match="format must be 0"):
            ClusterRequest(format=fmt)


class TestDescribeRequestFormatRaisesEncodeError:
    def test_nonzero_format_rejected(self) -> None:
        with pytest.raises(EncodeError, match="format must be 0"):
            DescribeRequest(format=1)


class TestStmtResponseSchemaRaisesEncodeError:
    def test_unknown_schema_byte(self) -> None:
        with pytest.raises(EncodeError, match="must be 0 or 1"):
            StmtResponse(db_id=0, stmt_id=0, num_params=0, tail_offset=None, schema=2)

    def test_v0_schema_with_tail_offset(self) -> None:
        with pytest.raises(EncodeError, match="schema=0"):
            StmtResponse(db_id=0, stmt_id=0, num_params=0, tail_offset=42, schema=0)


class TestNodeInfoRoleRaisesEncodeError:
    def test_unknown_role_int(self) -> None:
        with pytest.raises(EncodeError, match="role"):
            NodeInfo(node_id=1, address="leader:9001", role=999)  # type: ignore[arg-type]


class TestConnectRequestAddressTypeRaisesEncodeError:
    """``_ConnectRequest`` gates ``address`` with an ``isinstance(str)`` check
    at construction; without it a bad value fails late at ``encode_body``
    after being stored in a field that may be repr'd into logs."""

    @pytest.mark.parametrize("bad", [b"x:1", None, 9001, ("host", 9001), ["host", 9001]])
    def test_non_str_address_rejected(self, bad: object) -> None:
        with pytest.raises(EncodeError, match="address must be str"):
            _ConnectRequest(node_id=1, address=bad)  # type: ignore[arg-type]


class TestAddRequestAddressTypeRaisesEncodeError:
    @pytest.mark.parametrize("bad", [b"node:9001", None, 9001, ("host", 9001), ["host", 9001]])
    def test_non_str_address_rejected(self, bad: object) -> None:
        with pytest.raises(EncodeError, match="address must be str"):
            AddRequest(node_id=2, address=bad)  # type: ignore[arg-type]


class TestDumpRequestNameTypeRaisesEncodeError:
    @pytest.mark.parametrize("bad", [b"db.sqlite", None, 0, ["db"], (1, 2)])
    def test_non_str_name_rejected(self, bad: object) -> None:
        with pytest.raises(EncodeError, match="name must be str"):
            DumpRequest(name=bad)  # type: ignore[arg-type]


class TestOpenRequestNameTypeRaisesEncodeError:
    """``OpenRequest`` gates ``name``/``vfs`` with ``isinstance(str)`` at
    construction; without it a bad value fails late at ``encode_body`` after
    being stored in a field that may be repr'd into logs."""

    @pytest.mark.parametrize("bad", [b"db.sqlite", None, 0, ["db"], (1, 2)])
    def test_non_str_name_rejected(self, bad: object) -> None:
        with pytest.raises(EncodeError, match="OpenRequest.name must be str"):
            OpenRequest(name=bad, flags=0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [b"vfs", None, 0, ["vfs"], (1, 2)])
    def test_non_str_vfs_rejected(self, bad: object) -> None:
        with pytest.raises(EncodeError, match="OpenRequest.vfs must be str"):
            OpenRequest(name="db", flags=0, vfs=bad)  # type: ignore[arg-type]

    def test_ordinary_str_name_and_vfs_accepted(self) -> None:
        req = OpenRequest(name="users_db", flags=0, vfs="")
        assert req.name == "users_db"
        assert req.vfs == ""


# ``ServersResponse.decode_body``'s ``unknown_role_policy`` raises
# ``DecodeError`` (decode-path), not ``EncodeError``; pinned in
# test_servers_response_unknown_role_policy.py.
