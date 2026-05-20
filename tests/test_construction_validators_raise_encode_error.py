"""Pin the public-exception taxonomy for wire-format invariant
validators that fire at dataclass-construction time.

Each cited site is a wire-format gate: the value crossing the gate
either came off the wire or is about to be put on the wire. The
canonical public exception class for "wire layer cannot serialise
this value" is ``EncodeError`` (a ``ProtocolError`` subclass exported
from ``dqlitewire.exceptions``). Pre-fix these construction-time
validators raised plain ``ValueError``, splitting the public taxonomy
between two unrelated classes (``ValueError`` is not in any
parent/child relationship with ``EncodeError``).

Pure Python-level argument validators (e.g. ``ReadBuffer.__init__``'s
``max_message_size < 1``) intentionally remain ``ValueError`` —
those are caller bugs that don't represent a wire violation.
"""

import pytest

from dqlitewire.exceptions import EncodeError, ProtocolError
from dqlitewire.messages.requests import (
    AddRequest,
    AssignRequest,
    ClusterRequest,
    DescribeRequest,
    DumpRequest,
    ExecRequest,
    ExecSqlRequest,
    PrepareRequest,
    QueryRequest,
    QuerySqlRequest,
)
from dqlitewire.messages.responses import NodeInfo, StmtResponse


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


# NOTE: ``ServersResponse.decode_body`` validates the
# ``unknown_role_policy`` kwarg and raises ``DecodeError`` (not
# ``EncodeError``) — every error from a ``decode_*`` path surfaces
# under ``DecodeError``. The pin lives at
# ``test_servers_response_unknown_role_policy.py::test_invalid_policy_raises_decode_error``.
