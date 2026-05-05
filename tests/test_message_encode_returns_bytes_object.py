"""``Message.encode()`` returns a fully-built ``bytes`` object —
the dqliteclient layer relies on this so a ``_WireEncodeError`` on
the wire side leaves zero bytes on the writer.

Without the invariant, a streaming encoder that calls
``writer.write`` mid-encode would put partial bytes on the wire
when the encoding raises mid-flight, leaving the connection in a
half-decoded state. The downstream client's
``_run_protocol`` arm at ``connection.py:1808`` documents the
dependency.

This is a structural pin: any refactor to streaming encoding must
update both this test and the client-layer arm together.
"""

from dqlitewire.messages.base import Message
from dqlitewire.messages.requests import (
    ExecRequest,
    ExecSqlRequest,
    OpenRequest,
    PrepareRequest,
    QuerySqlRequest,
)
from dqlitewire.messages.responses import (
    EmptyResponse,
    LeaderResponse,
    ResultResponse,
    WelcomeResponse,
)


def _check_returns_bytes(msg: Message) -> None:
    encoded = msg.encode()
    assert isinstance(encoded, bytes), (
        f"{type(msg).__name__}.encode() must return a fully-built "
        f"bytes object so a mid-encoding raise leaves zero bytes on "
        f"the wire; got {type(encoded).__name__}"
    )


def test_open_request_encode_returns_bytes() -> None:
    _check_returns_bytes(OpenRequest(name="db", flags=0, vfs=""))


def test_exec_request_encode_returns_bytes() -> None:
    _check_returns_bytes(ExecRequest(db_id=1, stmt_id=1, params=[]))


def test_exec_sql_request_encode_returns_bytes() -> None:
    _check_returns_bytes(ExecSqlRequest(db_id=1, sql="SELECT 1", params=[]))


def test_query_sql_request_encode_returns_bytes() -> None:
    _check_returns_bytes(QuerySqlRequest(db_id=1, sql="SELECT 1", params=[]))


def test_prepare_request_encode_returns_bytes() -> None:
    _check_returns_bytes(PrepareRequest(db_id=1, sql="SELECT 1"))


def test_empty_response_encode_returns_bytes() -> None:
    _check_returns_bytes(EmptyResponse())


def test_welcome_response_encode_returns_bytes() -> None:
    _check_returns_bytes(WelcomeResponse(heartbeat_timeout=15000))


def test_leader_response_encode_returns_bytes() -> None:
    _check_returns_bytes(LeaderResponse(node_id=1, address="h:9001"))


def test_result_response_encode_returns_bytes() -> None:
    _check_returns_bytes(ResultResponse(last_insert_id=0, rows_affected=0))
