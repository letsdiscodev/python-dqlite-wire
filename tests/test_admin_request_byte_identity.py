"""Admin-request encoders are byte-identity round-trippable; field-equality
round-trips miss encoder/decoder asymmetries (e.g. spurious padding)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from dqlitewire.constants import HEADER_SIZE, NodeRole
from dqlitewire.messages.base import Message
from dqlitewire.messages.requests import (
    AddRequest,
    AssignRequest,
    ClusterRequest,
    DescribeRequest,
    DumpRequest,
    FinalizeRequest,
    InterruptRequest,
    RemoveRequest,
    TransferRequest,
    WeightRequest,
)

# Lambdas wrap constructors so parametrise IDs stay stable across pytest
# versions (dataclass __repr__ changes would otherwise leak into the ID).
_CASES: list[tuple[str, Callable[[], Message]]] = [
    ("FinalizeRequest-min", lambda: FinalizeRequest(db_id=0, stmt_id=0)),
    ("FinalizeRequest-max", lambda: FinalizeRequest(db_id=0xFFFFFFFF, stmt_id=0xFFFFFFFF)),
    ("InterruptRequest-min", lambda: InterruptRequest(db_id=0)),
    ("InterruptRequest-max", lambda: InterruptRequest(db_id=0xFFFFFFFFFFFFFFFF)),
    ("AddRequest-empty-addr", lambda: AddRequest(node_id=1, address="")),
    ("AddRequest-1char", lambda: AddRequest(node_id=1, address="a")),
    ("AddRequest-7char-exact-word", lambda: AddRequest(node_id=1, address="a" * 7)),
    ("AddRequest-8char-forces-pad", lambda: AddRequest(node_id=1, address="a" * 8)),
    ("AddRequest-15char", lambda: AddRequest(node_id=1, address="a" * 15)),
    ("AddRequest-large", lambda: AddRequest(node_id=1, address="a" * 200)),
    ("AssignRequest-voter", lambda: AssignRequest(node_id=1, role=NodeRole.VOTER)),
    ("AssignRequest-standby", lambda: AssignRequest(node_id=1, role=NodeRole.STANDBY)),
    ("AssignRequest-spare", lambda: AssignRequest(node_id=1, role=NodeRole.SPARE)),
    ("AssignRequest-id0-voter", lambda: AssignRequest(node_id=0, role=NodeRole.VOTER)),
    (
        "AssignRequest-id-max",
        lambda: AssignRequest(node_id=0xFFFFFFFFFFFFFFFF, role=NodeRole.VOTER),
    ),
    ("RemoveRequest-min", lambda: RemoveRequest(node_id=0)),
    ("RemoveRequest-max", lambda: RemoveRequest(node_id=0xFFFFFFFFFFFFFFFF)),
    ("DumpRequest-empty", lambda: DumpRequest(name="")),
    ("DumpRequest-1char", lambda: DumpRequest(name="x")),
    ("DumpRequest-7char", lambda: DumpRequest(name="x" * 7)),
    ("DumpRequest-8char", lambda: DumpRequest(name="x" * 8)),
    ("DumpRequest-utf8", lambda: DumpRequest(name="café-db")),
    ("ClusterRequest-v1", lambda: ClusterRequest(format=1)),
    ("TransferRequest-min", lambda: TransferRequest(target_node_id=0)),
    ("TransferRequest-max", lambda: TransferRequest(target_node_id=0xFFFFFFFFFFFFFFFF)),
    ("DescribeRequest-v0", lambda: DescribeRequest(format=0)),
    ("WeightRequest-min", lambda: WeightRequest(weight=0)),
    ("WeightRequest-max", lambda: WeightRequest(weight=0xFFFFFFFFFFFFFFFF)),
]


@pytest.mark.parametrize(("label", "constructor"), _CASES, ids=[c[0] for c in _CASES])
def test_admin_request_encode_decode_reencode_byte_identical(
    label: str, constructor: Callable[[], Message]
) -> None:
    """Encode → decode → re-encode must produce identical bytes."""
    msg = constructor()
    encoded_1 = msg.encode()
    cls = type(msg)
    decoded = cls.decode_body(encoded_1[HEADER_SIZE:])
    encoded_2 = decoded.encode()
    assert encoded_1 == encoded_2, (
        f"{cls.__name__} ({label}) encode → decode → re-encode "
        f"is not byte-identical: {encoded_1.hex()} != {encoded_2.hex()}"
    )


def test_assign_request_legacy_shape_byte_identical_via_encode_body_legacy() -> None:
    """Byte-identity for the legacy 8-byte PROMOTE AssignRequest shape."""
    legacy_body = (42).to_bytes(8, "little")
    decoded = AssignRequest.decode_body(legacy_body)
    reencoded = decoded.encode_body_legacy()
    assert legacy_body == reencoded
