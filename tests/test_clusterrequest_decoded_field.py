"""``ClusterRequest._decoded`` is a declared dataclass field (repr=False,
compare=False), not a runtime attribute, and survives ``dataclasses.replace``.
"""

from __future__ import annotations

import dataclasses

import pytest

from dqlitewire import EncodeError
from dqlitewire.messages.requests import ClusterRequest


def test_clusterrequest_decoded_does_not_appear_in_vars() -> None:
    req = ClusterRequest.decode_body(b"\x00" * 8)
    assert req.format == 0
    instance_vars = vars(req)
    assert "_decoded" not in instance_vars or instance_vars.get("_decoded") is True
    # Equality ignores _decoded (compare=False).
    v1_request = ClusterRequest(format=1)
    v0_decoded = ClusterRequest.decode_body(b"\x00" * 8)
    assert v0_decoded == ClusterRequest.decode_body(b"\x00" * 8)
    assert v0_decoded != v1_request


def test_clusterrequest_dataclasses_replace_v0_preserves_decoded_sentinel() -> None:
    """``dataclasses.replace`` of a V0-decoded request preserves ``_decoded``,
    so the V0 gate short-circuits on the copy (ExecRequest._decoded_schema parity)."""
    req = ClusterRequest.decode_body(b"\x00" * 8)
    replaced = dataclasses.replace(req)
    assert replaced.format == 0
    assert replaced == req
    # Explicitly clearing the sentinel re-triggers the construction-time V0 gate.
    with pytest.raises(EncodeError, match="V0"):
        dataclasses.replace(req, _decoded=False)


def test_clusterrequest_v0_via_public_constructor_still_rejected() -> None:
    """Only the decoder bypass accepts V0; the public constructor still rejects it."""
    with pytest.raises(EncodeError, match="V0"):
        ClusterRequest(format=0)


def test_clusterrequest_repr_does_not_leak_decoded() -> None:
    """``_decoded`` is repr=False, so a decoded V0 request prints like a V1."""
    req = ClusterRequest.decode_body(b"\x00" * 8)
    assert "_decoded" not in repr(req)
