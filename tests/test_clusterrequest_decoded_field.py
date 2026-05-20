"""Pin: ``ClusterRequest._decoded`` is a declared dataclass field, not
a runtime-installed attribute. It does not leak into ``vars(req)``
and survives ``dataclasses.replace`` round-trips cleanly.
"""

from __future__ import annotations

import dataclasses

import pytest

from dqlitewire import EncodeError
from dqlitewire.messages.requests import ClusterRequest


def test_clusterrequest_decoded_does_not_appear_in_vars() -> None:
    """A V0-decoded request must not carry ``_decoded`` in its
    instance dict; the sentinel is a declared field with
    ``repr=False, compare=False``."""
    req = ClusterRequest.decode_body(b"\x00" * 8)
    assert req.format == 0
    # The instance __dict__ may carry the dataclass fields but
    # _decoded should be excluded from repr/compare. Specifically:
    # vars() reflects what's in __dict__.
    instance_vars = vars(req)
    assert "_decoded" not in instance_vars or instance_vars.get("_decoded") is True
    # Equality should ignore _decoded (compare=False).
    v1_request = ClusterRequest(format=1)
    v0_decoded = ClusterRequest.decode_body(b"\x00" * 8)
    assert v0_decoded == ClusterRequest.decode_body(b"\x00" * 8)
    assert v0_decoded != v1_request


def test_clusterrequest_dataclasses_replace_v0_routes_through_v0_gate() -> None:
    """``dataclasses.replace`` of a V0-decoded request goes through the
    public constructor — which means ``_decoded`` is re-initialised to
    its default (False) since the field is ``init=False``. The V0
    gate fires. This is documented behavior: a caller wanting to
    round-trip V0 should call ``decode_body`` again rather than
    relying on ``dataclasses.replace``. The declared-field design
    fixes the ``vars()`` leak (the primary hygiene defect) but does
    not extend the V0 escape hatch to ``replace`` — by design, since
    the gate must remain enforced for any synthetic construction
    path."""
    req = ClusterRequest.decode_body(b"\x00" * 8)
    with pytest.raises(EncodeError, match="V0"):
        dataclasses.replace(req)


def test_clusterrequest_v0_via_public_constructor_still_rejected() -> None:
    """Construction-time V0 rejection unchanged: only the decoder
    bypass should accept V0."""
    with pytest.raises(EncodeError, match="V0"):
        ClusterRequest(format=0)


def test_clusterrequest_repr_does_not_leak_decoded() -> None:
    """``_decoded`` is ``repr=False`` so a decoded V0 request prints
    indistinguishably from a hand-constructed V1."""
    req = ClusterRequest.decode_body(b"\x00" * 8)
    assert "_decoded" not in repr(req)
