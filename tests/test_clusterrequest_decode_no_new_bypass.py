"""``ClusterRequest.decode_body`` must construct via the dataclass __init__
(``_decoded=True`` kwarg), not ``cls.__new__`` — the bypass is fragile under
frozen/slots/new-field changes and breaks the sibling sentinel-pattern parity.
"""

from __future__ import annotations

import inspect

from dqlitewire.messages.requests import ClusterRequest


def test_cluster_request_decode_does_not_use_cls_new_bypass() -> None:
    source = inspect.getsource(ClusterRequest)
    assert "cls.__new__" not in source, (
        "ClusterRequest.decode_body must construct via cls(format=..., _decoded=True) — "
        "the cls.__new__ bypass is fragile under frozen=True / slots=True / new-field "
        "additions and inconsistent with the sibling _decoded_schema pattern."
    )


def test_cluster_request_decode_v0_constructor_kwarg_path() -> None:
    """A V0 request constructs directly via ``_decoded=True`` — the decoder's path."""
    req = ClusterRequest(format=0, _decoded=True)
    assert req.format == 0
    decoded = ClusterRequest.decode_body(b"\x00" * 8)
    assert req == decoded
