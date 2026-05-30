"""``ClusterRequest.decode_body`` constructs via the dataclass __init__
(``_decoded=True`` kwarg); a V0 request round-trips through that path.
"""

from __future__ import annotations

from dqlitewire.messages.requests import ClusterRequest


def test_cluster_request_decode_v0_constructor_kwarg_path() -> None:
    """A V0 request constructs directly via ``_decoded=True`` — the decoder's path."""
    req = ClusterRequest(format=0, _decoded=True)
    assert req.format == 0
    decoded = ClusterRequest.decode_body(b"\x00" * 8)
    assert req == decoded
