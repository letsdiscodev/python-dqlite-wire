"""Pin: ``ClusterRequest.decode_body`` constructs instances through
the dataclass-generated ``__init__`` (passing ``_decoded=True`` as a
constructor kwarg), not via ``cls.__new__(cls)`` + manual field
assignment.

The ``__new__`` bypass works today but inverts the constructor's
invariants — a future maintainer who adds a frozen=True / slots=True
flip, a new field with a non-trivial default, or a property setter on
a validated field would silently break. The sibling ``ExecRequest``
family uses the constructor-kwarg sentinel pattern (``_decoded_schema``
/ ``_decoded_empty_header``); this pin asserts ``ClusterRequest``
follows the same canonical shape.
"""

from __future__ import annotations

import inspect

from dqlitewire.messages.requests import ClusterRequest


def test_cluster_request_decode_does_not_use_cls_new_bypass() -> None:
    """Mechanical guard against the ``cls.__new__`` bypass returning."""
    source = inspect.getsource(ClusterRequest)
    assert "cls.__new__" not in source, (
        "ClusterRequest.decode_body must construct via cls(format=..., _decoded=True) — "
        "the cls.__new__ bypass is fragile under frozen=True / slots=True / new-field "
        "additions and inconsistent with the sibling _decoded_schema pattern."
    )


def test_cluster_request_decode_v0_constructor_kwarg_path() -> None:
    """A V0-decoded request can also be constructed directly by passing
    ``_decoded=True`` as a kwarg — the same path the decoder uses."""
    req = ClusterRequest(format=0, _decoded=True)
    assert req.format == 0
    decoded = ClusterRequest.decode_body(b"\x00" * 8)
    assert req == decoded
