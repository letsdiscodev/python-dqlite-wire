"""Pin: ``_ConnectRequest`` class docstring names the C Raft TCP
transport handshake as the body-layout anchor.

CONNECT (type 11) is the only request type whose canonical reference
is C-only — ``internal/protocol/schema.go`` contains no Connect
directive. Without an explicit C-side anchor in the docstring, a
maintainer reconciling against a future protocol revision following
the project's Go-anchor convention reaches a dead end. Pin the
``uv_tcp_listen.c`` citation so a future docstring rewrite cannot
silently drop the anchor.
"""

from __future__ import annotations

from dqlitewire.messages.requests import _ConnectRequest


def test_connect_request_docstring_cites_c_transport_handshake() -> None:
    doc = _ConnectRequest.__doc__ or ""
    assert "uv_tcp_listen.c" in doc, (
        "_ConnectRequest docstring must cite the C Raft TCP transport handshake "
        "as the body-layout anchor; Go has no equivalent reference."
    )
