"""Pin: ``_HeartbeatRequest`` class docstring cites Go's
``schema.go`` as the authoritative byte-shape anchor and does NOT
frame the layout as "speculative".

Every other request class in this package anchors to Go's schema
directive; the prior "speculative" wording was a documentation
asymmetry that misled readers into thinking the byte layout had no
spec source. Pin the corrected wording so a future rewrite cannot
silently re-introduce the misleading framing.
"""

from __future__ import annotations

from dqlitewire.messages.requests import _HeartbeatRequest


def test_heartbeat_request_docstring_does_not_call_layout_speculative() -> None:
    doc = _HeartbeatRequest.__doc__ or ""
    assert "speculative" not in doc.lower()


def test_heartbeat_request_docstring_cites_go_schema() -> None:
    doc = _HeartbeatRequest.__doc__ or ""
    assert "schema.go" in doc
    assert "Go-canonical" in doc
