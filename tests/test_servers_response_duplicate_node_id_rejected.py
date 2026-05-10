"""Pin: ``ServersResponse.decode_body`` rejects duplicate ``node_id``
values mid-stream.

The sibling ``FilesResponse.decode_body`` rejects duplicate filenames;
``ServersResponse`` was previously silent on duplicate node IDs.
Conforming C/Go servers populate this response from the Raft log,
which guarantees unique IDs by construction — a duplicate would only
surface from a misframed or hostile peer. Detecting it here keeps the
strictness uniform with the sibling decoder and prevents downstream
consumers (a ``dict`` keyed by ``node_id``) from silently overwriting
a prior entry.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import ServersResponse
from dqlitewire.types import encode_text, encode_uint64


def _build_body_with_duplicate_ids() -> bytes:
    """Two nodes with the same node_id but different addresses —
    illegal per the Raft log uniqueness invariant."""
    return (
        encode_uint64(2)  # node count
        + encode_uint64(1)  # node 1
        + encode_text("addr1:9001")
        + encode_uint64(0)  # voter
        + encode_uint64(1)  # node 1 (DUPLICATE id)
        + encode_text("addr2:9002")
        + encode_uint64(0)
    )


def test_duplicate_node_id_rejected() -> None:
    body = _build_body_with_duplicate_ids()
    with pytest.raises(DecodeError, match="Duplicate node_id 1"):
        ServersResponse.decode_body(body)


def test_unique_node_ids_accepted() -> None:
    """Negative pin: legitimate distinct IDs decode cleanly."""
    body = (
        encode_uint64(2)
        + encode_uint64(1)
        + encode_text("addr1:9001")
        + encode_uint64(0)
        + encode_uint64(2)
        + encode_text("addr2:9002")
        + encode_uint64(0)
    )
    response = ServersResponse.decode_body(body)
    assert len(response.nodes) == 2
    assert response.nodes[0].node_id == 1
    assert response.nodes[1].node_id == 2
