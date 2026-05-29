"""Pin: response dataclasses with unbounded list / dict fields emit a
bounded summary ``__repr__`` rather than enumerating every element
(the dataclass-generated repr would produce multi-megabyte strings)."""

from __future__ import annotations

from dqlitewire.constants import NodeRole, ValueType
from dqlitewire.messages.responses import (
    FilesResponse,
    NodeInfo,
    RowsResponse,
    ServersResponse,
)
from dqlitewire.types import WireValue


def test_rows_response_repr_is_bounded_under_large_rows() -> None:
    rows: list[list[WireValue]] = [[i, f"name-{i}"] for i in range(10_000)]
    row_types = [[ValueType.INTEGER, ValueType.TEXT] for _ in range(10_000)]
    msg = RowsResponse(
        column_names=["id", "name"],
        column_types=[ValueType.INTEGER, ValueType.TEXT],
        row_types=row_types,
        rows=rows,
        has_more=False,
    )
    rendered = repr(msg)

    assert len(rendered) < 500, (
        f"RowsResponse repr is {len(rendered)} chars for 10k rows; expected a bounded summary"
    )
    assert "<10000 items>" in rendered
    assert "has_more=False" in rendered


def test_files_response_repr_is_bounded_under_large_per_file_content() -> None:
    """A 50-file response with multi-MB content per file must
    produce a short repr that does not dump the bytes."""
    files = {f"file-{i:04d}": b"\x00" * (1 << 20) for i in range(50)}
    msg = FilesResponse(files=files)
    rendered = repr(msg)

    assert len(rendered) < 500, (
        f"FilesResponse repr is {len(rendered)} chars for 50 × 1MB files; "
        "expected a bounded summary"
    )
    # Truncated form should show a small head sample and an "and N more"
    # marker.
    assert "more" in rendered
    assert "bytes" in rendered


def test_servers_response_repr_is_bounded_under_large_node_count() -> None:
    """A 1000-node response must produce a short repr that does not
    enumerate every entry."""
    nodes = [
        NodeInfo(node_id=i + 1, address=f"10.0.0.{i // 256}.{i % 256}:9001", role=NodeRole.VOTER)
        for i in range(1000)
    ]
    msg = ServersResponse(nodes=nodes)
    rendered = repr(msg)

    assert len(rendered) < 500, (
        f"ServersResponse repr is {len(rendered)} chars for 1k nodes; expected a bounded summary"
    )
    assert "more" in rendered


def test_rows_response_small_repr_still_includes_field_names() -> None:
    """Small payloads still render the field names + summary count
    so the summary is informative."""
    msg = RowsResponse(
        column_names=["x"],
        column_types=[ValueType.INTEGER],
        rows=[[1], [2]],
        has_more=True,
    )
    rendered = repr(msg)
    assert "column_names=['x']" in rendered
    assert "<2 items>" in rendered
    assert "has_more=True" in rendered


def test_files_response_empty_repr() -> None:
    """Empty FilesResponse renders cleanly."""
    msg = FilesResponse()
    rendered = repr(msg)
    assert "FilesResponse" in rendered


def test_servers_response_empty_repr() -> None:
    """Empty ServersResponse renders cleanly."""
    msg = ServersResponse()
    rendered = repr(msg)
    assert "ServersResponse" in rendered
