"""Regression guard: no source file in ``src/dqlitewire/`` carries
``from __future__ import annotations``.

The workspace floor is Python 3.13; PEP 604 union syntax and PEP 695
type-alias syntax work natively at runtime. The done item
``workspace-future-annotations-inconsistent.md`` removed four
pre-existing such imports across the workspace. ``_truncate.py``
re-introduced the import in a later commit and was cleaned up
alongside this test. Prevents future drift.
"""

import pathlib

import dqlitewire


def test_no_future_annotations_in_dqlitewire_src() -> None:
    root = pathlib.Path(dqlitewire.__file__).parent
    offenders: list[str] = []
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "from __future__ import annotations" in text:
            offenders.append(str(py.relative_to(root)))
    assert not offenders, (
        f"unexpected `from __future__ import annotations` imports in dqlitewire/src/: {offenders}"
    )
