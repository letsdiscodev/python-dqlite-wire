"""Guard: no src/dqlitewire file carries ``from __future__ import annotations`` (3.13 floor)."""

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
