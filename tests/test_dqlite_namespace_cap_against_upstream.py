"""Every upstream ``DQLITE_*`` 4-digit error-code define must stay under
``_DQLITE_NAMESPACE_MAX_EXCLUSIVE`` (1024).

A code at/above 1024 would fall outside ``is_dqlite_namespace_code`` and
bucket-collide in ``primary_sqlite_code`` (``1030 & 0xFF = 6 = SQLITE_NOMEM``).
Scan the in-tree upstream checkout and fail fast so the cap bump / namespace
mapping is a deliberate decision rather than a silent classifier break."""

import re
from pathlib import Path

import pytest

from dqlitewire.constants import _DQLITE_NAMESPACE_MAX_EXCLUSIVE

# Headers hosting DQLITE_* error-code defines; add more if upstream relocates
# one. Only 4-digit values match, excluding non-error defines like
# DQLITE_PROTOCOL_VERSION = 1.
_UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "dqlite-upstream" / "src"
_UPSTREAM_HEADERS = (
    _UPSTREAM_ROOT / "protocol.h",
    _UPSTREAM_ROOT / "lib" / "registry.h",
    _UPSTREAM_ROOT / "lib" / "serialize.h",
)

# Captures the symbol and a 4-digit decimal value; four digits excludes
# version constants and 0x-formatted protocol markers.
_DEFINE_RE = re.compile(
    r"^\s*#\s*define\s+(DQLITE_[A-Z0-9_]+)\s+(\d{4})\b",
    re.MULTILINE,
)


def _missing_headers() -> list[Path]:
    return [p for p in _UPSTREAM_HEADERS if not p.is_file()]


@pytest.mark.skipif(
    bool(_missing_headers()),
    reason=f"upstream headers not present: {_missing_headers()}",
)
def test_every_upstream_dqlite_error_code_is_below_namespace_cap() -> None:
    found: list[tuple[str, int, Path]] = []
    for header in _UPSTREAM_HEADERS:
        text = header.read_text(encoding="utf-8")
        for match in _DEFINE_RE.finditer(text):
            symbol, raw_value = match.group(1), match.group(2)
            value = int(raw_value)
            found.append((symbol, value, header))

    # A zero-match scan means the regex drifted or upstream relocated the
    # defines without updating _UPSTREAM_HEADERS.
    assert found, (
        "no DQLITE_* 4-digit defines found in upstream headers; "
        "check that _UPSTREAM_HEADERS still points at the correct "
        "files (upstream may have relocated the defines)"
    )

    out_of_range = [(s, v, p) for (s, v, p) in found if v >= _DQLITE_NAMESPACE_MAX_EXCLUSIVE]
    assert not out_of_range, (
        "Upstream defines DQLITE_* error code(s) at or above the "
        f"Python-side namespace cap ({_DQLITE_NAMESPACE_MAX_EXCLUSIVE}): "
        f"{out_of_range}. Either widen ``_DQLITE_NAMESPACE_MAX_EXCLUSIVE`` "
        "in src/dqlitewire/constants.py to cover the new code, or extend "
        "the namespace mapping in ``primary_sqlite_code`` / "
        "``is_dqlite_namespace_code`` to handle the new range explicitly."
    )


@pytest.mark.skipif(
    bool(_missing_headers()),
    reason=f"upstream headers not present: {_missing_headers()}",
)
def test_namespace_scan_finds_known_codes() -> None:
    """Guards against the regex drifting and making the out-of-range check
    above vacuous by no longer matching the canonical codes."""
    text = "\n".join(p.read_text(encoding="utf-8") for p in _UPSTREAM_HEADERS)
    matches = {m.group(1): int(m.group(2)) for m in _DEFINE_RE.finditer(text)}
    assert matches.get("DQLITE_PROTO") == 1001
    assert matches.get("DQLITE_NOTFOUND") == 1002
    assert matches.get("DQLITE_PARSE") == 1005
