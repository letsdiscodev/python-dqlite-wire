"""Pin: every upstream ``DQLITE_*`` 4-digit error-code define must
fall under ``_DQLITE_NAMESPACE_MAX_EXCLUSIVE`` (1024).

The Python-side namespace cap is a guess against the upstream
"10xx range" reservation. Today only three codes exist
(``DQLITE_PROTO=1001``, ``DQLITE_NOTFOUND=1002``, ``DQLITE_PARSE=1005``);
all comfortably under 1024. Upstream could legitimately add codes
above 1024 (e.g. ``DQLITE_OVERFLOW = 1030``), which would silently
fall outside ``is_dqlite_namespace_code`` — and ``primary_sqlite_code``
would then bucket-collide (``1030 & 0xFF = 6 = SQLITE_NOMEM``).

Scan the in-tree upstream checkout and fail fast if any new code
crosses the cap, so the maintainer makes a deliberate decision
(widen the cap or extend the namespace mapping) instead of
silently breaking the classifier.
"""

import re
from pathlib import Path

import pytest

from dqlitewire.constants import _DQLITE_NAMESPACE_MAX_EXCLUSIVE

# Upstream headers that historically host DQLITE_* error-code defines.
# Add new headers here if upstream relocates a define. The scan only
# considers 4-digit values (i.e. >= 1000) — non-error-code DQLITE_*
# defines (e.g. DQLITE_PROTOCOL_VERSION = 1, DQLITE_BOOLEAN = 11,
# DQLITE_REQUEST_PARAMS_SCHEMA_V0 = 0) intentionally fall outside the
# scan because they are not error codes.
_UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "dqlite-upstream" / "src"
_UPSTREAM_HEADERS = (
    _UPSTREAM_ROOT / "protocol.h",
    _UPSTREAM_ROOT / "lib" / "registry.h",
    _UPSTREAM_ROOT / "lib" / "serialize.h",
)

# ``#define DQLITE_NAME 1234`` — captures the symbol and the 4-digit
# decimal value. We deliberately constrain to four digits so version
# constants like ``DQLITE_PROTOCOL_VERSION 1`` and the 0x...-formatted
# protocol-marker constants don't match.
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
    """The ``_DQLITE_NAMESPACE_MAX_EXCLUSIVE`` (1024) cap is a pinned
    guess against the upstream "10xx range" reservation. If upstream
    ever adds a code at or above 1024, this test fails fast and the
    Python-side mapping must be updated alongside the bump."""
    found: list[tuple[str, int, Path]] = []
    for header in _UPSTREAM_HEADERS:
        text = header.read_text(encoding="utf-8")
        for match in _DEFINE_RE.finditer(text):
            symbol, raw_value = match.group(1), match.group(2)
            value = int(raw_value)
            found.append((symbol, value, header))

    # Sanity guard: the scan should find at least the three known codes
    # that exist as of this commit. A zero-match scan would indicate the
    # regex drifted or upstream relocated the defines without updating
    # ``_UPSTREAM_HEADERS``.
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
    """Scan-shape guard: the regex must continue to surface the three
    canonical codes. A regression in the regex (e.g. accidentally
    requiring ``= `` instead of whitespace) would silently make the
    out-of-range guard above vacuous."""
    text = "\n".join(p.read_text(encoding="utf-8") for p in _UPSTREAM_HEADERS)
    matches = {m.group(1): int(m.group(2)) for m in _DEFINE_RE.finditer(text)}
    assert matches.get("DQLITE_PROTO") == 1001
    assert matches.get("DQLITE_NOTFOUND") == 1002
    assert matches.get("DQLITE_PARSE") == 1005
