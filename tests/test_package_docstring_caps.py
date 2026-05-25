"""Doc-pin: the package-level docstring's stated values for the
defense-in-depth caps must match the runtime constants.

Stale values mislead a downstream maintainer who reads ``help(dqlitewire)``
to discover the wire-cap surface and sizes buffers / mirrors caps based
on the docstring rather than chasing the constants through three modules.
"""

import re

import dqlitewire
from dqlitewire import types
from dqlitewire.messages import responses


def _extract_docstring_value(constant_name: str) -> int:
    """Return the integer value the docstring claims for ``constant_name``."""
    doc = dqlitewire.__doc__ or ""
    # The docstring lists each cap as
    # ``- ``NAME``: <int> (...)`` or similar. Match the first integer
    # after the constant name on the same paragraph.
    pattern = rf"``{re.escape(constant_name)}``\s*:\s*([0-9]+)"
    match = re.search(pattern, doc)
    assert match is not None, (
        f"Package docstring does not state a value for {constant_name}. "
        f"Update dqlitewire/__init__.py docstring to enumerate it."
    )
    return int(match.group(1))


def test_docstring_max_column_count_matches_constant() -> None:
    assert _extract_docstring_value("_MAX_COLUMN_COUNT") == responses._MAX_COLUMN_COUNT


def test_docstring_blob_and_text_caps_acknowledge_minus_64_framing_overhead() -> None:
    """The docstring must NOT claim a flat 64 MiB for the BLOB / TEXT caps.

    The actual constants subtract 64 bytes for worst-case in-row framing
    overhead. A caller sizing a payload against a flat 64 MiB number would
    trip ``EncodeError`` at the cap boundary.
    """
    doc = dqlitewire.__doc__ or ""
    flat_pattern = re.compile(
        r"``_MAX_BLOB_SIZE``\s+and\s+``_MAX_TEXT_VALUE_SIZE``\s*:\s*64 MiB each\b",
    )
    assert flat_pattern.search(doc) is None, (
        "Package docstring claims a flat 64 MiB for _MAX_BLOB_SIZE / "
        "_MAX_TEXT_VALUE_SIZE, but the runtime constants subtract 64 bytes "
        "for in-row framing overhead. Update the docstring to mention the "
        "``- 64`` reservation so callers do not size payloads against the "
        "wrong number."
    )
    # Sanity-check the constants haven't drifted from the documented
    # ``64 MiB - 64`` shape.
    expected = 64 * 1024 * 1024 - 64
    assert expected == types._MAX_BLOB_SIZE
    assert expected == types._MAX_TEXT_VALUE_SIZE
