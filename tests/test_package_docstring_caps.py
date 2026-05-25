"""Doc-pin: the package-level docstring's stated values for the
defense-in-depth caps must match the runtime constants.

Stale values mislead a downstream maintainer who reads ``help(dqlitewire)``
to discover the wire-cap surface and sizes buffers / mirrors caps based
on the docstring rather than chasing the constants through three modules.
"""

import re

import dqlitewire
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
