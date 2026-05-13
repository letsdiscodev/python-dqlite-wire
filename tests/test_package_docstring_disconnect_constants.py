"""Doc-pin: the package-level docstring enumerates the two disconnect-
classification constants (``WIRE_DECODE_FAILED_PREFIX`` and
``LEADER_LOST_DB_LOOKUP_SUBSTRING``).

Both constants are exported and load-bearing across the
dqliteclient / dqlitedbapi / sqlalchemydqlite consumer surface. A
rename ripples through grep; the package docstring is the canonical
discoverability surface and must enumerate them so a maintainer
debugging a disconnect-classification regression has a single
entry point.
"""

import dqlitewire


def test_package_docstring_mentions_wire_decode_failed_prefix() -> None:
    doc = dqlitewire.__doc__ or ""
    assert "WIRE_DECODE_FAILED_PREFIX" in doc, (
        "WIRE_DECODE_FAILED_PREFIX is load-bearing across four packages "
        "and must be enumerated in the dqlitewire package docstring "
        "so a maintainer can find the cross-package contract via "
        "``help(dqlitewire)``."
    )


def test_package_docstring_mentions_leader_lost_db_lookup_substring() -> None:
    doc = dqlitewire.__doc__ or ""
    assert "LEADER_LOST_DB_LOOKUP_SUBSTRING" in doc, (
        "LEADER_LOST_DB_LOOKUP_SUBSTRING is load-bearing across the "
        "client and SA dialect consumers and must be enumerated in "
        "the dqlitewire package docstring."
    )
