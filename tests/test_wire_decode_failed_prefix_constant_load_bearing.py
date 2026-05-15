"""Pin: ``WIRE_DECODE_FAILED_PREFIX`` is the canonical phrase the
disconnect-classification chain (wire → client.ProtocolError →
dbapi.OperationalError(code=None) → SA's
``_dqlite_disconnect_messages`` substring scan) depends on.

The constant exists as the single source of truth so a rename
ripples through grep. This test pins the canonical value so a future
rename has to update the constant deliberately.
"""

from __future__ import annotations

from dqlitewire import WIRE_DECODE_FAILED_PREFIX


def test_wire_decode_failed_prefix_value_is_canonical() -> None:
    """The exact lowercase phrase ``"wire decode failed"``. SA's
    substring scan lowercases the rendered exception text before
    matching, so emission sites can use any casing — but the
    constant's value is the canonical lowercase form that SA looks
    for."""
    assert WIRE_DECODE_FAILED_PREFIX == "wire decode failed"


def test_wire_decode_failed_prefix_is_str_final() -> None:
    """Typed ``Final[str]`` so a mypy-checked rebind is a typing
    error, mirroring the rest of the wire-constant family."""
    assert isinstance(WIRE_DECODE_FAILED_PREFIX, str)
    # Module-level Final[str] — re-import asserts identity.
    from dqlitewire import WIRE_DECODE_FAILED_PREFIX as again

    assert again is WIRE_DECODE_FAILED_PREFIX
