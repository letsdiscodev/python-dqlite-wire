"""Pin: ``LEADER_LOST_DB_LOOKUP_SUBSTRING`` is the canonical wording
gateway.c emits when ``g->leader == NULL`` after a Raft demotion.

Paired with ``SQLITE_NOTFOUND`` (=12) and substring-matched on the
server-supplied ``raw_message``, this is the wire-side
mark-this-connection-dead signal that the client and SA layers
dispatch on. Mirrors Go's ``driverError`` ``errNotFound → ErrBadConn``
arm with the "potentially after leadership loss" comment.

The constant lives at the wire layer so the four-way disconnect-
classification chain (wire → client.connection → dbapi → SA) stays
in lockstep on rename.

The existing ``LEADER_ERROR_CODES`` cardinality pin in
``test_leader_error_codes_legacy_subcodes.py`` stays at 4: code 12
is overloaded between LOOKUP_DB (leader flip — invalidate) and
LOOKUP_STMT (server-side state bug — do not invalidate), so the
discriminator is the substring not the numeric code.
"""

from __future__ import annotations

from dqlitewire import LEADER_LOST_DB_LOOKUP_SUBSTRING


def test_leader_lost_db_lookup_substring_value() -> None:
    """Exact value pinned against upstream gateway.c
    ``LOOKUP_DB`` macro emission text."""
    assert LEADER_LOST_DB_LOOKUP_SUBSTRING == "no database opened"


def test_leader_lost_db_lookup_substring_is_str_final() -> None:
    """Same shape as the sibling ``NO_TRANSACTION_MESSAGE_SUBSTRINGS``
    and ``WIRE_DECODE_FAILED_PREFIX`` — module-level ``Final[str]``."""
    assert isinstance(LEADER_LOST_DB_LOOKUP_SUBSTRING, str)
    from dqlitewire import LEADER_LOST_DB_LOOKUP_SUBSTRING as again

    assert again is LEADER_LOST_DB_LOOKUP_SUBSTRING
