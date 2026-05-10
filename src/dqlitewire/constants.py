"""Protocol constants for dqlite wire protocol."""

from enum import IntEnum
from typing import Final

__all__ = [
    "DEFAULT_MAX_CONTINUATION_FRAMES",
    "DEFAULT_MAX_TOTAL_ROWS",
    "DQLITE_NOTFOUND",
    "DQLITE_PARSE",
    "DQLITE_PROTO",
    "HEADER_SIZE",
    "LEADER_ERROR_CODES",
    "NO_TRANSACTION_MESSAGE_SUBSTRINGS",
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_LEGACY",
    "ROW_DONE_BYTE",
    "ROW_DONE_MARKER",
    "ROW_PART_BYTE",
    "ROW_PART_MARKER",
    "SQLITE_ABORT",
    "SQLITE_AUTH",
    "SQLITE_BUSY",
    "SQLITE_CONSTRAINT",
    "SQLITE_CONSTRAINT_CHECK",
    "SQLITE_CONSTRAINT_COMMITHOOK",
    "SQLITE_CONSTRAINT_FOREIGNKEY",
    "SQLITE_CONSTRAINT_FUNCTION",
    "SQLITE_CONSTRAINT_NOTNULL",
    "SQLITE_CONSTRAINT_PINNED",
    "SQLITE_CONSTRAINT_PRIMARYKEY",
    "SQLITE_CONSTRAINT_ROWID",
    "SQLITE_CONSTRAINT_TRIGGER",
    "SQLITE_CONSTRAINT_UNIQUE",
    "SQLITE_CONSTRAINT_VTAB",
    "SQLITE_CORRUPT",
    "SQLITE_ERROR",
    "SQLITE_FORMAT",
    "SQLITE_FULL",
    "SQLITE_INTERNAL",
    "SQLITE_INTERRUPT",
    "SQLITE_IOERR",
    "SQLITE_IOERR_LEADERSHIP_LOST",
    "SQLITE_IOERR_LEADERSHIP_LOST_LEGACY",
    "SQLITE_IOERR_NOT_LEADER",
    "SQLITE_IOERR_NOT_LEADER_LEGACY",
    "SQLITE_MISMATCH",
    "SQLITE_MISUSE",
    "SQLITE_NOLFS",
    "SQLITE_NOMEM",
    "SQLITE_NOTADB",
    "SQLITE_NOTFOUND",
    "SQLITE_NOTICE",
    "SQLITE_PRIMARY_CODE_MASK",
    "SQLITE_PROTOCOL",
    "SQLITE_RANGE",
    "SQLITE_TOOBIG",
    "SQLITE_WARNING",
    "TX_AUTO_ROLLBACK_PRIMARY_CODES",
    "WIRE_DECODE_FAILED_PREFIX",
    "WORD_SIZE",
    "NodeRole",
    "RequestType",
    "ResponseType",
    "ValueType",
    "is_dqlite_namespace_code",
    "primary_sqlite_code",
]

# Protocol versions
PROTOCOL_VERSION: Final[int] = 1
PROTOCOL_VERSION_LEGACY: Final[int] = 0x86104DD760433FE5  # Pre-1.0 dqlite servers

# Word size in bytes (all messages are padded to 8-byte boundaries)
WORD_SIZE: Final[int] = 8

# Header size in bytes
HEADER_SIZE: Final[int] = 8

# Row markers — written as full uint64 words on the wire. Detection
# validates all 8 bytes of the sentinel (via ``_ROW_DONE_MARKER`` /
# ``_ROW_PART_MARKER`` bytes in ``tuples.py``); the original first-byte
# check was tightened in the ``row-marker-full-word`` fix to close a
# misclassification window where a legitimate value starting with 0xFF
# or 0xEE could be confused with a marker. Use the uint64 constants
# below for encoding and the single-byte constants to build the detection
# byte sequences.
ROW_DONE_BYTE: Final[int] = 0xFF
ROW_PART_BYTE: Final[int] = 0xEE
ROW_DONE_MARKER: Final[int] = 0xFFFFFFFFFFFFFFFF
ROW_PART_MARKER: Final[int] = 0xEEEEEEEEEEEEEEEE


class RequestType(IntEnum):
    """Client to server message types."""

    LEADER = 0
    CLIENT = 1
    HEARTBEAT = 2
    OPEN = 3
    PREPARE = 4
    EXEC = 5
    QUERY = 6
    FINALIZE = 7
    EXEC_SQL = 8
    QUERY_SQL = 9
    INTERRUPT = 10
    CONNECT = 11  # Raft transport connection (node-to-node, not in Go client)
    ADD = 12
    ASSIGN = 13
    REMOVE = 14
    DUMP = 15
    CLUSTER = 16
    TRANSFER = 17
    DESCRIBE = 18
    WEIGHT = 19


class ResponseType(IntEnum):
    """Server to client message types."""

    FAILURE = 0
    LEADER = 1  # Also called NODE (NodeLegacy is also 1 in Go)
    WELCOME = 2
    SERVERS = 3  # Also called NODES
    DB = 4
    STMT = 5
    RESULT = 6
    ROWS = 7
    EMPTY = 8
    FILES = 9
    METADATA = 10


class ValueType(IntEnum):
    """Value types for parameters and row values."""

    INTEGER = 1
    FLOAT = 2
    TEXT = 3
    BLOB = 4
    NULL = 5
    # Unix time (deprecated, maps to INTEGER).
    # Server-to-client ONLY: the upstream C server emits UNIXTIME from
    # ``query.c`` for DATETIME columns, but the upstream C tuple_decoder
    # (``tuple.c``) has no inbound case for DQLITE_UNIXTIME and rejects
    # it with DQLITE_PARSE. This Python decoder is strictly more
    # permissive than the C client — it accepts UNIXTIME on incoming row
    # values and returns int64 seconds-since-epoch. Mock servers built
    # on this encoder must NOT send UNIXTIME to real C clients.
    # ``encode_params_tuple`` additionally enforces that this tag never
    # appears on outgoing parameters (the server's parameter parser
    # rejects it too).
    UNIXTIME = 9
    ISO8601 = 10  # ISO8601 string (maps to TEXT)
    BOOLEAN = 11  # Boolean (maps to INTEGER)


class NodeRole(IntEnum):
    """Node roles in a dqlite cluster.

    Matches Go's protocol.Voter/StandBy/Spare and C's
    DQLITE_VOTER/DQLITE_STANDBY/DQLITE_SPARE.
    """

    VOTER = 0
    STANDBY = 1
    SPARE = 2


# SQLite primary error codes upstream's gateway emits via
# ``failure(req, ...)`` calls. Exported for grep-friendliness so
# downstream callers can ``code == SQLITE_BUSY`` instead of
# ``code == 5``. Source: ``dqlite-upstream/src/gateway.c`` emit
# sites, ``include/dqlite.h`` (extended IOERR variants).
SQLITE_ERROR: Final[int] = 1  # generic SQL error (e.g. nonempty statement tail)
SQLITE_BUSY: Final[int] = 5  # busy retry-or-fail — engine-side OR Raft-side
SQLITE_NOTFOUND: Final[int] = 12  # gateway.c LOOKUP_DB / LOOKUP_STMT
SQLITE_PROTOCOL: Final[int] = 15  # gateway.c "bad format version" / wire mismatch

# SQLite extended error codes that signal leader changes in a dqlite
# cluster. Upstream definitions in ``dqlite-upstream/include/dqlite.h``:
#
#     #define SQLITE_IOERR_NOT_LEADER       (SQLITE_IOERR | (40 << 8))
#     #define SQLITE_IOERR_LEADERSHIP_LOST  (SQLITE_IOERR | (41 << 8))
#
# where ``SQLITE_IOERR = 10``. Callers (``dqliteclient`` and
# ``sqlalchemy-dqlite``) import these to decide whether to invalidate a
# connection and retry against a fresh leader.
SQLITE_IOERR: Final[int] = 10
SQLITE_IOERR_NOT_LEADER: Final[int] = SQLITE_IOERR | (40 << 8)  # 10250
SQLITE_IOERR_LEADERSHIP_LOST: Final[int] = SQLITE_IOERR | (41 << 8)  # 10506

# Legacy sub-code variants emitted by pre-3.32.1 dqlite servers.
# Go's ``go-dqlite/driver/driver.go::driverError`` carries both the
# modern (40/41) and the legacy (32/33) sub-codes through the same
# ``ErrBadConn`` arm so a mixed cluster (one peer on a pre-3.32.1
# server, others on the modern build) classifies leader-flips
# uniformly. Without these, a leader flip on a legacy peer surfaces
# as a non-retryable ``OperationalError`` rather than a retryable
# ``DqliteConnectionError`` — `cluster.connect`'s retry tuple
# excludes ``OperationalError`` and the SA dialect's
# ``is_disconnect`` does not classify it as a disconnect, so the
# pool returns a broken connection.
# WARNING: numerical collision with stdlib sqlite3.
# - SQLITE_IOERR_NOT_LEADER_LEGACY = 8202 collides with
#   ``sqlite3.SQLITE_IOERR_DATA`` (assigned in SQLite 3.31).
# - SQLITE_IOERR_LEADERSHIP_LOST_LEGACY = 8458 collides with
#   ``sqlite3.SQLITE_IOERR_CORRUPTFS`` (assigned in SQLite 3.36).
#
# This is an upstream-protocol artifact: pre-3.32.1 dqlite assigned
# subcodes 32/33 for leader semantics before SQLite assigned semantic
# meaning to those sub-codes. A legacy-cluster operator reading
# ``code == sqlite3.SQLITE_IOERR_DATA`` is being misled — the value
# is identical, but the meaning is "leader changed", not "I/O data
# error". Documented here so a future audit considering re-exporting
# the stdlib ``SQLITE_IOERR_*`` family at the wire layer (or adding
# alias named exports like ``SQLITE_IOERR_DATA = 8202``) does not
# silently amplify the collision footprint.
SQLITE_IOERR_NOT_LEADER_LEGACY: Final[int] = SQLITE_IOERR | (32 << 8)  # 8202
SQLITE_IOERR_LEADERSHIP_LOST_LEGACY: Final[int] = SQLITE_IOERR | (33 << 8)  # 8458

LEADER_ERROR_CODES: Final[frozenset[int]] = frozenset(
    {
        SQLITE_IOERR_NOT_LEADER,
        SQLITE_IOERR_LEADERSHIP_LOST,
        SQLITE_IOERR_NOT_LEADER_LEGACY,
        SQLITE_IOERR_LEADERSHIP_LOST_LEGACY,
    }
)

# dqlite-namespace error codes (>= 1000). These do NOT belong to the
# SQLite primary-code namespace and ``primary_sqlite_code`` returns
# them unchanged so a downstream ``code == primary_sqlite_code(...)``
# check does not collide with a real SQLite primary. Source:
# ``dqlite-upstream/src/protocol.h`` (DQLITE_PROTO),
# ``dqlite-upstream/src/lib/registry.h`` (DQLITE_NOTFOUND), and
# ``dqlite-upstream/src/lib/serialize.h`` (DQLITE_PARSE).
DQLITE_PROTO: Final[int] = 1001  # protocol.h:9 — Raft FSM-internal protocol error.
# Currently emitted only inside command.c / fsm.c
# apply paths and never reaches gateway.c::failure(),
# but included here for namespace completeness so a
# future change that surfaces it through the gateway
# passes through ``primary_sqlite_code`` cleanly.
DQLITE_NOTFOUND: Final[int] = 1002  # registry lookup miss (server-side scratch)
DQLITE_PARSE: Final[int] = 1005  # gateway.c::INIT_V0 — unrecognized request type
# or unrecognized schema for the eleven canonical-INIT_V0 handlers.
# The five params-schema-aware handlers (handle_prepare /
# handle_exec / handle_query / handle_exec_sql / handle_query_sql)
# bypass INIT_V0 to admit both V0 and V1 params schema, and on
# unrecognized schema instead emit ``SQLITE_ERROR`` (= 1) with the
# same ``"unrecognized schema version"`` message. Tests synthesising
# the failure shape for those five handlers must pin code=1, not 1005.

# Bit mask for extracting the primary (low-byte) code from an
# extended SQLite error code. Upstream encodes extended codes as
# ``primary | (sub << 8)``; ``code & SQLITE_PRIMARY_CODE_MASK``
# recovers the primary. Use via :func:`primary_sqlite_code`.
SQLITE_PRIMARY_CODE_MASK: Final[int] = 0xFF


# dqlite-namespace error codes live in ``[1000, 1024)``. Extended
# SQLite codes (``SQLITE_IOERR_NOT_LEADER = 10250``) never land in
# this range — extended codes are ``primary | (subcode << 8)`` and
# the only way to produce a value in [1000, 1024) is primary=232..255
# with subcode=3, but valid SQLite primaries top out at ~28. The
# range check auto-includes any future upstream namespace code
# without a manual sync commit (the historic enumeration was
# ``{DQLITE_PROTO=1001, DQLITE_NOTFOUND=1002, DQLITE_PARSE=1005}``).
_DQLITE_NAMESPACE_MIN: Final[int] = 1000
_DQLITE_NAMESPACE_MAX_EXCLUSIVE: Final[int] = 1024


def is_dqlite_namespace_code(code: int) -> bool:
    """True if ``code`` is a dqlite-namespace error code.

    dqlite uses non-SQLite codes for server-internal failure shapes
    (``DQLITE_PARSE = 1005``, ``DQLITE_NOTFOUND = 1002``,
    ``DQLITE_PROTO = 1001``). The discriminator is a numeric range:
    namespace codes live in ``[1000, 1024)``, which extended SQLite
    codes never enter (extended = ``primary | (subcode << 8)``, and
    no valid SQLite primary lands at 232–255). A range check
    auto-includes any future namespace addition upstream without
    requiring a manual sync commit.
    """
    return _DQLITE_NAMESPACE_MIN <= code < _DQLITE_NAMESPACE_MAX_EXCLUSIVE


def primary_sqlite_code(code: int) -> int:
    """Return the primary SQLite error code from an extended code.

    ``SQLITE_IOERR_NOT_LEADER`` (``10250``) → ``SQLITE_IOERR`` (``10``).
    A primary code (low byte only) passes through unchanged.

    dqlite-namespace codes (``DQLITE_PARSE``, ``DQLITE_NOTFOUND``)
    are returned UNCHANGED instead of masked. Masking
    ``DQLITE_PARSE = 1005`` with 0xFF yields 237 — not a real SQLite
    primary — and a future SQLite primary code 237 would silently
    collide. Use :func:`is_dqlite_namespace_code` to dispatch on
    those codes deliberately.

    Matches the ``code & 0xFF`` pattern used by dqlitedbapi for its
    ``Error`` subclass dispatch and by the no-transaction mask.
    Exposed so callers don't have to recompute the shift inline and
    so a future change to the extended-code layout has a single
    point of rework.
    """
    if is_dqlite_namespace_code(code):
        return code
    return code & SQLITE_PRIMARY_CODE_MASK


# Primary SQLite error codes whose semantics imply server-side
# auto-rollback of the active transaction. The dqlite leader's polling
# of ``sqlite3_txn_state`` observes ``SQLITE_TXN_NONE`` after any of
# these, so the cluster-side tx is gone and the client must clear its
# tracking state. Source: SQLite docs at
# https://www.sqlite.org/lang_transaction.html#response_to_errors_within_a_transaction
# documents NOMEM, IOERR, INTERRUPT, FULL as auto-rollback. ABORT and
# CORRUPT are NOT on SQLite's documented list but are kept here as
# defensive over-clears: false-positive auto-rollback is benign (pool
# reset would do it anyway), false-negative leaves tx state stale.
# BUSY is intentionally NOT included — dqlite has TWO BUSY origins
# (engine-side vs Raft-side) that aren't distinguishable from the wire
# without inspecting the message text; the client-side handler at
# ``dqliteclient.connection`` carves out the Raft-side
# "checkpoint in progress" case explicitly.
SQLITE_ABORT: Final[int] = 4  # operation aborted (e.g., sqlite3_interrupt) — defensive
SQLITE_NOMEM: Final[int] = 7  # out of memory
SQLITE_INTERRUPT: Final[int] = 9  # query interrupted via INTERRUPT
SQLITE_CORRUPT: Final[int] = 11  # database disk image malformed — defensive
SQLITE_FULL: Final[int] = 13  # database/disk full
# Auxiliary database-file format errors. Both surface from the engine
# when the file on disk cannot be read as a SQLite database — schema
# version mismatch, header magic mismatch, or a non-database file
# opened by mistake. The SA dialect treats codes 11/24/26 as
# slot-fatal during pre-ping (``do_ping``) and as disconnect-class
# during failure dispatch (``is_disconnect``); keeping the constants
# in the wire layer alongside the other SQLite primaries lets both
# the dialect and the dbapi import them by name instead of inlining
# magic literals.
SQLITE_FORMAT: Final[int] = 24
SQLITE_NOTADB: Final[int] = 26


# Additional primary SQLite codes the dbapi PEP 249 classifier
# dispatches on (cursor.py's ``_call_client`` arms). Previously
# duplicated as ``cursor.py``-private ``_SQLITE_*`` literals; promoting
# them to the wire layer keeps the values in one place and lets the
# dbapi import them by name (sibling pattern to ``SQLITE_NOTFOUND`` /
# ``SQLITE_NOMEM`` which are already public).
SQLITE_INTERNAL: Final[int] = 2
SQLITE_TOOBIG: Final[int] = 18
SQLITE_CONSTRAINT: Final[int] = 19
SQLITE_MISMATCH: Final[int] = 20
SQLITE_MISUSE: Final[int] = 21
SQLITE_NOLFS: Final[int] = 22
SQLITE_AUTH: Final[int] = 23
SQLITE_RANGE: Final[int] = 25
SQLITE_NOTICE: Final[int] = 27
SQLITE_WARNING: Final[int] = 28


# Extended ``SQLITE_CONSTRAINT_*`` family (high byte 1..11). Stdlib
# `sqlite3` exposes these for branching on specific constraint
# violations (UNIQUE / FOREIGNKEY / NOTNULL / CHECK / etc.) — the
# integer values arrive at the dbapi caller via the wire round-trip,
# but the symbolic names were missing. Cross-driver porting code
# doing ``e.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE``
# previously required ``import sqlite3`` separately; expose the names
# at the wire layer so callers can use ``dqlitewire.SQLITE_CONSTRAINT_*``
# uniformly. Values match stdlib bit-for-bit.
SQLITE_CONSTRAINT_CHECK: Final[int] = SQLITE_CONSTRAINT | (1 << 8)  # 275
SQLITE_CONSTRAINT_COMMITHOOK: Final[int] = SQLITE_CONSTRAINT | (2 << 8)  # 531
SQLITE_CONSTRAINT_FOREIGNKEY: Final[int] = SQLITE_CONSTRAINT | (3 << 8)  # 787
SQLITE_CONSTRAINT_FUNCTION: Final[int] = SQLITE_CONSTRAINT | (4 << 8)  # 1043
SQLITE_CONSTRAINT_NOTNULL: Final[int] = SQLITE_CONSTRAINT | (5 << 8)  # 1299
SQLITE_CONSTRAINT_PRIMARYKEY: Final[int] = SQLITE_CONSTRAINT | (6 << 8)  # 1555
SQLITE_CONSTRAINT_TRIGGER: Final[int] = SQLITE_CONSTRAINT | (7 << 8)  # 1811
SQLITE_CONSTRAINT_UNIQUE: Final[int] = SQLITE_CONSTRAINT | (8 << 8)  # 2067
SQLITE_CONSTRAINT_VTAB: Final[int] = SQLITE_CONSTRAINT | (9 << 8)  # 2323
SQLITE_CONSTRAINT_ROWID: Final[int] = SQLITE_CONSTRAINT | (10 << 8)  # 2579
SQLITE_CONSTRAINT_PINNED: Final[int] = SQLITE_CONSTRAINT | (11 << 8)  # 2835


# Message-text substrings that mark a benign "no transaction was
# active" reply from the server. The full reply is a SQLite-engine-
# emitted ``SQLITE_ERROR`` (primary code 1) with one of these clauses.
# Both the dbapi (``commit``/``rollback`` swallow) and the client
# layer (``transaction()`` context manager / ``_reset_connection``)
# need to recognise this shape; the substrings are pinned here so
# both layers cannot drift apart.
#
# The integration test ``test_no_transaction_error_wording.py``
# (in dbapi) verifies the wording against a live cluster — this
# constant is the single source of truth that test inspects.
#
# A primary-code-1 substring guard is also why ``DQLITE_ERROR=1``
# (which shares the wire low-byte with ``SQLITE_ERROR=1``; see
# ``include/dqlite.h``) is not silently swallowed when an upstream
# ``failure(req, DQLITE_ERROR, ...)`` reply arrives — the dqlite
# emission's wording does not match these substrings.
# Canonical prefix for wire-decode-failure exception text. The
# disconnect-classification chain (wire → client.ProtocolError →
# dbapi.OperationalError(code=None) → SA's _dqlite_disconnect_messages
# substring scan) depends on this exact lowercase phrase appearing in
# the rendered exception message. Five sites cooperate via the
# substring; defining it here keeps them in lockstep.
WIRE_DECODE_FAILED_PREFIX: Final[str] = "wire decode failed"

NO_TRANSACTION_MESSAGE_SUBSTRINGS: Final[tuple[str, ...]] = (
    # Anchored to the canonical SQLite phrasing — the bare token
    # "cannot rollback" was previously also matched but is too
    # permissive: any unrelated SQLite (or future DQLITE_ERROR=1)
    # error message that happens to contain that bare phrase would
    # trigger the silent-swallow path. Both upstream wordings
    # ("cannot rollback ..." / "cannot commit ...") contain
    # "no transaction is active" so a single anchored substring
    # covers both legitimate cases. Pinned by
    # ``test_no_transaction_error_wording.py`` against a live
    # cluster.
    "no transaction is active",
)

TX_AUTO_ROLLBACK_PRIMARY_CODES: Final[frozenset[int]] = frozenset(
    {SQLITE_ABORT, SQLITE_NOMEM, SQLITE_INTERRUPT, SQLITE_IOERR, SQLITE_CORRUPT, SQLITE_FULL}
)
# See also ``sqlalchemy-dqlite/src/sqlalchemydqlite/base.py``'s
# ``_BARE_DBE_DISCONNECT_CODES`` which classifies CORRUPT/FORMAT/
# NOTADB as slot-fatal at the SA pool layer. The two sets serve
# different purposes — wire-side auto-rollback bookkeeping clear vs
# pool-side connection eviction — and overlap on SQLITE_CORRUPT only.
# A maintainer adjusting either set should re-evaluate the other:
# they are independently correct today but the overlap is not
# accidental.


# Cumulative-row cap for a single SELECT result spanning multiple
# continuation frames. The default protects against unbounded memory
# growth on a maliciously slow-drip server while staying well above
# any realistic legitimate result set. Forwarded to every
# :class:`DqliteConnection` the public ``connect()`` /
# ``ConnectionPool`` / dbapi ``connect()`` entry points hand out.
DEFAULT_MAX_TOTAL_ROWS: Final[int] = 10_000_000

# Per-query continuation-frame cap. Complements
# ``DEFAULT_MAX_TOTAL_ROWS``: a server sending one row per frame can
# inflict O(n) Python decode work where n is the row cap; the frame
# cap bounds that work even when the row cap is large.
DEFAULT_MAX_CONTINUATION_FRAMES: Final[int] = 100_000
