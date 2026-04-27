"""Protocol constants for dqlite wire protocol."""

from enum import IntEnum

# Protocol versions
PROTOCOL_VERSION = 1
PROTOCOL_VERSION_LEGACY = 0x86104DD760433FE5  # Pre-1.0 dqlite servers

# Word size in bytes (all messages are padded to 8-byte boundaries)
WORD_SIZE = 8

# Header size in bytes
HEADER_SIZE = 8

# Row markers — written as full uint64 words on the wire. Detection
# validates all 8 bytes of the sentinel (via ``_ROW_DONE_MARKER`` /
# ``_ROW_PART_MARKER`` bytes in ``tuples.py``); the original first-byte
# check was tightened in the ``row-marker-full-word`` fix to close a
# misclassification window where a legitimate value starting with 0xFF
# or 0xEE could be confused with a marker. Use the uint64 constants
# below for encoding and the single-byte constants to build the detection
# byte sequences.
ROW_DONE_BYTE = 0xFF
ROW_PART_BYTE = 0xEE
ROW_DONE_MARKER = 0xFFFFFFFFFFFFFFFF
ROW_PART_MARKER = 0xEEEEEEEEEEEEEEEE


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
SQLITE_ERROR = 1  # generic SQL error (e.g. nonempty statement tail)
SQLITE_BUSY = 5  # busy retry-or-fail — engine-side OR Raft-side
SQLITE_NOTFOUND = 12  # gateway.c LOOKUP_DB / LOOKUP_STMT
SQLITE_PROTOCOL = 15  # gateway.c "bad format version" / wire mismatch

# SQLite extended error codes that signal leader changes in a dqlite
# cluster. Upstream definitions in ``dqlite-upstream/include/dqlite.h``:
#
#     #define SQLITE_IOERR_NOT_LEADER       (SQLITE_IOERR | (40 << 8))
#     #define SQLITE_IOERR_LEADERSHIP_LOST  (SQLITE_IOERR | (41 << 8))
#
# where ``SQLITE_IOERR = 10``. Callers (``dqliteclient`` and
# ``sqlalchemy-dqlite``) import these to decide whether to invalidate a
# connection and retry against a fresh leader.
SQLITE_IOERR = 10
SQLITE_IOERR_NOT_LEADER = SQLITE_IOERR | (40 << 8)  # 10250
SQLITE_IOERR_LEADERSHIP_LOST = SQLITE_IOERR | (41 << 8)  # 10506
LEADER_ERROR_CODES: frozenset[int] = frozenset(
    {SQLITE_IOERR_NOT_LEADER, SQLITE_IOERR_LEADERSHIP_LOST}
)

# dqlite-namespace error codes (>= 1000). These do NOT belong to the
# SQLite primary-code namespace and ``primary_sqlite_code`` returns
# them unchanged so a downstream ``code == primary_sqlite_code(...)``
# check does not collide with a real SQLite primary. Source:
# ``dqlite-upstream/src/protocol.h`` (DQLITE_PROTO),
# ``dqlite-upstream/src/lib/registry.h`` (DQLITE_NOTFOUND), and
# ``dqlite-upstream/src/lib/serialize.h`` (DQLITE_PARSE).
DQLITE_PROTO = 1001  # protocol.h:9 — Raft FSM-internal protocol error.
# Currently emitted only inside command.c / fsm.c
# apply paths and never reaches gateway.c::failure(),
# but included here for namespace completeness so a
# future change that surfaces it through the gateway
# passes through ``primary_sqlite_code`` cleanly.
DQLITE_NOTFOUND = 1002  # registry lookup miss (server-side scratch)
DQLITE_PARSE = 1005  # gateway.c unrecognized request type / schema

# Bit mask for extracting the primary (low-byte) code from an
# extended SQLite error code. Upstream encodes extended codes as
# ``primary | (sub << 8)``; ``code & SQLITE_PRIMARY_CODE_MASK``
# recovers the primary. Use via :func:`primary_sqlite_code`.
SQLITE_PRIMARY_CODE_MASK = 0xFF


_DQLITE_NAMESPACE_CODES: frozenset[int] = frozenset({DQLITE_PROTO, DQLITE_NOTFOUND, DQLITE_PARSE})


def is_dqlite_namespace_code(code: int) -> bool:
    """True if ``code`` is a dqlite-namespace error code.

    dqlite uses non-SQLite codes for server-internal failure shapes
    (``DQLITE_PARSE = 1005``, ``DQLITE_NOTFOUND = 1002``). The set is
    enumerated here exactly — extended SQLite codes
    (``SQLITE_IOERR_NOT_LEADER = 10250`` etc.) happen to fall in
    ``code >= 1000`` too, but they decode via ``code & 0xFF`` to a
    valid primary while dqlite-namespace codes mask to nonsense
    bytes. Callers dispatching on these failure shapes should use
    this helper to special-case them, rather than masking.
    """
    return code in _DQLITE_NAMESPACE_CODES


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
SQLITE_ABORT = 4  # operation aborted (e.g., sqlite3_interrupt) — defensive
SQLITE_NOMEM = 7  # out of memory
SQLITE_INTERRUPT = 9  # query interrupted via INTERRUPT
SQLITE_CORRUPT = 11  # database disk image malformed — defensive
SQLITE_FULL = 13  # database/disk full
# Auxiliary database-file format errors. Both surface from the engine
# when the file on disk cannot be read as a SQLite database — schema
# version mismatch, header magic mismatch, or a non-database file
# opened by mistake. The SA dialect treats codes 11/24/26 as
# slot-fatal during pre-ping (``do_ping``) and as disconnect-class
# during failure dispatch (``is_disconnect``); keeping the constants
# in the wire layer alongside the other SQLite primaries lets both
# the dialect and the dbapi import them by name instead of inlining
# magic literals.
SQLITE_FORMAT = 24
SQLITE_NOTADB = 26

TX_AUTO_ROLLBACK_PRIMARY_CODES: frozenset[int] = frozenset(
    {SQLITE_ABORT, SQLITE_NOMEM, SQLITE_INTERRUPT, SQLITE_IOERR, SQLITE_CORRUPT, SQLITE_FULL}
)


# Cumulative-row cap for a single SELECT result spanning multiple
# continuation frames. The default protects against unbounded memory
# growth on a maliciously slow-drip server while staying well above
# any realistic legitimate result set. Forwarded to every
# :class:`DqliteConnection` the public ``connect()`` /
# ``ConnectionPool`` / dbapi ``connect()`` entry points hand out.
DEFAULT_MAX_TOTAL_ROWS = 10_000_000

# Per-query continuation-frame cap. Complements
# ``DEFAULT_MAX_TOTAL_ROWS``: a server sending one row per frame can
# inflict O(n) Python decode work where n is the row cap; the frame
# cap bounds that work even when the row cap is large.
DEFAULT_MAX_CONTINUATION_FRAMES = 100_000
