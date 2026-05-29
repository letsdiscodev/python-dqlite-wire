"""Protocol constants for dqlite wire protocol."""

from enum import IntEnum
from typing import Final

__all__ = [
    "BARE_DATABASE_ERROR_CODES",
    "DEFAULT_MAX_CONTINUATION_FRAMES",
    "DEFAULT_MAX_TOTAL_ROWS",
    "DQLITE_NOTFOUND",
    "DQLITE_PARSE",
    "DQLITE_PROTO",
    "HEADER_SIZE",
    "LEADER_ERROR_CODES",
    "LEADER_LOST_DB_LOOKUP_SUBSTRING",
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

PROTOCOL_VERSION: Final[int] = 1
PROTOCOL_VERSION_LEGACY: Final[int] = 0x86104DD760433FE5  # Pre-1.0 dqlite servers

WORD_SIZE: Final[int] = 8  # all messages padded to 8-byte boundaries
HEADER_SIZE: Final[int] = 8

# Row markers are full uint64 words on the wire; detection validates all
# 8 bytes to avoid misclassifying a value starting 0xFF/0xEE as a marker.
# The single-byte constants are the source of truth: the byte form in
# tuples.py and the uint64 form below are both derived from them.
ROW_DONE_BYTE: Final[int] = 0xFF
ROW_PART_BYTE: Final[int] = 0xEE
ROW_DONE_MARKER: Final[int] = int.from_bytes(bytes([ROW_DONE_BYTE]) * 8, "little")
ROW_PART_MARKER: Final[int] = int.from_bytes(bytes([ROW_PART_BYTE]) * 8, "little")
# Derivation-correctness invariants (not runtime guards); strippable under -O.
if __debug__:
    assert ROW_DONE_MARKER == 0xFFFFFFFFFFFFFFFF
    assert ROW_PART_MARKER == 0xEEEEEEEEEEEEEEEE


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
    # Deprecated, maps to INTEGER. Server-to-client ONLY: the C client
    # rejects inbound UNIXTIME with DQLITE_PARSE, so mock servers must not
    # send it to real clients; encode_params_tuple also rejects it outbound.
    # Wire is integer seconds — subsecond precision is lost; use ISO8601.
    UNIXTIME = 9
    ISO8601 = 10  # ISO8601 string (maps to TEXT)
    BOOLEAN = 11  # Boolean (maps to INTEGER)


class NodeRole(IntEnum):
    """Matches Go's Voter/StandBy/Spare and C's DQLITE_VOTER/STANDBY/SPARE."""

    VOTER = 0
    STANDBY = 1
    SPARE = 2


# SQLite primary error codes upstream's gateway emits via failure().
SQLITE_ERROR: Final[int] = 1  # generic SQL error (e.g. nonempty statement tail)
SQLITE_BUSY: Final[int] = 5  # busy retry-or-fail — engine-side OR Raft-side
SQLITE_NOTFOUND: Final[int] = 12  # gateway.c LOOKUP_DB / LOOKUP_STMT
SQLITE_PROTOCOL: Final[int] = 15  # gateway.c "bad format version" / wire mismatch

# Extended codes signalling leader changes; callers invalidate the
# connection and retry against a fresh leader.
SQLITE_IOERR: Final[int] = 10
SQLITE_IOERR_NOT_LEADER: Final[int] = SQLITE_IOERR | (40 << 8)  # 10250
SQLITE_IOERR_LEADERSHIP_LOST: Final[int] = SQLITE_IOERR | (41 << 8)  # 10506

# Legacy leader-change sub-codes from pre-3.32.1 dqlite servers, kept so a
# mixed cluster classifies leader-flips uniformly (Go carries both modern
# 40/41 and legacy 32/33 through ErrBadConn).
# WARNING: these values collide with stdlib sqlite3 codes assigned later —
# 8202 == sqlite3.SQLITE_IOERR_DATA, 8458 == sqlite3.SQLITE_IOERR_CORRUPTFS.
# Do NOT re-export the stdlib SQLITE_IOERR_* family or alias these values.
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
# Go maps errNotFound (SQLITE_NOTFOUND = 12) to ErrBadConn "potentially
# after leadership loss", but 12 is NOT in LEADER_ERROR_CODES: it is
# overloaded (LOOKUP_STMT also emits 12 for an unknown statement id, a
# server-side bug). Dispatch is substring-gated via
# LEADER_LOST_DB_LOOKUP_SUBSTRING instead.

# dqlite-namespace error codes (>= 1000), outside the SQLite primary-code
# namespace; primary_sqlite_code returns them unchanged to avoid colliding
# with a real SQLite primary.
DQLITE_PROTO: Final[int] = 1001  # Raft FSM-internal; never reaches the wire today
DQLITE_NOTFOUND: Final[int] = 1002  # registry lookup miss; emitted by handle_dump
DQLITE_PARSE: Final[int] = 1005  # gateway.c::INIT_V0 — unrecognized request type/schema
# Note: the five params-schema-aware handlers (prepare/exec/query/exec_sql/
# query_sql) bypass INIT_V0 and emit SQLITE_ERROR (=1), not 1005, on an
# unrecognized schema; tests for those must pin code=1.

# Recover the primary (low-byte) code from an extended SQLite code
# (encoded as ``primary | (sub << 8)``). Use via primary_sqlite_code.
SQLITE_PRIMARY_CODE_MASK: Final[int] = 0xFF


# Range chosen so extended SQLite codes (primary | (sub << 8)) never land
# here — valid primaries top out at ~28 — and any future namespace code is
# auto-included without a manual sync.
_DQLITE_NAMESPACE_MIN: Final[int] = 1000
_DQLITE_NAMESPACE_MAX_EXCLUSIVE: Final[int] = 1024


def is_dqlite_namespace_code(code: int) -> bool:
    """True if ``code`` is a dqlite-namespace error code (``[1000, 1024)``)."""
    return _DQLITE_NAMESPACE_MIN <= code < _DQLITE_NAMESPACE_MAX_EXCLUSIVE


def primary_sqlite_code(code: int) -> int:
    """Return the primary SQLite code from an extended code; e.g. 10250 -> 10.

    dqlite-namespace codes pass through UNCHANGED rather than masked: masking
    1005 with 0xFF yields 237, which a future SQLite primary 237 would collide
    with. Use is_dqlite_namespace_code to dispatch on those deliberately.
    """
    if is_dqlite_namespace_code(code):
        return code
    return code & SQLITE_PRIMARY_CODE_MASK


# Primary codes implying server-side auto-rollback (see
# TX_AUTO_ROLLBACK_PRIMARY_CODES). NOMEM/IOERR/INTERRUPT/FULL are
# SQLite-documented; ABORT/CORRUPT are defensive over-clears (false
# positive is benign). BUSY is excluded: its two origins (engine vs Raft)
# are indistinguishable without the message text.
SQLITE_ABORT: Final[int] = 4  # operation aborted (e.g., sqlite3_interrupt) — defensive
SQLITE_NOMEM: Final[int] = 7  # out of memory
SQLITE_INTERRUPT: Final[int] = 9  # query interrupted via INTERRUPT
SQLITE_CORRUPT: Final[int] = 11  # database disk image malformed — defensive
SQLITE_FULL: Final[int] = 13  # database/disk full
# File-format errors (file on disk not a readable SQLite database). The SA
# dialect treats 11/24/26 as slot-fatal at pre-ping and disconnect dispatch.
SQLITE_FORMAT: Final[int] = 24
SQLITE_NOTADB: Final[int] = 26


# Codes dqlite emits as bare DatabaseError (not a PEP 249 subclass); the
# slot-fatal disconnect set under SA. Single source of truth: SA's
# _BARE_DBE_DISCONNECT_CODES and dbapi's _CODE_TO_EXCEPTION both derive
# from this, staying aligned if the set grows.
BARE_DATABASE_ERROR_CODES: Final[frozenset[int]] = frozenset(
    {SQLITE_CORRUPT, SQLITE_FORMAT, SQLITE_NOTADB}
)


# Additional primary codes the dbapi PEP 249 classifier dispatches on.
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


# Extended SQLITE_CONSTRAINT_* family (high byte 1..11); values match stdlib.
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


# Substrings marking a benign "no transaction active" SQLITE_ERROR (code 1)
# reply, recognised by both dbapi and client layers (single source of truth,
# pinned by test_no_transaction_error_wording.py against a live cluster).
# Anchored to "no transaction is active" rather than the over-permissive bare
# "cannot rollback" so an unrelated code-1 message is not silently swallowed.
NO_TRANSACTION_MESSAGE_SUBSTRINGS: Final[tuple[str, ...]] = ("no transaction is active",)

# Exact lowercase phrase the disconnect-classification chain (wire ->
# client.ProtocolError -> dbapi.OperationalError -> SA substring scan)
# depends on appearing in the rendered exception message.
WIRE_DECODE_FAILED_PREFIX: Final[str] = "wire decode failed"

# Server message (paired with SQLITE_NOTFOUND=12) marking the post-leader-loss
# connection-dead arm. Substring-gating distinguishes it from LOOKUP_STMT,
# which emits the same code for an unknown statement id (a server-side bug).
# Consumers: dqliteclient.connection._run_protocol and SA's is_disconnect.
LEADER_LOST_DB_LOOKUP_SUBSTRING: Final[str] = "no database opened"

TX_AUTO_ROLLBACK_PRIMARY_CODES: Final[frozenset[int]] = frozenset(
    {SQLITE_ABORT, SQLITE_NOMEM, SQLITE_INTERRUPT, SQLITE_IOERR, SQLITE_CORRUPT, SQLITE_FULL}
)
# Distinct from SA's _BARE_DBE_DISCONNECT_CODES (pool-side eviction); the two
# overlap on SQLITE_CORRUPT only — adjust both together.


# Cumulative-row cap for one SELECT across continuation frames; guards against
# a slow-drip server exhausting memory.
DEFAULT_MAX_TOTAL_ROWS: Final[int] = 10_000_000

# Per-query frame cap: bounds O(n) decode work when a server sends one row
# per frame, even with a large row cap.
DEFAULT_MAX_CONTINUATION_FRAMES: Final[int] = 100_000
