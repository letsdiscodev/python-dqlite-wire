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

# Bit mask for extracting the primary (low-byte) code from an
# extended SQLite error code. Upstream encodes extended codes as
# ``primary | (sub << 8)``; ``code & SQLITE_PRIMARY_CODE_MASK``
# recovers the primary. Use via :func:`primary_sqlite_code`.
SQLITE_PRIMARY_CODE_MASK = 0xFF


def primary_sqlite_code(code: int) -> int:
    """Return the primary SQLite error code from an extended code.

    ``SQLITE_IOERR_NOT_LEADER`` (``10250``) → ``SQLITE_IOERR`` (``10``).
    A primary code (low byte only) passes through unchanged.

    Matches the ``code & 0xFF`` pattern used by dqlitedbapi for its
    ``Error`` subclass dispatch and by the no-transaction mask.
    Exposed so callers don't have to recompute the shift inline and
    so a future change to the extended-code layout has a single
    point of rework.
    """
    return code & SQLITE_PRIMARY_CODE_MASK
