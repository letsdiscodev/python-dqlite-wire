"""Protocol constants for dqlite wire protocol."""

from enum import IntEnum

# Protocol versions
PROTOCOL_VERSION = 1
PROTOCOL_VERSION_LEGACY = 0x86104DD760433FE5  # Pre-1.0 dqlite servers

# Word size in bytes (all messages are padded to 8-byte boundaries)
WORD_SIZE = 8

# Header size in bytes
HEADER_SIZE = 8

# Row markers — written as full uint64 words on the wire, but detected by
# checking only the first byte (0xFF = done, 0xEE = part). This matches Go's
# byte-by-byte detection in columnTypes(). Use ROW_DONE_BYTE/ROW_PART_BYTE
# for detection logic; use the full markers for encoding.
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
    # Server-to-client ONLY: the C server's tuple_decoder has no inbound
    # case for DQLITE_UNIXTIME, so this tag must never appear on outgoing
    # parameters. encode_params_tuple enforces this defensively.
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
