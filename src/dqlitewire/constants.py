"""Protocol constants for dqlite wire protocol."""

from enum import IntEnum

# Protocol version
PROTOCOL_VERSION = 1

# Word size in bytes (all messages are padded to 8-byte boundaries)
WORD_SIZE = 8

# Header size in bytes
HEADER_SIZE = 8

# Row markers
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
    CONNECT = 11
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
    LEADER = 1  # Also called NODE
    WELCOME = 2
    NODE_LEGACY = 3
    DB = 4
    STMT = 5
    RESULT = 6
    ROWS = 7
    EMPTY = 8
    FILES = 9
    SERVERS = 10  # Also called CLUSTER or NODES
    METADATA = 11
    DESCRIPTION = 12


class ValueType(IntEnum):
    """Value types for parameters and row values."""

    INTEGER = 1
    FLOAT = 2
    TEXT = 3
    BLOB = 4
    NULL = 5
    UNIXTIME = 9  # Unix time (deprecated, maps to INTEGER)
    ISO8601 = 10  # ISO8601 string (maps to TEXT)
    BOOLEAN = 11  # Boolean (maps to INTEGER)
