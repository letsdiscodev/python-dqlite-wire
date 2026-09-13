"""Size caps applied on top of the wire format (the format itself leaves these fields
uncapped). The 64 MiB figures match the default frame envelope minus framing overhead."""

from typing import Final

__all__ = [
    "DEFAULT_MAX_ROWS",
    "MAX_ADDRESS_SIZE",
    "MAX_BLOB_SIZE",
    "MAX_COLUMN_COUNT",
    "MAX_COLUMN_NAME_SIZE",
    "MAX_DUMP_FILENAME_SIZE",
    "MAX_FAILURE_MESSAGE_SIZE",
    "MAX_FILENAME_SIZE",
    "MAX_FILE_CONTENT_SIZE",
    "MAX_FILE_COUNT",
    "MAX_NODE_COUNT",
    "MAX_PARAM_COUNT",
    "MAX_ROWS_AFFECTED",
    "MAX_TAIL_OFFSET",
    "MAX_TEXT_VALUE_SIZE",
]

_FRAME_PAYLOAD: Final[int] = 64 * 1024 * 1024 - 64

MAX_BLOB_SIZE: Final[int] = _FRAME_PAYLOAD
MAX_TEXT_VALUE_SIZE: Final[int] = _FRAME_PAYLOAD
MAX_FILE_CONTENT_SIZE: Final[int] = _FRAME_PAYLOAD
# Byte offset into prepared SQL (StmtResponse V1); a structural ceiling above the text cap.
MAX_TAIL_OFFSET: Final[int] = 64 * 1024 * 1024

# SQLite's SQLITE_MAX_VARIABLE_NUMBER default; upstream rejects more parameters anyway.
MAX_PARAM_COUNT: Final[int] = 32_766
# SQLite's SQLITE_MAX_COLUMN default.
MAX_COLUMN_COUNT: Final[int] = 2000
MAX_FILE_COUNT: Final[int] = 100
MAX_NODE_COUNT: Final[int] = 10_000
# sqlite3_changes() returns a C int.
MAX_ROWS_AFFECTED: Final[int] = (1 << 31) - 1
DEFAULT_MAX_ROWS: Final[int] = 1_000_000

MAX_FAILURE_MESSAGE_SIZE: Final[int] = 64 * 1024
MAX_COLUMN_NAME_SIZE: Final[int] = 4096
MAX_FILENAME_SIZE: Final[int] = 4096
# The C gateway's handle_dump builds "<name>-wal" in a 1024-byte buffer.
MAX_DUMP_FILENAME_SIZE: Final[int] = 1019
# RFC 1035 caps a domain name at 253 bytes; 256 leaves room for the port.
MAX_ADDRESS_SIZE: Final[int] = 256
