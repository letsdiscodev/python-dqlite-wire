"""Shared ``raw_message``-cap helper (SSOT for the client and dbapi packages):
truncate to a codepoint budget and append a deterministic suffix.
"""

from typing import Final

__all__ = [
    "DEFAULT_MAX_RAW_MESSAGE",
    "cap_raw_message",
]


# 4 KiB: above any realistic SQLite error string, but bounds worst-case
# exception payloads fanned out via BaseExceptionGroup chains + pickling.
DEFAULT_MAX_RAW_MESSAGE: Final[int] = 4 * 1024

_TRUNCATION_SUFFIX_TEMPLATE: Final[str] = "... [raw_message truncated, {overflow} codepoints]"
# Per-call-site suffix wording ("codepoints" vs "chars", "aggregate", etc.)
# diverges across the client/dbapi/sqlalchemy packages by design; "chars"
# and "codepoints" measure the same quantity on CPython.


def cap_raw_message(raw_message: str | None, max_chars: int) -> str | None:
    """Truncate ``raw_message`` to ``max_chars`` codepoints and append a suffix
    carrying the dropped count; ``None`` and within-cap inputs pass through.

    Raises ``ValueError`` on negative ``max_chars`` (0 is a valid cap).

    SECURITY: this is a length cap only, NOT a sanitiser — server-supplied
    LF/Tab/control chars survive. Log sites MUST apply
    :func:`dqlitewire.sanitize_for_log` to avoid log-line splitting (CWE-117).
    """
    if max_chars < 0:
        raise ValueError(f"max_chars must be >= 0 (codepoint budget); got {max_chars}")
    if raw_message is None or len(raw_message) <= max_chars:
        return raw_message
    overflow = len(raw_message) - max_chars
    return raw_message[:max_chars] + _TRUNCATION_SUFFIX_TEMPLATE.format(overflow=overflow)
