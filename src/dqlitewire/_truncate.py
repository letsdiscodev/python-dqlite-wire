"""Shared text-truncation helper for cross-package single-source-of-truth.

The dqlite client and dbapi packages both need a `raw_message`-cap
helper that truncates to a fixed codepoint budget and appends a
deterministic suffix. Hosting it in the wire layer (the lowest common
ancestor) eliminates the literal duplication that previously lived
in `dqliteclient.exceptions._cap_raw_message` and
`dqlitedbapi.exceptions._cap_raw_message`.

The helper is intentionally **private** (underscore-prefixed module
+ underscore-prefixed name). Both consumers import via the underscore
path. Promote to public only if a downstream consumer surfaces — the
sibling pattern established by the prior round's `sanitize_for_log`
promotion.

The constants and suffix wording are fixed: the client side maintains
`DqliteError._MAX_RAW_MESSAGE` as a class-level cap that subclasses
may override; the dbapi side uses the module-level cap. Both pass
their cap into this helper, which has no opinion on the value beyond
"non-negative integer".
"""

from __future__ import annotations

from typing import Final

__all__ = ["_cap_raw_message"]


_TRUNCATION_SUFFIX_TEMPLATE: Final[str] = "... [raw_message truncated, {overflow} codepoints]"


def _cap_raw_message(raw_message: str | None, max_chars: int) -> str | None:
    """Truncate ``raw_message`` to ``max_chars`` codepoints (inclusive) and
    append a deterministic suffix carrying the dropped count.

    Returns ``None`` unchanged. Returns the input unchanged if it fits
    within the cap. Caller passes their canonical cap; this helper has
    no default — the absence of a default forces every caller to spell
    out its budget at the call site.

    Raises ``ValueError`` if ``max_chars`` is negative. ``max_chars=0`` is
    a valid (degenerate) cap: every non-empty input is truncated to the
    empty string with the overflow suffix appended. A negative value
    has no meaningful semantic — Python's negative-index slicing would
    silently drop one character while the suffix's overflow count
    reported the difference between length and (negative) cap, producing
    inconsistent diagnostics for what is unambiguously a contributor
    typo.
    """
    if max_chars < 0:
        raise ValueError(f"max_chars must be >= 0 (codepoint budget); got {max_chars}")
    if raw_message is None or len(raw_message) <= max_chars:
        return raw_message
    overflow = len(raw_message) - max_chars
    return raw_message[:max_chars] + _TRUNCATION_SUFFIX_TEMPLATE.format(overflow=overflow)
