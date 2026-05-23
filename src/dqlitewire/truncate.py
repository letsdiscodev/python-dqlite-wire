"""Shared text-truncation helper for cross-package single-source-of-truth.

The dqlite client and dbapi packages both need a ``raw_message``-cap
helper that truncates to a fixed codepoint budget and appends a
deterministic suffix. Hosting it in the wire layer (the lowest common
ancestor) eliminates the literal duplication that previously lived
in ``dqliteclient.exceptions._cap_raw_message`` and
``dqlitedbapi.exceptions._cap_raw_message``.

Public names: :data:`DEFAULT_MAX_RAW_MESSAGE` and :func:`cap_raw_message`,
re-exported from ``dqlitewire`` top-level. These promotions follow the
same path :func:`sanitize_for_log` took: started private here, surfaced
to two cross-package consumers, then promoted once the cross-package
use stabilised. The underscore-prefixed names (``_DEFAULT_MAX_RAW_MESSAGE``
/ ``_cap_raw_message``) survive as back-compat aliases for one release
cycle so downstream consumers can migrate without lockstep.

The canonical default codepoint cap is :data:`DEFAULT_MAX_RAW_MESSAGE`
(4 KiB). The client side exposes it as a class-level
``DqliteError._MAX_RAW_MESSAGE`` so subclasses may override per
exception class; the dbapi side imports the constant directly and
passes it through verbatim. Both consumers pass their effective cap
into :func:`cap_raw_message`, which has no opinion on the value beyond
"non-negative integer".
"""

from typing import Final

__all__ = [
    "DEFAULT_MAX_RAW_MESSAGE",
    "cap_raw_message",
]


# Canonical raw-message codepoint cap shared by ``dqliteclient`` and
# ``dqlitedbapi``. The wire layer caps a single ``FailureResponse``
# message at ~64 KiB; combined with ``BaseExceptionGroup`` chains in
# leader-discovery and pool-initialise paths plus cross-process
# pickling (``ProcessPoolExecutor``, Celery, structured-error
# capture), an unbounded ``raw_message`` would otherwise produce
# multi-MB exception payloads. 4 KiB is well above any realistic
# SQLite error string while bounding the worst-case fan-out. Hosted
# here so the two downstream packages share a single source of truth
# rather than duplicate the literal.
DEFAULT_MAX_RAW_MESSAGE: Final[int] = 4 * 1024

_TRUNCATION_SUFFIX_TEMPLATE: Final[str] = "... [raw_message truncated, {overflow} codepoints]"
# Cross-package truncation-suffix divergence note:
# Five sites carry truncation logic with three vocabularies. Both
# "codepoints" and "chars" refer to the same quantity on CPython
# (``len(str)`` counts codepoints), so the unit-divergence is purely
# vocabulary, not measurement. The wording-divergence is intentional
# scope drift across packages — keeping per-call-site templates lets
# context-specific wording (``raw_message`` / ``aggregate`` /
# bare-``truncated``) survive consolidation without a SSOT migration.
# The five sites are:
#   - this module (SSOT for FailureResponse-style raw_message capping)
#   - python-dqlite-client/src/dqliteclient/exceptions.py — "[truncated, {overflow} codepoints]"
#   - python-dqlite-client/src/dqliteclient/cluster.py — "[truncated, {overflow} chars]" and
#                                                       "[aggregate truncated, {overflow} chars]"
#   - python-dqlite-dbapi/src/dqlitedbapi/types.py — "[truncated, ... chars]"
#   - sqlalchemy-dqlite/src/sqlalchemydqlite/base.py — "[truncated, {overflow} chars]"
# A future consolidation could promote ``suffix_template`` to a kwarg
# and route the four downstream sites through this helper; until then,
# this comment is the single index of the divergence.


def cap_raw_message(raw_message: str | None, max_chars: int) -> str | None:
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

    SECURITY NOTE — does NOT strip LF / Tab / control characters: this
    helper is a pure codepoint-length cap. Server-supplied text
    containing LF survives into the returned value and from there into
    any ``self.raw_message`` attribute populated from it. The wire-layer
    ``sanitize_server_text`` deliberately preserves LF / Tab for
    human-readable multi-line server diagnostics at exception-display
    time; this cap is the payload-size defence, not a sanitiser.
    Consumers that route ``exc.raw_message`` into ``logger.X`` records
    (directly, or via a third-party hook such as SQLAlchemy's
    ``is_disconnect`` substring scan that may itself log the matched
    text) MUST apply :func:`dqlitewire.sanitize_for_log` at the log
    call site to prevent log-line splitting (CWE-117). A safer-default
    ``_cap_raw_message_for_log`` companion that composes the two could
    be added if a third call site needs it; today every first-party
    log site already sanitises explicitly, so the docs-only warning is
    the minimum-viable defence.
    """
    if max_chars < 0:
        raise ValueError(f"max_chars must be >= 0 (codepoint budget); got {max_chars}")
    if raw_message is None or len(raw_message) <= max_chars:
        return raw_message
    overflow = len(raw_message) - max_chars
    return raw_message[:max_chars] + _TRUNCATION_SUFFIX_TEMPLATE.format(overflow=overflow)
