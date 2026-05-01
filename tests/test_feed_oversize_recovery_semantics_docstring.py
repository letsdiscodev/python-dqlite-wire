"""Pin: ``ReadBuffer.feed()`` docstring scopes the "recoverable via
reset()" claim to the safe case (caller's buffer happens to exceed
the cap with no in-flight over-large message body), and explicitly
flags the unsafe case (server emitted a single message larger than
the cap; rejected bytes are body of that message; reset+continue
produces silent wire desync).

The original wording advertised recoverability unconditionally, which
external callers building on top of ``dqlitewire.ReadBuffer`` would
follow into silent corruption. The in-tree client layer is masked
because every wire DecodeError → wrapped → ``_invalidate`` → connection
drop, but the public-API docstring is the contract for downstream
users.
"""

from __future__ import annotations

from dqlitewire.buffer import ReadBuffer


def test_feed_docstring_does_not_unconditionally_advertise_recoverable() -> None:
    doc = ReadBuffer.feed.__doc__ or ""
    # The original phrase "recoverable via ``reset()``/``clear()``"
    # was the misleading unconditional claim — pin against its
    # literal return.
    assert "recoverable via ``reset()``/``clear()``" not in doc, (
        "feed() docstring must not unconditionally claim recoverability via reset()/clear()"
    )


def test_feed_docstring_documents_unsafe_oversize_case() -> None:
    """The docstring must explicitly call out that an oversize
    message-body rejection requires dropping the connection."""
    doc = ReadBuffer.feed.__doc__ or ""
    assert "drop the connection" in doc.lower(), (
        "feed() docstring must document that the unsafe case requires dropping the connection"
    )


def test_feed_docstring_documents_skip_message_recovery_path() -> None:
    """Cross-reference the read-side oversize path's
    ``skip_message`` recovery affordance — it IS genuinely
    recoverable, distinct from the feed-side rejection."""
    doc = ReadBuffer.feed.__doc__ or ""
    assert "skip_message" in doc
