"""Pin: ``decode_text``'s docstring warns that ``errors="replace"``
and ``errors="surrogateescape"`` produce codepoints (``U+FFFD`` and
``U+DC80``-``U+DCFF``) that bypass ``sanitize_for_log`` /
``sanitize_server_text``.

A caller opting into permissive decoding for legacy-data tolerance
must not assume the log-side scrub will catch the residual
codepoints. The docstring is the contract; pin the warning so a
future docstring rewrite cannot silently drop it.
"""

from __future__ import annotations

from dqlitewire.types import decode_text


def test_decode_text_docstring_warns_about_log_sanitiser_bypass() -> None:
    doc = decode_text.__doc__ or ""
    assert "U+FFFD" in doc, "docstring must name the replacement-character artefact"
    assert "U+DC80" in doc, "docstring must name the surrogateescape range"
    assert "sanitize_for_log" in doc, "docstring must reference the log scrub"
