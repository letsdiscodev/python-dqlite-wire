"""``_sanitize_for_log`` escapes LF and CR so log records carrying
server-supplied text are immune to line-injection: a hostile
``FailureResponse.message`` containing ``\\n`` would otherwise
split the log line in syslog / journald, and the operator's
structured pipeline would treat the second half as if it came from
the dqlite client.

The exception-message rendering keeps raw LF for multi-line
interactive debugging — only the log-render layer escapes.
"""

from dqlitewire.messages.responses import _sanitize_for_log, _sanitize_server_text


def test_lf_escaped_in_log_output() -> None:
    assert _sanitize_for_log("line1\nline2") == "line1\\nline2"


def test_cr_replaced_by_underlying_sanitiser() -> None:
    """CR is in the C0 control-chars regex of
    _sanitize_server_text, so it's replaced with '?' before reaching
    the LF-escape step."""
    assert _sanitize_for_log("a\rb") == "a?b"


def test_crlf_escaped() -> None:
    """CR replaced first, then LF escaped."""
    assert _sanitize_for_log("a\r\nb") == "a?\\nb"


def test_tab_escaped_in_log_output() -> None:
    """Tabs are escaped to ``\\t`` for the same defense-in-depth
    reason LF is escaped to ``\\n``: TSV-style log parsers and
    journald ``MESSAGE=`` ad-hoc consumers treat tabs as field
    separators, and a server-supplied tab can break those pipelines
    the way an un-escaped LF would. ``_sanitize_server_text`` keeps
    tabs intact for general server-text rendering; only this log-
    line-oriented helper escapes them."""
    assert _sanitize_for_log("a\tb") == "a\\tb"


def test_control_chars_replaced_via_sanitize_server_text() -> None:
    """C0 control char passes through the wrapped helper too."""
    assert _sanitize_for_log("a\x00b") == _sanitize_server_text("a\x00b")
    assert "\x00" not in _sanitize_for_log("a\x00b")


def test_log_injection_attempt_neutralised() -> None:
    """A hostile server message attempting to fake a log line."""
    hostile = "real msg\nFAKE: forged log entry"
    sanitised = _sanitize_for_log(hostile)
    assert "\n" not in sanitised
    assert "FAKE: forged log entry" in sanitised  # still visible
    assert sanitised == "real msg\\nFAKE: forged log entry"
