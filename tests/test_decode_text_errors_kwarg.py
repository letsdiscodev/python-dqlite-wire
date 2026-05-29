"""``decode_text`` accepts an ``errors=`` kwarg threaded to ``bytes.decode``.
The ``"strict"`` default is a Python-side narrowing (C and Go clients pass
non-UTF-8 bytes through unvalidated).
"""

import pytest

from dqlitewire.exceptions import DecodeError
from dqlitewire.types import decode_text


def test_strict_default_rejects_invalid_utf8() -> None:
    # 0xff is invalid UTF-8.
    data = b"\xff" + b"\x00" + b"\x00" * 6
    with pytest.raises(DecodeError, match="Invalid UTF-8"):
        decode_text(data)


def test_replace_kwarg_coerces_invalid_utf8() -> None:
    data = b"\xff" + b"\x00" + b"\x00" * 6
    text, _ = decode_text(data, errors="replace")
    assert text == "�"


def test_backslashreplace_kwarg_coerces_invalid_utf8() -> None:
    data = b"\xff" + b"\x00" + b"\x00" * 6
    text, _ = decode_text(data, errors="backslashreplace")
    assert text == "\\xff"


def test_errors_kwarg_works_through_memoryview_branch() -> None:
    """The chunked / one-shot memoryview path also honours ``errors``."""
    data = memoryview(b"\xff" + b"\x00" + b"\x00" * 6)
    text, _ = decode_text(data, errors="replace")
    assert text == "�"


def test_strict_explicit_is_default_behaviour() -> None:
    """Passing ``errors='strict'`` explicitly matches the default."""
    data = b"\xff" + b"\x00" + b"\x00" * 6
    with pytest.raises(DecodeError, match="Invalid UTF-8"):
        decode_text(data, errors="strict")
