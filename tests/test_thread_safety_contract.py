"""Regression tests for single-owner contract documentation.

Earlier documentation of the thread-safety analysis plus the poison
mechanism created a misleading impression that concurrent misuse
would always surface as ``ProtocolError``. Empirical fuzz testing
confirmed it does not — under threaded contention, ``ReadBuffer``
produces duplicate messages and corrupt (garbage) bytes with
``is_poisoned`` remaining ``False``, because torn reads often yield
valid-looking slices that decode cleanly.

The remediation is documentation, not code: the docstrings on
``ReadBuffer``, ``WriteBuffer``, ``MessageDecoder``, and
``MessageEncoder`` must make the single-owner contract explicit AND
call out that the poison mechanism cannot and does not detect
concurrent misuse. These tests assert the docstrings carry that
warning so future refactors that reformat docstrings don't silently
drop it.
"""

from __future__ import annotations

from dqlitewire.buffer import ReadBuffer, WriteBuffer
from dqlitewire.codec import MessageDecoder, MessageEncoder


def _docstring(obj: object) -> str:
    return (obj.__doc__ or "").lower()


class TestThreadSafetyContractDocumented:
    """Assert the thread-safety contract is explicit in class docstrings.

    The contract has three parts, and all three must be mentioned:

    1. The class is NOT thread-safe / requires a single owner.
    2. Concurrent misuse produces SILENT data corruption, not
       exceptions.
    3. The poison mechanism does NOT detect concurrent misuse.

    Any docstring-reformat that drops one of these silently
    undoes the documentation fix.
    """

    def test_readbuffer_docstring_warns_about_thread_safety(self) -> None:
        doc = _docstring(ReadBuffer)
        assert "not thread-safe" in doc or "not thread safe" in doc, (
            "ReadBuffer docstring must explicitly say it is not thread-safe"
        )
        assert "silent" in doc and ("corruption" in doc or "corrupt" in doc), (
            "ReadBuffer docstring must warn that concurrent misuse produces silent corruption"
        )
        assert "poison" in doc, (
            "ReadBuffer docstring must explain the poison mechanism's "
            "relationship to concurrent misuse"
        )

    def test_writebuffer_docstring_warns_about_thread_safety(self) -> None:
        doc = _docstring(WriteBuffer)
        assert "not thread-safe" in doc or "not thread safe" in doc, (
            "WriteBuffer docstring must explicitly say it is not thread-safe"
        )

    def test_message_decoder_docstring_warns_about_thread_safety(self) -> None:
        doc = _docstring(MessageDecoder)
        assert "not thread-safe" in doc or "not thread safe" in doc, (
            "MessageDecoder docstring must explicitly say it is not thread-safe"
        )
        assert "silent" in doc and ("corruption" in doc or "corrupt" in doc), (
            "MessageDecoder docstring must warn that concurrent misuse produces silent corruption"
        )
        assert "poison" in doc, (
            "MessageDecoder docstring must explain the poison mechanism's "
            "relationship to concurrent misuse"
        )

    def test_message_encoder_docstring_warns_about_thread_safety(self) -> None:
        doc = _docstring(MessageEncoder)
        assert "not thread-safe" in doc or "not thread safe" in doc, (
            "MessageEncoder docstring must explicitly say it is not thread-safe"
        )
