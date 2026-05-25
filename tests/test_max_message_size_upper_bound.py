"""Pin: ``max_message_size`` is bounded above by
``ReadBuffer.MAX_MESSAGE_SIZE_CEILING`` (``UINT32_MAX`` bytes,
mirroring the C server's per-frame ceiling at
``dqlite-upstream/src/conn.c:169``).

A caller that constructs ``ReadBuffer(max_message_size=2**40)`` (or
``MessageEncoder`` / ``MessageDecoder``) would otherwise silently
disable the composite-frame protection the envelope is supposed to
provide. The C server hard-caps inbound frames at ``UINT32_MAX``
bytes; a Python ceiling at the same bound can never refuse a frame
the C server would accept.
"""

from __future__ import annotations

import pytest

from dqlitewire.buffer import ReadBuffer
from dqlitewire.codec import MessageDecoder, MessageEncoder


def test_read_buffer_rejects_oversize_max_message_size() -> None:
    with pytest.raises(ValueError, match="UINT32_MAX"):
        ReadBuffer(max_message_size=2**40)


def test_message_encoder_rejects_oversize_max_message_size() -> None:
    with pytest.raises(ValueError, match="UINT32_MAX"):
        MessageEncoder(max_message_size=2**40)


def test_message_decoder_rejects_oversize_max_message_size() -> None:
    with pytest.raises(ValueError, match="UINT32_MAX"):
        MessageDecoder(max_message_size=2**40)


def test_ceiling_itself_is_admissible() -> None:
    """At ``UINT32_MAX`` exactly the constructor still works — the
    upper-bound check is non-strict (``>``, not ``>=``)."""
    buf = ReadBuffer(max_message_size=ReadBuffer.MAX_MESSAGE_SIZE_CEILING)
    assert buf._max_message_size == ReadBuffer.MAX_MESSAGE_SIZE_CEILING


def test_default_and_smaller_values_still_accepted() -> None:
    ReadBuffer(max_message_size=1)
    ReadBuffer(max_message_size=ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE)
    MessageEncoder(max_message_size=ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE)
    MessageDecoder(max_message_size=ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE)
