"""``DumpRequest.name`` is capped at the C gateway's WAL filename ceiling
(1019 = 1024-byte buffer - len("-wal") - NUL). A longer name encodes valid
wire bytes but the C side silently truncates the WAL filename, returning a
``FilesResponse`` whose ``-wal`` entry mismatches the main entry (silent dump
corruption); reject it at the Python wire boundary instead."""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.messages.requests import DumpRequest
from dqlitewire.messages.responses import _MAX_DUMP_FILENAME_SIZE


def test_dump_request_filename_at_c_server_ceiling_accepted() -> None:
    """Exactly the C ceiling (1019 bytes) must encode cleanly."""
    name = "a" * _MAX_DUMP_FILENAME_SIZE
    body = DumpRequest(name).encode_body()
    decoded = DumpRequest.decode_body(body)
    assert decoded.name == name


def test_dump_request_filename_one_past_c_server_ceiling_rejected() -> None:
    """One byte past the ceiling (the WAL truncation boundary) must raise."""
    name = "a" * (_MAX_DUMP_FILENAME_SIZE + 1)
    with pytest.raises(EncodeError):
        DumpRequest(name).encode_body()


def test_dump_request_decoder_rejects_oversize_peer_request() -> None:
    """Decode side must also reject an oversize peer-supplied name, keeping the
    wire-symmetric contract against a misbehaving or pre-fix peer."""
    from dqlitewire.types import encode_text

    # Synthesise a body via the lax 4 KiB encoder cap to bypass DumpRequest's
    # own cap; decode_body must still refuse it.
    oversize_name = "a" * (_MAX_DUMP_FILENAME_SIZE + 1)
    bogus_body = encode_text(oversize_name, max_size=4096, label="database name")
    with pytest.raises(DecodeError):
        DumpRequest.decode_body(bogus_body)


def test_dump_request_pre_fix_4kib_no_longer_accepted() -> None:
    """Regression against the pre-fix 4 KiB cap: a 2 KiB name must now refuse."""
    name = "a" * 2048
    with pytest.raises(EncodeError):
        DumpRequest(name).encode_body()
