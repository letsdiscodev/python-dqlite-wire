"""Pin: ``DumpRequest.name`` is capped at the C gateway's WAL
filename ceiling (1019 bytes).

The C gateway (``dqlite-upstream src/gateway.c::handle_dump``) uses
a 1024-byte stack buffer for the WAL filename and reserves room for
the ``-wal`` suffix and a NUL terminator
(``1024 - len("-wal") - 1`` = 1019). A longer name encodes valid
wire bytes but the C side silently truncates the WAL filename,
returning a ``FilesResponse`` whose ``-wal`` entry refers to a
different path than the main entry — silent dump corruption.

This pin lives at the Python encoder / decoder layer so the
mismatch is rejected at the wire boundary rather than producing
torn output.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import DecodeError, EncodeError
from dqlitewire.messages.requests import DumpRequest
from dqlitewire.messages.responses import _MAX_DUMP_FILENAME_SIZE


def test_dump_request_filename_at_c_server_ceiling_accepted() -> None:
    """Exactly 1019 bytes — the C ceiling — must encode cleanly."""
    name = "a" * _MAX_DUMP_FILENAME_SIZE
    body = DumpRequest(name).encode_body()
    decoded = DumpRequest.decode_body(body)
    assert decoded.name == name


def test_dump_request_filename_one_past_c_server_ceiling_rejected() -> None:
    """One byte past the C ceiling — the WAL truncation boundary —
    must raise ``EncodeError`` so the silent-data-loss path is
    closed off at the Python boundary."""
    name = "a" * (_MAX_DUMP_FILENAME_SIZE + 1)
    with pytest.raises(EncodeError):
        DumpRequest(name).encode_body()


def test_dump_request_decoder_rejects_oversize_peer_request() -> None:
    """Symmetric pin on the decode side: a peer-supplied
    ``DumpRequest`` whose name exceeds the C ceiling must be
    rejected at decode time. Otherwise a misbehaving (or pre-fix)
    peer could send a request the encoder would refuse, breaking
    the wire-symmetric contract."""
    from dqlitewire.types import encode_text

    # Bypass DumpRequest's encoder (which already enforces the cap)
    # and synthesise a wire body whose length-prefixed text exceeds
    # the C ceiling. Use the lax 4 KiB encoder cap to emit a
    # legal-looking text frame; decode_body must refuse it.
    oversize_name = "a" * (_MAX_DUMP_FILENAME_SIZE + 1)
    bogus_body = encode_text(oversize_name, max_size=4096, label="database name")
    with pytest.raises(DecodeError):
        DumpRequest.decode_body(bogus_body)


def test_dump_request_pre_fix_4kib_no_longer_accepted() -> None:
    """Regression pin against the pre-fix 4 KiB cap. A 2 KiB name was
    silently encodable pre-fix; with the C-server cap in place it
    must now be refused."""
    name = "a" * 2048
    with pytest.raises(EncodeError):
        DumpRequest(name).encode_body()
