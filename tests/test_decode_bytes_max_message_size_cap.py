"""Pin: ``MessageDecoder.decode_bytes`` (and the ``decode_message`` helper
that wraps it) enforces ``max_message_size``, matching the streaming path.
The stateless path previously skipped the cap entirely."""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder, decode_message
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.base import Header


def _build_oversize_frame(body_size: int) -> bytes:
    """Frame advertising ``body_size`` bytes of zero body. ``msg_type=0`` is
    irrelevant: the cap check fires before per-class dispatch."""
    header = Header(size_words=body_size // 8, msg_type=0).encode()
    body = b"\x00" * body_size
    return header + body


class TestDecodeBytesCap:
    def test_decode_bytes_enforces_max_message_size(self) -> None:
        cap = 1024 * 1024  # 1 MiB cap
        body = 2 * 1024 * 1024  # 2 MiB body
        wire = _build_oversize_frame(body)

        dec = MessageDecoder(max_message_size=cap)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            dec.decode_bytes(wire)

    def test_decode_bytes_accepts_under_cap_boundary(self) -> None:
        """A frame under the cap must not be rejected by the cap check (it may
        still fail per-class decode for unrelated reasons)."""
        cap = 64 * 1024  # 64 KiB cap
        wire = _build_oversize_frame(4096)
        dec = MessageDecoder(max_message_size=cap)
        try:
            dec.decode_bytes(wire)
        except DecodeError as exc:
            assert "exceeds maximum" not in str(exc)


class TestDecodeBytesShortHeaderStillTakesShortPath:
    """A torn-header input still produces the short diagnostic: the cap check
    applies after ``Header.decode`` succeeds, not before."""

    def test_torn_header_short_message_rejected_with_short_diagnostic(self) -> None:
        dec = MessageDecoder(max_message_size=1024)
        with pytest.raises(DecodeError, match="too short"):
            dec.decode_bytes(b"\x00\x00\x00")


class TestDecodeMessageHelperForwardsCap:
    """The ``decode_message`` helper forwards the four cap kwargs that
    ``MessageDecoder`` exposes, so callers can raise caps for legitimately
    large frames."""

    def test_decode_message_default_cap_rejects_oversize(self) -> None:
        # 65 MiB body > 64 MiB default cap; zero body to keep allocation cheap.
        big_body = 65 * 1024 * 1024
        wire = _build_oversize_frame(big_body)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            decode_message(wire)

    def test_decode_message_max_message_size_kwarg_overrides_cap(self) -> None:
        """``max_message_size=...`` raises the cap."""
        body = 2 * 1024 * 1024
        wire = _build_oversize_frame(body)
        # Small cap rejects; raised cap must not fire (per-class dispatch may
        # still raise for an unrelated reason).
        with pytest.raises(DecodeError, match="exceeds maximum"):
            decode_message(wire, max_message_size=1024 * 1024)
        try:
            decode_message(wire, max_message_size=4 * 1024 * 1024)
        except DecodeError as exc:
            assert "exceeds maximum" not in str(exc)

    def test_decode_message_max_rows_kwarg_forwarded(self) -> None:
        """``max_rows`` reaches the inner ``MessageDecoder``."""
        from dqlitewire.codec import encode_message
        from dqlitewire.messages.responses import RowsResponse
        from dqlitewire.types import WireValue

        rows: list[list[WireValue]] = [[i] for i in range(5)]
        msg = RowsResponse(column_names=["a"], rows=rows)
        wire = encode_message(msg)

        with pytest.raises(DecodeError, match="max_rows"):
            decode_message(wire, max_rows=3)
        decoded = decode_message(wire, max_rows=10)
        assert isinstance(decoded, RowsResponse)
        assert len(decoded.rows) == 5

    def test_decode_message_continuation_caps_omitted_default_intact(self) -> None:
        """Omitting the continuation caps keeps the default; explicit ``None``
        disables them (operator opt-out)."""
        from dqlitewire.codec import encode_message
        from dqlitewire.messages.responses import RowsResponse

        msg = RowsResponse(column_names=["a"], rows=[[1]])
        wire = encode_message(msg)
        decode_message(wire)
        decode_message(wire, max_continuation_frames=None, max_total_rows=None)


class TestFilesResponseContentCap:
    """Pin: each ``FilesResponse`` file's content is capped at
    ``_MAX_FILE_CONTENT_SIZE`` independently of the envelope, so one hostile
    entry can't consume the whole ``max_message_size`` budget."""

    def test_files_response_encode_content_over_cap_rejected(self) -> None:
        from dqlitewire.exceptions import EncodeError
        from dqlitewire.messages.responses import (
            _MAX_FILE_CONTENT_SIZE,
            FilesResponse,
        )

        oversize = b"\x00" * (_MAX_FILE_CONTENT_SIZE + 8)
        with pytest.raises(EncodeError, match="exceeds maximum"):
            FilesResponse(files={"main": oversize}).encode_body()

    def test_files_response_decode_content_over_cap_rejected(self) -> None:
        from dqlitewire.messages.responses import _MAX_FILE_CONTENT_SIZE, FilesResponse
        from dqlitewire.types import encode_text, encode_uint64

        # Body claims an oversize content size but carries none: the cap check
        # fires before the over-read bounds check.
        oversize_size = _MAX_FILE_CONTENT_SIZE + 8
        body = (
            encode_uint64(1)  # count
            + encode_text("main", max_size=4096, label="filename")
            + encode_uint64(oversize_size)
        )
        with pytest.raises(DecodeError, match="exceeds maximum"):
            FilesResponse.decode_body(body)

    def test_files_response_at_cap_round_trips(self) -> None:
        """The cap is exclusive: a file of exactly ``_MAX_FILE_CONTENT_SIZE``
        bytes round-trips cleanly."""
        from dqlitewire.messages.responses import (
            _MAX_FILE_CONTENT_SIZE,
            FilesResponse,
        )

        at_cap = b"\x00" * _MAX_FILE_CONTENT_SIZE
        encoded = FilesResponse(files={"main": at_cap}).encode_body()
        decoded = FilesResponse.decode_body(encoded)
        assert decoded.files["main"] == at_cap
