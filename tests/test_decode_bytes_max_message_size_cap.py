"""Pin: ``MessageDecoder.decode_bytes`` enforces ``max_message_size``.

The streaming path (``feed()`` → ``decode()``) routes through
``ReadBuffer.read_message`` which enforces the cap. The stateless
``decode_bytes`` path went directly through ``Header.decode`` + a
``len(data)`` truncation check with no cap consultation — a caller
using ``decode_bytes`` (or the ``decode_message`` convenience helper
that wraps it) silently bypassed the envelope cap the package
documents.

This pin exercises both entry points (the method and the helper) and
confirms the cap is consulted on each.
"""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder, decode_message
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.base import Header


def _build_oversize_frame(body_size: int) -> bytes:
    """Build a synthetic frame whose header advertises ``body_size``
    bytes of body, padded out to that size with zeros. ``msg_type=0``
    is ``ResponseType.FAILURE`` — irrelevant to the cap check, which
    fires before per-class dispatch."""
    header = Header(size_words=body_size // 8, msg_type=0).encode()
    body = b"\x00" * body_size
    return header + body


class TestDecodeBytesCap:
    def test_decode_bytes_enforces_max_message_size(self) -> None:
        """A frame larger than ``max_message_size`` must be rejected
        on the stateless path, matching the streaming path's behaviour."""
        cap = 1024 * 1024  # 1 MiB cap
        body = 2 * 1024 * 1024  # 2 MiB body
        wire = _build_oversize_frame(body)

        dec = MessageDecoder(max_message_size=cap)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            dec.decode_bytes(wire)

    def test_decode_bytes_accepts_under_cap_boundary(self) -> None:
        """Defense-in-depth: a frame just under the cap must NOT be
        rejected by the new check (it may still fail per-class
        decode for unrelated reasons). The cap should not over-fire."""
        cap = 64 * 1024  # 64 KiB cap
        # 4 KiB body — well under the cap.
        wire = _build_oversize_frame(4096)
        dec = MessageDecoder(max_message_size=cap)
        # The frame is well-formed at the envelope level, so the cap
        # must not fire. Per-class decode may then succeed or raise for
        # an unrelated reason, but the failure must NOT be the cap
        # exceedance.
        try:
            dec.decode_bytes(wire)
        except DecodeError as exc:
            assert "exceeds maximum" not in str(exc)


class TestDecodeBytesShortHeaderStillTakesShortPath:
    """A torn-header input must still produce the ``Message too short``
    diagnostic. The new cap check must apply AFTER ``Header.decode``
    succeeds, not before."""

    def test_torn_header_short_message_rejected_with_short_diagnostic(self) -> None:
        dec = MessageDecoder(max_message_size=1024)
        with pytest.raises(DecodeError, match="too short"):
            dec.decode_bytes(b"\x00\x00\x00")


class TestDecodeMessageHelperForwardsCap:
    """The ``decode_message`` convenience helper now accepts the four
    cap kwargs that ``MessageDecoder`` exposes. Without this, a caller
    using the helper had no escape from the default caps for legitimate
    large frames captured from real clusters."""

    def test_decode_message_default_cap_rejects_oversize(self) -> None:
        """Sanity: the helper inherits the constructor default cap."""
        # 65 MiB body > 64 MiB default cap. Build a body of zeros so the
        # bytes-level allocation stays cheap on CI runners.
        big_body = 65 * 1024 * 1024
        wire = _build_oversize_frame(big_body)
        with pytest.raises(DecodeError, match="exceeds maximum"):
            decode_message(wire)

    def test_decode_message_max_message_size_kwarg_overrides_cap(self) -> None:
        """A caller passing ``max_message_size=...`` raises the cap.
        The frame below would be rejected at the default cap; with
        the override the cap check passes."""
        body = 2 * 1024 * 1024
        wire = _build_oversize_frame(body)
        # Default cap (64 MiB) would accept this trivially; force the
        # interesting case via a small cap.
        with pytest.raises(DecodeError, match="exceeds maximum"):
            decode_message(wire, max_message_size=1024 * 1024)
        # Raise the cap above the frame: the cap check must not fire.
        # Per-class dispatch may still raise for an unrelated reason,
        # but never for the cap.
        try:
            decode_message(wire, max_message_size=4 * 1024 * 1024)
        except DecodeError as exc:
            assert "exceeds maximum" not in str(exc)

    def test_decode_message_max_rows_kwarg_forwarded(self) -> None:
        """``max_rows`` reaches the inner ``MessageDecoder``. A frame
        encoding a ``RowsResponse`` with row count above the cap
        should be rejected with the row-count diagnostic when the
        cap is at the default, and accepted when raised."""
        from dqlitewire.codec import encode_message
        from dqlitewire.messages.responses import RowsResponse
        from dqlitewire.types import WireValue

        # 5 rows, 1 column. Pick a cap below 5 to force the failure
        # and a cap at/above 5 to clear it.
        rows: list[list[WireValue]] = [[i] for i in range(5)]
        msg = RowsResponse(column_names=["a"], rows=rows)
        wire = encode_message(msg)

        with pytest.raises(DecodeError, match="max_rows"):
            decode_message(wire, max_rows=3)
        # Raise the cap; decode succeeds.
        decoded = decode_message(wire, max_rows=10)
        assert isinstance(decoded, RowsResponse)
        assert len(decoded.rows) == 5

    def test_decode_message_continuation_caps_omitted_default_intact(self) -> None:
        """Sentinel: omitting ``max_continuation_frames`` / ``max_total_rows``
        keeps the ``MessageDecoder`` default. Passing ``None`` explicitly
        disables the cap (legitimate operator opt-out)."""
        # The helper is stateless and decodes a single frame, so
        # the continuation caps are mostly inert here; pinning the
        # plumbing keeps the kwarg surface honest.
        from dqlitewire.codec import encode_message
        from dqlitewire.messages.responses import RowsResponse

        msg = RowsResponse(column_names=["a"], rows=[[1]])
        wire = encode_message(msg)
        # Default omission: works.
        decode_message(wire)
        # Explicit None: works (operator opt-out).
        decode_message(wire, max_continuation_frames=None, max_total_rows=None)


class TestFilesResponseContentCap:
    """Pin: each ``FilesResponse`` file's content is capped at
    ``_MAX_FILE_CONTENT_SIZE`` independently of the frame envelope.

    Without the per-file cap, a single hostile entry can consume the
    entire ``max_message_size`` budget, defeating the per-file count
    cap. Sibling fields (BLOB, TEXT, failure message, tail offset) all
    have their own per-field caps below the envelope; FILES content
    was the outlier.
    """

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

        # Build a synthetic body that claims a content size above the
        # per-file cap. The body does NOT need to actually carry the
        # oversize content (the cap check fires before the over-read
        # bounds check); a short prefix is enough.
        oversize_size = _MAX_FILE_CONTENT_SIZE + 8
        body = (
            encode_uint64(1)  # count
            + encode_text("main", max_size=4096, label="filename")
            + encode_uint64(oversize_size)
            # No actual content — cap check fires first.
        )
        with pytest.raises(DecodeError, match="exceeds maximum"):
            FilesResponse.decode_body(body)

    def test_files_response_at_cap_round_trips(self) -> None:
        """The cap is exclusive: a file of exactly ``_MAX_FILE_CONTENT_SIZE``
        bytes must encode and decode cleanly. Defense-in-depth
        regression guard."""
        from dqlitewire.messages.responses import (
            _MAX_FILE_CONTENT_SIZE,
            FilesResponse,
        )

        at_cap = b"\x00" * _MAX_FILE_CONTENT_SIZE
        encoded = FilesResponse(files={"main": at_cap}).encode_body()
        decoded = FilesResponse.decode_body(encoded)
        assert decoded.files["main"] == at_cap
