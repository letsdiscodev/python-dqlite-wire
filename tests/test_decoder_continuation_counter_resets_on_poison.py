"""Pin: every termination arm of ``decode_continuation`` (clean and
poison) finalises the per-stream counters via
``_finalize_continuation_state`` so a future telemetry accessor or
partial reset cannot surface stale data.

The previous code asymmetric: clean-exit arms (FAILURE, EMPTY,
wrong-type, has_more=False) cleared all four counters; the two
poison arms (oversized read_message DecodeError, broad BaseException
catch) cleared at most one. Adopt the helper everywhere and pin
each termination shape.
"""

from __future__ import annotations

import pytest

from dqlitewire.codec import MessageDecoder, MessageEncoder
from dqlitewire.exceptions import DecodeError
from dqlitewire.messages.responses import RowsResponse


def _drive_to_continuation(decoder: MessageDecoder, columns: int = 2) -> None:
    """Feed an initial ROWS frame with has_more=True so the decoder is
    in continuation mode with non-zero counters."""
    initial = RowsResponse(
        column_names=[f"c{i}" for i in range(columns)],
        rows=[[i for i in range(columns)]],
        has_more=True,
    )
    decoder.feed(MessageEncoder().encode(initial))
    decoder.decode()
    # State now: continuation expected, frame_count=1, total_rows=1,
    # column_count=columns, column_names tuple snapshotted.
    assert decoder._continuation_expected is True
    assert decoder._continuation_frame_count == 1
    assert decoder._continuation_total_rows == 1
    assert decoder._continuation_column_count == columns


def _assert_state_finalised(decoder: MessageDecoder) -> None:
    assert decoder._continuation_expected is False
    assert decoder._continuation_frame_count == 0
    assert decoder._continuation_total_rows == 0
    assert decoder._continuation_column_count is None
    assert decoder._continuation_column_names is None


def test_finalize_after_max_continuation_frames_cap_poison() -> None:
    decoder = MessageDecoder(is_request=False, max_continuation_frames=1)
    _drive_to_continuation(decoder)
    # Feed a second continuation frame — exceeds cap of 1.
    cont = RowsResponse(column_names=["c0", "c1"], rows=[[0, 0]], has_more=False)
    decoder.feed(MessageEncoder().encode(cont))
    with pytest.raises(DecodeError, match="max_continuation_frames"):
        decoder.decode_continuation()
    _assert_state_finalised(decoder)


def test_finalize_after_max_total_rows_cap_poison() -> None:
    decoder = MessageDecoder(is_request=False, max_total_rows=1)
    _drive_to_continuation(decoder)
    # Feed a continuation frame whose total would exceed 1.
    cont = RowsResponse(column_names=["c0", "c1"], rows=[[0, 0]], has_more=False)
    decoder.feed(MessageEncoder().encode(cont))
    with pytest.raises(DecodeError, match="max_total_rows"):
        decoder.decode_continuation()
    _assert_state_finalised(decoder)


def test_finalize_after_column_count_drift_poison() -> None:
    decoder = MessageDecoder(is_request=False)
    _drive_to_continuation(decoder, columns=2)
    # Continuation with different column count.
    cont = RowsResponse(column_names=["c0"], rows=[[0]], has_more=False)
    decoder.feed(MessageEncoder().encode(cont))
    with pytest.raises(DecodeError, match="column count drift"):
        decoder.decode_continuation()
    _assert_state_finalised(decoder)


def test_finalize_after_clean_final_frame() -> None:
    """Clean-exit arm: counters back to defaults after has_more=False."""
    decoder = MessageDecoder(is_request=False)
    _drive_to_continuation(decoder)
    cont = RowsResponse(column_names=["c0", "c1"], rows=[[0, 0]], has_more=False)
    decoder.feed(MessageEncoder().encode(cont))
    decoder.decode_continuation()
    _assert_state_finalised(decoder)


def test_decode_initial_frame_cap_exceeded_does_not_populate_continuation_state() -> None:
    """Sixth termination path: the cap check inside ``decode()`` (NOT
    ``decode_continuation``) at codec.py:685-707 fires when the FIRST
    has_more=True frame already exceeds ``max_total_rows``. The
    pre-existing code populated the five per-stream fields BEFORE
    raising, leaving them dirty after poison. The fix moves the cap
    check above the state mutation so the fields never get written.

    The hazard the helper docstring at codec.py:392-407 warned about
    ("a future telemetry accessor or a partial reset would surface
    the stale data"): the five fields are publicly readable and the
    docstring was written knowing this arm existed.
    """
    decoder = MessageDecoder(is_request=False, max_total_rows=1)
    # Initial frame with 2 rows — exceeds cap of 1 on first frame.
    initial = RowsResponse(
        column_names=["c0", "c1"],
        rows=[[1, 2], [3, 4]],
        has_more=True,
    )
    decoder.feed(MessageEncoder().encode(initial))
    with pytest.raises(DecodeError, match="max_total_rows"):
        decoder.decode()
    # Continuation fields must NOT be populated. Functionally the
    # buffer is poisoned and every subsequent decoder call short-
    # circuits via ``_check_poisoned``, but a debugger inspection or
    # operator-facing diagnostic touch must not see stale data.
    _assert_state_finalised(decoder)
