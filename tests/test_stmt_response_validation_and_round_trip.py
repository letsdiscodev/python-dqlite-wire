"""Pin: ``StmtResponse.__post_init__`` validates uint ranges and
``StmtResponse(schema=1, tail_offset=None)`` normalises to
``tail_offset=0`` so encode/decode round-trip preserves equality.

Sibling Response classes (``FailureResponse``, ``LeaderResponse``,
``HeartbeatResponse``) all run ``_validate_uint*`` in ``__post_init__``.
``StmtResponse`` was the lone outlier, silently accepting negative
ints, ints exceeding the wire field width, and bool-as-int — every
other Response rejects these at construction.

The V1-implicit-zero shape (``schema=1`` with ``tail_offset=None``)
encodes identically to ``tail_offset=0`` but decoded back to a
non-equal instance. Normalise on construction so the wire-equivalent
forms compare equal.
"""

from __future__ import annotations

import pytest

from dqlitewire.exceptions import EncodeError
from dqlitewire.messages.responses import StmtResponse
from dqlitewire.tuples import _MAX_PARAM_COUNT


class TestPostInitValidation:
    def test_negative_db_id_rejected(self) -> None:
        with pytest.raises(EncodeError, match="db_id"):
            StmtResponse(db_id=-1, stmt_id=0, num_params=0)

    def test_negative_stmt_id_rejected(self) -> None:
        with pytest.raises(EncodeError, match="stmt_id"):
            StmtResponse(db_id=0, stmt_id=-1, num_params=0)

    def test_db_id_overflow_rejected(self) -> None:
        with pytest.raises(EncodeError, match="db_id"):
            StmtResponse(db_id=2**32, stmt_id=0, num_params=0)

    def test_stmt_id_overflow_rejected(self) -> None:
        with pytest.raises(EncodeError, match="stmt_id"):
            StmtResponse(db_id=0, stmt_id=2**32, num_params=0)

    def test_num_params_overflow_rejected(self) -> None:
        with pytest.raises(EncodeError, match="num_params"):
            StmtResponse(db_id=0, stmt_id=0, num_params=2**64)

    def test_negative_tail_offset_rejected(self) -> None:
        with pytest.raises(EncodeError, match="tail_offset"):
            StmtResponse(db_id=0, stmt_id=0, num_params=0, tail_offset=-1)

    def test_bool_db_id_rejected(self) -> None:
        # bool is a subclass of int in Python; sibling validators
        # reject bool explicitly so a typo like ``db_id=True`` doesn't
        # silently get encoded as 1.
        with pytest.raises(EncodeError, match="db_id"):
            StmtResponse(db_id=True, stmt_id=0, num_params=0)

    def test_num_params_above_max_rejected_at_construction(self) -> None:
        """``num_params > _MAX_PARAM_COUNT`` must be rejected at
        construction, mirroring ``ResultResponse.__post_init__``'s
        ``_MAX_ROWS_AFFECTED`` cap. Without this, a constructed
        ``StmtResponse(num_params=2**40)`` is admitted, accepted in
        equality / hashing / pickle round-trips, and only fails when
        ``encode_body()`` is called — a construction-vs-encode
        asymmetry against the established sibling pattern.
        """
        with pytest.raises(EncodeError, match="num_params"):
            StmtResponse(db_id=0, stmt_id=0, num_params=2**40)

    def test_num_params_at_max_allowed(self) -> None:
        """The cap is inclusive: ``num_params == _MAX_PARAM_COUNT``
        must construct cleanly so the construction-time cap doesn't
        narrow the legal range below what the wire admits."""
        r = StmtResponse(db_id=0, stmt_id=0, num_params=_MAX_PARAM_COUNT)
        assert r.num_params == _MAX_PARAM_COUNT

    def test_num_params_encode_cap_still_enforced(self) -> None:
        """Defense-in-depth: even though ``__post_init__`` now blocks
        out-of-range ``num_params`` at construction, ``encode_body``
        keeps its own cap so a future regression in the constructor
        (or an instance built via ``object.__new__``-style bypass)
        still cannot emit an out-of-range wire frame.
        """
        r = StmtResponse(db_id=0, stmt_id=0, num_params=0)
        # Bypass the constructor cap to simulate a future regression.
        r.num_params = _MAX_PARAM_COUNT + 1
        with pytest.raises(EncodeError, match="num_params"):
            r.encode_body()


class TestRoundTripIdentity:
    def test_v1_implicit_zero_normalises_to_zero(self) -> None:
        """``StmtResponse(schema=1, tail_offset=None)`` encodes
        identically to ``tail_offset=0``; normalise on construction
        so the dataclass is comparable to the decode result."""
        r = StmtResponse(db_id=1, stmt_id=2, num_params=3, schema=1)
        assert r.tail_offset == 0

    def test_v1_round_trip_identity_implicit_zero(self) -> None:
        r1 = StmtResponse(db_id=1, stmt_id=2, num_params=3, schema=1)
        body = r1.encode_body()
        r2 = StmtResponse.decode_body(body, schema=1)
        assert r1 == r2

    def test_v1_round_trip_identity_explicit_zero(self) -> None:
        r1 = StmtResponse(db_id=1, stmt_id=2, num_params=3, tail_offset=0, schema=1)
        body = r1.encode_body()
        r2 = StmtResponse.decode_body(body, schema=1)
        assert r1 == r2

    def test_v0_round_trip_identity_with_explicit_schema(self) -> None:
        """V0 round-trip preserves wire bytes; for dataclass equality
        the original needs an explicit ``schema=0`` so the decoder's
        schema-byte preservation matches."""
        r1 = StmtResponse(db_id=1, stmt_id=2, num_params=3, schema=0)
        body = r1.encode_body()
        r2 = StmtResponse.decode_body(body, schema=0)
        assert r1 == r2
