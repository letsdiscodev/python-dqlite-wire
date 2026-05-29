"""StmtResponse.__post_init__ validates uint ranges and normalises
schema=1/tail_offset=None to tail_offset=0 so encode/decode round-trips equal."""

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
        # bool is an int subclass, so db_id=True must be rejected (not encoded as 1).
        with pytest.raises(EncodeError, match="db_id"):
            StmtResponse(db_id=True, stmt_id=0, num_params=0)

    def test_num_params_above_max_rejected_at_construction(self) -> None:
        """num_params > _MAX_PARAM_COUNT is rejected at construction, not
        deferred to encode_body."""
        with pytest.raises(EncodeError, match="num_params"):
            StmtResponse(db_id=0, stmt_id=0, num_params=2**40)

    def test_num_params_at_max_allowed(self) -> None:
        """The cap is inclusive: num_params == _MAX_PARAM_COUNT constructs cleanly."""
        r = StmtResponse(db_id=0, stmt_id=0, num_params=_MAX_PARAM_COUNT)
        assert r.num_params == _MAX_PARAM_COUNT

    def test_num_params_encode_cap_still_enforced(self) -> None:
        """Defense-in-depth: encode_body keeps its own num_params cap even
        though __post_init__ now blocks it at construction."""
        r = StmtResponse(db_id=0, stmt_id=0, num_params=0)
        r.num_params = _MAX_PARAM_COUNT + 1
        with pytest.raises(EncodeError, match="num_params"):
            r.encode_body()


class TestRoundTripIdentity:
    def test_v1_implicit_zero_normalises_to_zero(self) -> None:
        """schema=1/tail_offset=None normalises to 0 so it equals the decode result."""
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
        """V0 dataclass equality needs an explicit schema=0 to match the decoder."""
        r1 = StmtResponse(db_id=1, stmt_id=2, num_params=3, schema=0)
        body = r1.encode_body()
        r2 = StmtResponse.decode_body(body, schema=0)
        assert r1 == r2
