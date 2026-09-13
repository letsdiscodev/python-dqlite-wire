"""Tests for protocol constants and error-code tables."""

from __future__ import annotations

import sqlite3

import pytest

import dqlitewire
from dqlitewire import (
    BARE_DATABASE_ERROR_CODES,
    LEADER_ERROR_CODES,
    LEADER_LOST_DB_LOOKUP_SUBSTRING,
    SQLITE_CORRUPT,
    SQLITE_FORMAT,
    SQLITE_NOTADB,
    WIRE_DECODE_FAILED_PREFIX,
)
from dqlitewire.constants import (
    DQLITE_NOTFOUND,
    DQLITE_PARSE,
    DQLITE_PROTO,
    SQLITE_BUSY,
    SQLITE_ERROR,
    SQLITE_IOERR,
    SQLITE_IOERR_LEADERSHIP_LOST,
    SQLITE_IOERR_LEADERSHIP_LOST_LEGACY,
    SQLITE_IOERR_NOT_LEADER,
    SQLITE_IOERR_NOT_LEADER_LEGACY,
    SQLITE_NOTFOUND,
    SQLITE_PROTOCOL,
    TX_AUTO_ROLLBACK_PRIMARY_CODES,
    RequestType,
    ResponseType,
    is_dqlite_namespace_code,
    primary_sqlite_code,
)


class TestResponseTypeValues:
    """Verify ResponseType enum matches go-dqlite constants.go.

    Reference: github.com/canonical/go-dqlite internal/protocol/constants.go
    """

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("FAILURE", 0),
            ("LEADER", 1),
            ("WELCOME", 2),
            ("SERVERS", 3),  # Go: ResponseNodes = 3 (cluster server listing)
            ("DB", 4),
            ("STMT", 5),
            ("RESULT", 6),
            ("ROWS", 7),
            ("EMPTY", 8),
            ("FILES", 9),
            ("METADATA", 10),  # Go: ResponseMetadata = 10
        ],
    )
    def test_response_type_value(self, name: str, value: int) -> None:
        assert getattr(ResponseType, name) == value

    def test_no_node_legacy_as_separate_type(self) -> None:
        """NODE_LEGACY should not exist as a separate type code.

        In Go, ResponseNodeLegacy = 1 (same as ResponseNode).
        """
        assert not hasattr(ResponseType, "NODE_LEGACY")


class TestRequestTypeValues:
    """Verify RequestType enum matches go-dqlite constants.go."""

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("LEADER", 0),
            ("CLIENT", 1),
            ("OPEN", 3),
            ("PREPARE", 4),
            ("EXEC", 5),
            ("QUERY", 6),
            ("FINALIZE", 7),
            ("EXEC_SQL", 8),
            ("QUERY_SQL", 9),
            ("INTERRUPT", 10),
            # C defines DQLITE_REQUEST_CONNECT = 11 for Raft transport; the Go
            # client omits it (client-only), but a complete implementation includes it.
            ("CONNECT", 11),
            ("ADD", 12),
            ("ASSIGN", 13),
            ("REMOVE", 14),
            ("DUMP", 15),
            ("CLUSTER", 16),
            ("TRANSFER", 17),
            ("DESCRIBE", 18),
            ("WEIGHT", 19),
        ],
    )
    def test_request_type_value(self, name: str, value: int) -> None:
        assert getattr(RequestType, name) == value


class TestPublicExports:
    """Verify important types are importable from public API paths."""

    def test_nodeinfo_importable_from_messages(self) -> None:
        from dqlitewire.messages import NodeInfo

        assert NodeInfo is not None
        # Same class ServersResponse uses.
        from dqlitewire.messages.responses import NodeInfo as DirectNodeInfo

        assert NodeInfo is DirectNodeInfo


class TestNodeRoleValues:
    """Verify NodeRole enum matches Go's protocol constants."""

    def test_voter_is_0(self) -> None:
        from dqlitewire.constants import NodeRole

        assert NodeRole.VOTER == 0

    def test_standby_is_1(self) -> None:
        from dqlitewire.constants import NodeRole

        assert NodeRole.STANDBY == 1

    def test_spare_is_2(self) -> None:
        from dqlitewire.constants import NodeRole

        assert NodeRole.SPARE == 2

    def test_importable_from_top_level(self) -> None:
        from dqlitewire import NodeRole

        assert NodeRole.VOTER == 0


class TestTypeDictCompleteness:
    """Verify REQUEST_TYPES and RESPONSE_TYPES cover all enum members."""

    def test_request_types_covers_all_enum_members(self) -> None:
        from dqlitewire.codec import REQUEST_TYPES

        # HEARTBEAT and CONNECT are reserved upstream
        # (``DQLITE_REQUEST_HEARTBEAT = 2``, ``DQLITE_REQUEST_CONNECT = 11``)
        # but upstream C's dispatcher falls through to ``DQLITE_PARSE`` for
        # both — HEARTBEAT is a transport ping that Go/C clients never send,
        # and CONNECT is a Raft-transport frame used only for inter-node
        # connections. The classes (``_HeartbeatRequest``, ``_ConnectRequest``)
        # stay private and are intentionally absent from ``REQUEST_TYPES``.
        # Every other enum member is a real public request type.
        private_dispatch_excluded = {RequestType.HEARTBEAT, RequestType.CONNECT}
        for member in RequestType:
            if member in private_dispatch_excluded:
                assert member.value not in REQUEST_TYPES, (
                    f"RequestType.{member.name} must stay out of REQUEST_TYPES; "
                    "upstream does not dispatch this frame to the gateway."
                )
                continue
            assert member.value in REQUEST_TYPES, (
                f"RequestType.{member.name} ({member.value}) has no entry in REQUEST_TYPES"
            )

    def test_response_types_covers_all_enum_members(self) -> None:
        from dqlitewire.codec import RESPONSE_TYPES

        for member in ResponseType:
            assert member.value in RESPONSE_TYPES, (
                f"ResponseType.{member.name} ({member.value}) has no entry in RESPONSE_TYPES"
            )


class TestLeaderErrorCodes:
    """Pin the SQLite extended error codes that signal leader changes."""

    def test_sqlite_ioerr_base(self) -> None:
        from dqlitewire.constants import SQLITE_IOERR

        assert SQLITE_IOERR == 10

    def test_not_leader_code_value(self) -> None:
        from dqlitewire.constants import SQLITE_IOERR, SQLITE_IOERR_NOT_LEADER

        assert SQLITE_IOERR_NOT_LEADER == 10250
        assert SQLITE_IOERR_NOT_LEADER == SQLITE_IOERR | (40 << 8)

    def test_leadership_lost_code_value(self) -> None:
        from dqlitewire.constants import SQLITE_IOERR, SQLITE_IOERR_LEADERSHIP_LOST

        assert SQLITE_IOERR_LEADERSHIP_LOST == 10506
        assert SQLITE_IOERR_LEADERSHIP_LOST == SQLITE_IOERR | (41 << 8)

    def test_leader_error_codes_is_a_frozenset_of_modern_and_legacy(self) -> None:
        from dqlitewire.constants import (
            LEADER_ERROR_CODES,
            SQLITE_IOERR_LEADERSHIP_LOST,
            SQLITE_IOERR_LEADERSHIP_LOST_LEGACY,
            SQLITE_IOERR_NOT_LEADER,
            SQLITE_IOERR_NOT_LEADER_LEGACY,
        )

        assert isinstance(LEADER_ERROR_CODES, frozenset)
        # Modern (40/41) and legacy (32/33) sub-codes both present so a leader-flip on
        # a pre-3.32.1 server stays retryable; see test_leader_error_codes_legacy_subcodes.py.
        assert SQLITE_IOERR_NOT_LEADER in LEADER_ERROR_CODES
        assert SQLITE_IOERR_LEADERSHIP_LOST in LEADER_ERROR_CODES
        assert SQLITE_IOERR_NOT_LEADER_LEGACY in LEADER_ERROR_CODES
        assert SQLITE_IOERR_LEADERSHIP_LOST_LEGACY in LEADER_ERROR_CODES

    def test_leader_error_codes_importable_from_top_level(self) -> None:
        from dqlitewire import (
            LEADER_ERROR_CODES,
            SQLITE_IOERR,
            SQLITE_IOERR_LEADERSHIP_LOST,
            SQLITE_IOERR_NOT_LEADER,
        )

        assert SQLITE_IOERR == 10
        assert SQLITE_IOERR_NOT_LEADER == 10250
        assert SQLITE_IOERR_LEADERSHIP_LOST == 10506
        # Legacy variants cover pre-3.32.1 servers emitting (32/33<<8) for the same
        # leader-flip conditions; see test_leader_error_codes_legacy_subcodes.py.
        assert {10250, 10506, 8202, 8458} == set(LEADER_ERROR_CODES)


class TestPrimarySqliteCode:
    def test_mask_value(self) -> None:
        from dqlitewire.constants import SQLITE_PRIMARY_CODE_MASK

        assert SQLITE_PRIMARY_CODE_MASK == 0xFF

    def test_extended_ioerr_not_leader_unmasks_to_ioerr(self) -> None:
        from dqlitewire.constants import (
            SQLITE_IOERR,
            SQLITE_IOERR_NOT_LEADER,
            primary_sqlite_code,
        )

        assert primary_sqlite_code(SQLITE_IOERR_NOT_LEADER) == SQLITE_IOERR

    def test_extended_ioerr_leadership_lost_unmasks_to_ioerr(self) -> None:
        from dqlitewire.constants import (
            SQLITE_IOERR,
            SQLITE_IOERR_LEADERSHIP_LOST,
            primary_sqlite_code,
        )

        assert primary_sqlite_code(SQLITE_IOERR_LEADERSHIP_LOST) == SQLITE_IOERR

    def test_primary_code_passes_through_unchanged(self) -> None:
        from dqlitewire.constants import primary_sqlite_code

        # Identity for codes that already are primary (low byte only).
        assert primary_sqlite_code(1) == 1  # SQLITE_ERROR
        assert primary_sqlite_code(5) == 5  # SQLITE_BUSY
        assert primary_sqlite_code(10) == 10  # SQLITE_IOERR
        assert primary_sqlite_code(0) == 0  # SQLITE_OK

    def test_helper_importable_from_top_level(self) -> None:
        from dqlitewire import primary_sqlite_code

        assert primary_sqlite_code(10250) == 10

    @pytest.mark.parametrize(
        ("primary", "extended"),
        [
            (4, 4),  # SQLITE_ABORT
            (4, 4 | (2 << 8)),  # SQLITE_ABORT_ROLLBACK
            (9, 9),  # SQLITE_INTERRUPT
            (11, 11),  # SQLITE_CORRUPT
            (13, 13),  # SQLITE_FULL
        ],
    )
    def test_primary_sqlite_code_extracts_auto_rollback_primaries(
        self, primary: int, extended: int
    ) -> None:
        """Extraction for the auto-rollback primary set, so a refactor can't mishandle one."""
        from dqlitewire.constants import primary_sqlite_code

        assert primary_sqlite_code(extended) == primary


class TestAutoRollbackPrimaryCodes:
    """The auto-rollback primary code set the client tracker uses to clear
    ``_in_transaction`` after a server-side auto-rollback."""

    def test_set_matches_documented_values(self) -> None:
        from dqlitewire.constants import TX_AUTO_ROLLBACK_PRIMARY_CODES

        assert frozenset({4, 7, 9, 10, 11, 13}) == TX_AUTO_ROLLBACK_PRIMARY_CODES

    def test_set_importable_from_top_level(self) -> None:
        from dqlitewire import TX_AUTO_ROLLBACK_PRIMARY_CODES

        assert 4 in TX_AUTO_ROLLBACK_PRIMARY_CODES
        assert 7 in TX_AUTO_ROLLBACK_PRIMARY_CODES  # SQLITE_NOMEM
        assert 9 in TX_AUTO_ROLLBACK_PRIMARY_CODES
        assert 10 in TX_AUTO_ROLLBACK_PRIMARY_CODES
        assert 11 in TX_AUTO_ROLLBACK_PRIMARY_CODES
        assert 13 in TX_AUTO_ROLLBACK_PRIMARY_CODES

    def test_set_includes_sqlite_nomem(self) -> None:
        """SQLITE_NOMEM (7) is on SQLite's auto-rollback list
        (sqlite.org/lang_transaction.html): a server-side OOM tears down the txn,
        so the client must clear tracker state."""
        from dqlitewire.constants import SQLITE_NOMEM, TX_AUTO_ROLLBACK_PRIMARY_CODES

        assert SQLITE_NOMEM == 7
        assert SQLITE_NOMEM in TX_AUTO_ROLLBACK_PRIMARY_CODES

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("SQLITE_ABORT", 4),
            ("SQLITE_NOMEM", 7),
            ("SQLITE_INTERRUPT", 9),
            ("SQLITE_IOERR", 10),
            ("SQLITE_CORRUPT", 11),
            ("SQLITE_FULL", 13),
        ],
    )
    def test_named_constants(self, name: str, expected: int) -> None:
        import dqlitewire

        assert getattr(dqlitewire, name) == expected


class TestBareDatabaseErrorPrimaryCodes:
    """The SQLite primaries the SA dialect treats as slot-fatal (11/24/26 =
    CORRUPT/FORMAT/NOTADB), hosted in the wire layer as the single source of truth."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("SQLITE_CORRUPT", 11),
            ("SQLITE_FORMAT", 24),
            ("SQLITE_NOTADB", 26),
        ],
    )
    def test_constant_values(self, name: str, expected: int) -> None:
        import dqlitewire

        assert getattr(dqlitewire, name) == expected

    def test_constants_importable_from_top_level(self) -> None:
        from dqlitewire import SQLITE_CORRUPT, SQLITE_FORMAT, SQLITE_NOTADB

        assert SQLITE_CORRUPT == 11
        assert SQLITE_FORMAT == 24
        assert SQLITE_NOTADB == 26


class TestDqliteErrorCollidesWithSqliteErrorOnWire:
    """DQLITE_ERROR = 1 (dqlite.h) shares the wire low-byte with SQLITE_ERROR = 1.
    Upstream emits it from gateway.c on REQUEST_TRANSFER, which the Python client never
    sends — so the collision is latent and disambiguated only by message text in
    dbapi.connection._is_no_transaction_error. Pinned for whoever adds TRANSFER support."""

    def test_dqlite_error_value_one_shares_low_byte_with_sqlite_error(self) -> None:
        from dqlitewire.constants import primary_sqlite_code

        # DQLITE_ERROR=1 is deliberately not exported as a named constant: it would
        # invite importing the wrong one of the two colliding values.
        DQLITE_ERROR_LITERAL = 1
        SQLITE_ERROR_LITERAL = 1
        assert DQLITE_ERROR_LITERAL == SQLITE_ERROR_LITERAL
        assert primary_sqlite_code(DQLITE_ERROR_LITERAL) == 1

    def test_dqlite_error_is_not_classified_as_namespace_code(self) -> None:
        """is_dqlite_namespace_code covers only codes >= 1000; 1/2/3 must stay outside it,
        else a colliding DQLITE_ERROR=1 would mask the SQLite primary from dispatch."""
        from dqlitewire.constants import is_dqlite_namespace_code

        assert is_dqlite_namespace_code(1) is False
        assert is_dqlite_namespace_code(2) is False  # DQLITE_MISUSE collides
        assert is_dqlite_namespace_code(3) is False  # DQLITE_NOMEM collides


class TestDefaultRowsAndFramesCaps:
    """The default per-query caps defend against slow-drip / amplification attacks;
    pin the literal magnitudes so a refactor can't silently change them."""

    def test_default_max_total_rows_is_ten_million(self) -> None:
        from dqlitewire import DEFAULT_MAX_TOTAL_ROWS

        assert DEFAULT_MAX_TOTAL_ROWS == 10_000_000

    def test_default_max_continuation_frames_is_one_hundred_thousand(self) -> None:
        from dqlitewire import DEFAULT_MAX_CONTINUATION_FRAMES

        assert DEFAULT_MAX_CONTINUATION_FRAMES == 100_000


class TestInternalDecodeBoundsPinnedAgainstReadme:
    """Pin the four decode-time cap constants the README documents, so bumping one
    without updating the README fails here (as MAX_COLUMN_COUNT once drifted)."""

    def test_max_param_count_matches_sqlite_max_variable_number(self) -> None:
        from dqlitewire.limits import MAX_PARAM_COUNT

        # SQLITE_MAX_VARIABLE_NUMBER standard-build max (sqlite.org/limits.html); the
        # server can't bind beyond it, so matching it caps a malicious peer's allocations.
        assert MAX_PARAM_COUNT == 32_766

    def test_max_column_count_matches_sqlite_default(self) -> None:
        from dqlitewire.limits import MAX_COLUMN_COUNT

        # SQLITE_MAX_COLUMN default is 2000. The C server emits column count as uncapped
        # uint64 (STMT__MAX_COLUMNS is defined but unused), so this cap is defence-in-depth.
        assert MAX_COLUMN_COUNT == 2000

    def test_max_file_count_is_one_hundred(self) -> None:
        from dqlitewire.limits import MAX_FILE_COUNT

        assert MAX_FILE_COUNT == 100

    def test_max_node_count_is_ten_thousand(self) -> None:
        from dqlitewire.limits import MAX_NODE_COUNT

        assert MAX_NODE_COUNT == 10_000


class TestNamedPrimaryCodeValues:
    """Pin each named primary code at its spec value: the FailureResponse round-trip
    matrix uses the constant on both sides, so a silent value drift stays internally
    consistent and goes unnoticed without this.

    References: SQLite codes at sqlite.org/rescode.html; DQLITE_PROTO=1001 (protocol.h:9),
    DQLITE_NOTFOUND=1002 (lib/registry.h:13), DQLITE_PARSE=1005 (lib/serialize.h:14).
    """

    @pytest.mark.parametrize(
        ("name", "expected_value"),
        [
            ("SQLITE_ERROR", 1),
            ("SQLITE_ABORT", 4),
            ("SQLITE_BUSY", 5),
            ("SQLITE_NOMEM", 7),
            ("SQLITE_INTERRUPT", 9),
            ("SQLITE_IOERR", 10),
            ("SQLITE_CORRUPT", 11),
            ("SQLITE_FULL", 13),
            ("SQLITE_FORMAT", 24),
            ("SQLITE_NOTADB", 26),
            ("DQLITE_PROTO", 1001),
            ("DQLITE_NOTFOUND", 1002),
            ("DQLITE_PARSE", 1005),
        ],
    )
    def test_named_constant_value_pinned(self, name: str, expected_value: int) -> None:
        import dqlitewire

        actual = getattr(dqlitewire, name)
        assert actual == expected_value, (
            f"Named constant {name} value drift: got {actual}, "
            f"expected {expected_value} (per sqlite.org/rescode.html or "
            f"dqlite protocol.h / registry.h / serialize.h). A rebase "
            f"that silently changes the value breaks every cross-driver "
            f"classifier; pin the value here."
        )


# ---- merged from test_constants_bare_database_error_codes.py ----
# Pin ``BARE_DATABASE_ERROR_CODES``: the SSOT cross-package contract (SA
# derives its slot-fatal disconnect codes from it), so pin membership and
# frozenset shape against silent drift.


def test_bare_database_error_codes_membership() -> None:
    assert (
        frozenset(
            {
                SQLITE_CORRUPT,  # 11
                SQLITE_FORMAT,  # 24
                SQLITE_NOTADB,  # 26
            }
        )
        == BARE_DATABASE_ERROR_CODES
    )


def test_bare_database_error_codes_numeric_values() -> None:
    """Pin absolute numeric codes so a future ``SQLITE_*`` alias rename can't
    silently change the contract."""
    assert frozenset({11, 24, 26}) == BARE_DATABASE_ERROR_CODES


def test_bare_database_error_codes_is_frozenset() -> None:
    """Pin frozenset shape: SA uses the constant as a dict key / in subset
    comparisons, which a regression to plain ``set`` would break."""
    assert isinstance(BARE_DATABASE_ERROR_CODES, frozenset)


# ---- merged from test_dqlite_namespace_range.py ----
# ``is_dqlite_namespace_code`` uses the numeric range ``[1000, 1024)`` rather
# than a hard-coded set, so a future dqlite code is auto-included instead of
# masking through ``primary_sqlite_code(code & 0xFF)`` to a phantom primary.
# Extended SQLite codes (e.g. ``SQLITE_IOERR_NOT_LEADER = 10250``) sit outside
# the range and keep masking correctly.


@pytest.mark.parametrize("code", [DQLITE_PROTO, DQLITE_NOTFOUND, DQLITE_PARSE])
def test_known_namespace_codes_passthrough(code: int) -> None:
    assert is_dqlite_namespace_code(code) is True
    assert primary_sqlite_code(code) == code


def test_hypothetical_future_namespace_code_passthrough() -> None:
    """A code added upstream later (e.g. 1010) must pass through unchanged
    instead of masking to 0xF2 = 242."""
    future_code = 1010
    assert is_dqlite_namespace_code(future_code) is True
    assert primary_sqlite_code(future_code) == future_code


def test_entire_namespace_range_treated_as_namespace() -> None:
    for code in range(1000, 1024):
        assert is_dqlite_namespace_code(code) is True
        assert primary_sqlite_code(code) == code


def test_below_namespace_range_is_not_namespace() -> None:
    """999 is just below the range and must be excluded from namespace
    handling."""
    assert is_dqlite_namespace_code(999) is False


def test_at_or_above_namespace_range_is_not_namespace() -> None:
    """1024 and above are not namespace; ``SQLITE_IOERR_NOT_LEADER = 10250`` is
    a real extended code that must still mask to SQLITE_IOERR (10)."""
    assert is_dqlite_namespace_code(1024) is False
    assert is_dqlite_namespace_code(10250) is False
    assert primary_sqlite_code(10250) == 10


# ---- merged from test_leader_error_codes_legacy_subcodes.py ----
# LEADER_ERROR_CODES has modern (40/41) and legacy (32/33) sub-codes; old flips stay retryable.


def test_modern_codes_are_in_set() -> None:
    assert SQLITE_IOERR_NOT_LEADER == 10250
    assert SQLITE_IOERR_LEADERSHIP_LOST == 10506
    assert SQLITE_IOERR_NOT_LEADER in LEADER_ERROR_CODES
    assert SQLITE_IOERR_LEADERSHIP_LOST in LEADER_ERROR_CODES


def test_legacy_codes_are_in_set() -> None:
    assert SQLITE_IOERR_NOT_LEADER_LEGACY == 8202
    assert SQLITE_IOERR_LEADERSHIP_LOST_LEGACY == 8458
    assert SQLITE_IOERR_NOT_LEADER_LEGACY in LEADER_ERROR_CODES
    assert SQLITE_IOERR_LEADERSHIP_LOST_LEGACY in LEADER_ERROR_CODES


def test_set_contains_exactly_the_four_codes() -> None:
    """Cardinality pin so a future revert that drops the legacy pair
    fails deterministically."""
    assert len(LEADER_ERROR_CODES) == 4
    assert SQLITE_IOERR_NOT_LEADER in LEADER_ERROR_CODES
    assert SQLITE_IOERR_LEADERSHIP_LOST in LEADER_ERROR_CODES
    assert SQLITE_IOERR_NOT_LEADER_LEGACY in LEADER_ERROR_CODES
    assert SQLITE_IOERR_LEADERSHIP_LOST_LEGACY in LEADER_ERROR_CODES


def test_legacy_codes_compute_from_byte_shift() -> None:
    """Tie the constants to the byte-shift formula so a future
    refactor that re-derives the values can be sanity-checked."""
    assert SQLITE_IOERR_NOT_LEADER_LEGACY == SQLITE_IOERR | (32 << 8)
    assert SQLITE_IOERR_LEADERSHIP_LOST_LEGACY == SQLITE_IOERR | (33 << 8)


# ---- merged from test_leader_lost_db_lookup_substring_constant.py ----
# Pin: ``LEADER_LOST_DB_LOOKUP_SUBSTRING`` is the canonical wording
# gateway.c emits when ``g->leader == NULL`` after a Raft demotion.
#
# Paired with ``SQLITE_NOTFOUND`` (=12) and substring-matched on the
# server-supplied ``raw_message``, this is the wire-side
# mark-this-connection-dead signal that the client and SA layers
# dispatch on. Mirrors Go's ``driverError`` ``errNotFound → ErrBadConn``
# arm with the "potentially after leadership loss" comment.
#
# The constant lives at the wire layer so the four-way disconnect-
# classification chain (wire → client.connection → dbapi → SA) stays
# in lockstep on rename.
#
# The existing ``LEADER_ERROR_CODES`` cardinality pin in
# ``test_leader_error_codes_legacy_subcodes.py`` stays at 4: code 12
# is overloaded between LOOKUP_DB (leader flip — invalidate) and
# LOOKUP_STMT (server-side state bug — do not invalidate), so the
# discriminator is the substring not the numeric code.


def test_leader_lost_db_lookup_substring_value() -> None:
    """Exact value pinned against upstream gateway.c
    ``LOOKUP_DB`` macro emission text."""
    assert LEADER_LOST_DB_LOOKUP_SUBSTRING == "no database opened"


def test_leader_lost_db_lookup_substring_is_str_final() -> None:
    """Same shape as the sibling ``NO_TRANSACTION_MESSAGE_SUBSTRINGS``
    and ``WIRE_DECODE_FAILED_PREFIX`` — module-level ``Final[str]``."""
    assert isinstance(LEADER_LOST_DB_LOOKUP_SUBSTRING, str)
    from dqlitewire import LEADER_LOST_DB_LOOKUP_SUBSTRING as again

    assert again is LEADER_LOST_DB_LOOKUP_SUBSTRING


# ---- merged from test_primary_sqlite_code_dqlite_namespace_passthrough.py ----
# Pin: ``primary_sqlite_code`` passes dqlite-namespace codes (>= 1000)
# through unchanged (masking with 0xFF would yield a meaningless byte).


@pytest.mark.parametrize(
    "code",
    [
        DQLITE_PROTO,  # 1001
        DQLITE_NOTFOUND,  # 1002
        DQLITE_PARSE,  # 1005
    ],
)
def test_primary_sqlite_code_dqlite_namespace_passthrough(code: int) -> None:
    assert primary_sqlite_code(code) == code


def test_primary_sqlite_code_extended_sqlite_code_still_masked() -> None:
    """Extended SQLite codes still mask to their primary (2067 -> 19)."""
    assert primary_sqlite_code(2067) == 19


# ---- merged from test_promoted_sqlite_constants.py ----
# Pin: the 10 promoted SQLite primary codes and the 11 extended
# ``SQLITE_CONSTRAINT_*`` codes match stdlib ``sqlite3`` bit-for-bit
# and are reachable via the wire package's top-level surface.

PROMOTED_PRIMARIES = [
    "SQLITE_INTERNAL",
    "SQLITE_TOOBIG",
    "SQLITE_CONSTRAINT",
    "SQLITE_MISMATCH",
    "SQLITE_MISUSE",
    "SQLITE_NOLFS",
    "SQLITE_AUTH",
    "SQLITE_RANGE",
    "SQLITE_NOTICE",
    "SQLITE_WARNING",
]

CONSTRAINT_FAMILY = [
    "SQLITE_CONSTRAINT_CHECK",
    "SQLITE_CONSTRAINT_COMMITHOOK",
    "SQLITE_CONSTRAINT_FOREIGNKEY",
    "SQLITE_CONSTRAINT_FUNCTION",
    "SQLITE_CONSTRAINT_NOTNULL",
    "SQLITE_CONSTRAINT_PRIMARYKEY",
    "SQLITE_CONSTRAINT_TRIGGER",
    "SQLITE_CONSTRAINT_UNIQUE",
    "SQLITE_CONSTRAINT_VTAB",
    "SQLITE_CONSTRAINT_ROWID",
    "SQLITE_CONSTRAINT_PINNED",
]


@pytest.mark.parametrize("name", PROMOTED_PRIMARIES)
def test_promoted_primary_matches_stdlib(name: str) -> None:
    assert getattr(dqlitewire, name) == getattr(sqlite3, name), (
        f"{name} drift: dqlitewire={getattr(dqlitewire, name)} "
        f"vs stdlib sqlite3={getattr(sqlite3, name)}"
    )


@pytest.mark.parametrize("name", CONSTRAINT_FAMILY)
def test_constraint_extended_matches_stdlib(name: str) -> None:
    assert getattr(dqlitewire, name) == getattr(sqlite3, name), (
        f"{name} drift: dqlitewire={getattr(dqlitewire, name)} "
        f"vs stdlib sqlite3={getattr(sqlite3, name)}"
    )


def test_constraint_extended_share_primary_byte() -> None:
    """Every extended ``SQLITE_CONSTRAINT_*`` masks to ``SQLITE_CONSTRAINT``
    (the dbapi classifier's grouping invariant)."""
    for name in CONSTRAINT_FAMILY:
        value = getattr(dqlitewire, name)
        assert value & 0xFF == dqlitewire.SQLITE_CONSTRAINT, (
            f"{name}={value} primary-byte differs from "
            f"SQLITE_CONSTRAINT={dqlitewire.SQLITE_CONSTRAINT}"
        )


# ---- merged from test_tx_auto_rollback_codes_negative.py ----
# Negative pin: codes upstream emits via ``failure(req, ...)`` that
# must NOT trigger the client's auto-rollback tracker clear.
#
# The positive set (ABORT/NOMEM/INTERRUPT/IOERR/CORRUPT/FULL) is
# already pinned. This file pins the negative space — adding a code
# to the auto-rollback set without flipping the parametrize here will
# trip the test, forcing a deliberate review.


@pytest.mark.parametrize(
    "code",
    [
        SQLITE_ERROR,  # gateway.c emits for nonempty statement tail, etc.
        SQLITE_BUSY,  # engine-side OR Raft-side; carved out at client layer
        SQLITE_NOTFOUND,  # gateway.c LOOKUP_DB / LOOKUP_STMT
        SQLITE_PROTOCOL,  # bad format version
        19,  # SQLITE_CONSTRAINT — CHECK on plain INSERT does NOT auto-rollback
        21,  # SQLITE_MISUSE
        DQLITE_NOTFOUND,
        DQLITE_PARSE,
    ],
)
def test_code_not_in_auto_rollback_set(code: int) -> None:
    """Pin: the listed codes are NOT in TX_AUTO_ROLLBACK_PRIMARY_CODES.
    A maintainer adding a code to the auto-rollback set must
    explicitly remove it from this matrix."""
    assert code not in TX_AUTO_ROLLBACK_PRIMARY_CODES


# ---- merged from test_wire_decode_failed_prefix_constant_load_bearing.py ----
# Pin: ``WIRE_DECODE_FAILED_PREFIX`` is the canonical phrase the
# disconnect-classification chain (wire → client.ProtocolError →
# dbapi.OperationalError(code=None) → SA's
# ``_dqlite_disconnect_messages`` substring scan) depends on.
#
# The constant exists as the single source of truth so a rename
# ripples through grep. This test pins the canonical value so a future
# rename has to update the constant deliberately.


def test_wire_decode_failed_prefix_value_is_canonical() -> None:
    """The exact lowercase phrase ``"wire decode failed"``. SA's
    substring scan lowercases the rendered exception text before
    matching, so emission sites can use any casing — but the
    constant's value is the canonical lowercase form that SA looks
    for."""
    assert WIRE_DECODE_FAILED_PREFIX == "wire decode failed"


def test_wire_decode_failed_prefix_is_str_final() -> None:
    """Typed ``Final[str]`` so a mypy-checked rebind is a typing
    error, mirroring the rest of the wire-constant family."""
    assert isinstance(WIRE_DECODE_FAILED_PREFIX, str)
    # Module-level Final[str] — re-import asserts identity.
    from dqlitewire import WIRE_DECODE_FAILED_PREFIX as again

    assert again is WIRE_DECODE_FAILED_PREFIX
