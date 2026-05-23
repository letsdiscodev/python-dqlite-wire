"""Tests for protocol constants matching Go reference implementation."""

import pytest

from dqlitewire.constants import RequestType, ResponseType


class TestResponseTypeValues:
    """Verify ResponseType enum matches go-dqlite constants.go.

    Reference: github.com/canonical/go-dqlite internal/protocol/constants.go
    """

    def test_failure_is_0(self) -> None:
        assert ResponseType.FAILURE == 0

    def test_node_is_1(self) -> None:
        assert ResponseType.LEADER == 1

    def test_welcome_is_2(self) -> None:
        assert ResponseType.WELCOME == 2

    def test_nodes_is_3(self) -> None:
        """Go: ResponseNodes = 3 (cluster server listing)."""
        assert ResponseType.SERVERS == 3

    def test_db_is_4(self) -> None:
        assert ResponseType.DB == 4

    def test_stmt_is_5(self) -> None:
        assert ResponseType.STMT == 5

    def test_result_is_6(self) -> None:
        assert ResponseType.RESULT == 6

    def test_rows_is_7(self) -> None:
        assert ResponseType.ROWS == 7

    def test_empty_is_8(self) -> None:
        assert ResponseType.EMPTY == 8

    def test_files_is_9(self) -> None:
        assert ResponseType.FILES == 9

    def test_metadata_is_10(self) -> None:
        """Go: ResponseMetadata = 10."""
        assert ResponseType.METADATA == 10

    def test_no_node_legacy_as_separate_type(self) -> None:
        """NODE_LEGACY should not exist as a separate type code.

        In Go, ResponseNodeLegacy = 1 (same as ResponseNode).
        """
        assert not hasattr(ResponseType, "NODE_LEGACY")


class TestRequestTypeValues:
    """Verify RequestType enum matches go-dqlite constants.go."""

    def test_leader_is_0(self) -> None:
        assert RequestType.LEADER == 0

    def test_client_is_1(self) -> None:
        assert RequestType.CLIENT == 1

    def test_open_is_3(self) -> None:
        assert RequestType.OPEN == 3

    def test_prepare_is_4(self) -> None:
        assert RequestType.PREPARE == 4

    def test_exec_is_5(self) -> None:
        assert RequestType.EXEC == 5

    def test_query_is_6(self) -> None:
        assert RequestType.QUERY == 6

    def test_finalize_is_7(self) -> None:
        assert RequestType.FINALIZE == 7

    def test_exec_sql_is_8(self) -> None:
        assert RequestType.EXEC_SQL == 8

    def test_query_sql_is_9(self) -> None:
        assert RequestType.QUERY_SQL == 9

    def test_interrupt_is_10(self) -> None:
        assert RequestType.INTERRUPT == 10

    def test_add_is_12(self) -> None:
        assert RequestType.ADD == 12

    def test_assign_is_13(self) -> None:
        assert RequestType.ASSIGN == 13

    def test_remove_is_14(self) -> None:
        assert RequestType.REMOVE == 14

    def test_dump_is_15(self) -> None:
        assert RequestType.DUMP == 15

    def test_cluster_is_16(self) -> None:
        assert RequestType.CLUSTER == 16

    def test_transfer_is_17(self) -> None:
        assert RequestType.TRANSFER == 17

    def test_describe_is_18(self) -> None:
        assert RequestType.DESCRIBE == 18

    def test_weight_is_19(self) -> None:
        assert RequestType.WEIGHT == 19

    def test_connect_is_11(self) -> None:
        """C protocol defines DQLITE_REQUEST_CONNECT = 11 for Raft transport connections.

        The Go client omits this (it's a client library, not a cluster node),
        but the C server defines it and a complete protocol implementation
        should include it.
        """
        assert RequestType.CONNECT == 11


class TestPublicExports:
    """Verify important types are importable from public API paths."""

    def test_nodeinfo_importable_from_messages(self) -> None:
        """NodeInfo should be importable from dqlitewire.messages."""
        from dqlitewire.messages import NodeInfo

        assert NodeInfo is not None
        # Verify it's the same class used by ServersResponse
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
        # Both modern (40/41) and legacy (32/33) sub-code variants are
        # in the set so a leader-flip on a pre-3.32.1 dqlite server is
        # classified as retryable. See companion test
        # ``test_leader_error_codes_legacy_subcodes.py`` for the Go
        # parity rationale.
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
        # Both modern and legacy variants are in the set; the latter
        # cover pre-3.32.1 dqlite servers that emit (32/33<<8) sub-
        # codes for the same leader-flip conditions. See companion
        # test ``test_leader_error_codes_legacy_subcodes.py``.
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
        """Pin extraction for the auto-rollback primary set so a future
        codec refactor can't silently mishandle one of them."""
        from dqlitewire.constants import primary_sqlite_code

        assert primary_sqlite_code(extended) == primary


class TestAutoRollbackPrimaryCodes:
    """Pin the canonical home for the auto-rollback primary code set
    that the downstream client tracker uses to clear ``_in_transaction``
    after a server-side auto-rollback."""

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
        """SQLITE_NOMEM (7) is on SQLite's documented auto-rollback list
        (https://www.sqlite.org/lang_transaction.html). A server-side
        OOM during step() causes SQLite to tear down the txn; the
        client must mirror that by clearing tracker state."""
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
    """Pin the SQLite primary codes that the SA dialect treats as
    slot-fatal via the bare-``DatabaseError`` arm of ``is_disconnect``
    and ``do_ping`` — codes 11 / 24 / 26 (CORRUPT / FORMAT / NOTADB).
    Hosted in the wire layer alongside the other SQLite primaries so
    every consumer (SA dialect, dbapi cursor, future tooling) imports
    one source of truth."""

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
    """``dqlite-upstream/include/dqlite.h`` defines ``DQLITE_ERROR = 1``,
    which shares the wire low-byte with ``SQLITE_ERROR = 1``. Upstream
    emits ``DQLITE_ERROR`` from ``gateway.c::handle_request_transfer``
    (``failure(req, DQLITE_ERROR, "leadership transfer failed")``) on the
    ``REQUEST_TRANSFER`` path. The Python client does NOT currently
    invoke that request type, so the collision is latent — but the
    architectural shape relies on message-text disambiguation in
    ``dbapi.connection._is_no_transaction_error``.

    Pin the collision so a future audit cycle finds the breadcrumbs
    and a future maintainer adding ``REQUEST_TRANSFER`` support sees the
    risk."""

    def test_dqlite_error_value_one_shares_low_byte_with_sqlite_error(self) -> None:
        from dqlitewire.constants import primary_sqlite_code

        # ``DQLITE_ERROR = 1`` is NOT exported as a named constant —
        # exporting it under the same value as a SQLite primary would
        # invite a use-site to import the wrong one. Inline the literal
        # here so the collision is documented at exactly the place a
        # maintainer would look to disambiguate.
        DQLITE_ERROR_LITERAL = 1
        SQLITE_ERROR_LITERAL = 1
        assert DQLITE_ERROR_LITERAL == SQLITE_ERROR_LITERAL
        assert primary_sqlite_code(DQLITE_ERROR_LITERAL) == 1

    def test_dqlite_error_is_not_classified_as_namespace_code(self) -> None:
        """``is_dqlite_namespace_code`` covers only codes >= 1000
        that pass through ``primary_sqlite_code`` unchanged.
        ``DQLITE_ERROR=1`` collides numerically with ``SQLITE_ERROR=1``
        but must NOT be classified as a namespace code — that would
        mask the SQLite primary from dispatch. Pin that 1, 2, 3 are
        outside the namespace range so a regression that widens the
        range or repurposes the discriminator surfaces here."""
        from dqlitewire.constants import is_dqlite_namespace_code

        assert is_dqlite_namespace_code(1) is False
        assert is_dqlite_namespace_code(2) is False  # DQLITE_MISUSE collides
        assert is_dqlite_namespace_code(3) is False  # DQLITE_NOMEM collides


class TestDefaultRowsAndFramesCaps:
    """The default per-query caps (row count and continuation-frame
    count) are operational defenses against slow-drip / amplification
    attacks from a hostile or buggy server. Pin the literal values so
    a refactor that "factors out" the constants (e.g. divides by
    1000 by accident) cannot silently change the magnitude."""

    def test_default_max_total_rows_is_ten_million(self) -> None:
        from dqlitewire import DEFAULT_MAX_TOTAL_ROWS

        assert DEFAULT_MAX_TOTAL_ROWS == 10_000_000

    def test_default_max_continuation_frames_is_one_hundred_thousand(self) -> None:
        from dqlitewire import DEFAULT_MAX_CONTINUATION_FRAMES

        assert DEFAULT_MAX_CONTINUATION_FRAMES == 100_000


class TestInternalDecodeBoundsPinnedAgainstReadme:
    """The README's "Internal sanity bounds" section enumerates four
    decode-time cap constants. Pin each value so a future refactor
    that bumps a constant without updating the README produces a test
    failure — exactly the drift this test catches today
    (``_MAX_COLUMN_COUNT`` was bumped from 10,000 to 32,767 to match
    SQLite's ``SQLITE_MAX_COLUMN`` ceiling, but the README was not
    updated)."""

    def test_max_param_count_matches_sqlite_max_variable_number(self) -> None:
        from dqlitewire.tuples import _MAX_PARAM_COUNT

        # SQLite's standard-build compile-time max for
        # SQLITE_MAX_VARIABLE_NUMBER (https://www.sqlite.org/limits.html).
        # The server cannot bind beyond this; tightening the client
        # cap to match prevents a malicious peer from inflating
        # intermediate allocations.
        assert _MAX_PARAM_COUNT == 32_766

    def test_max_column_count_matches_sqlite_default(self) -> None:
        from dqlitewire.messages.responses import _MAX_COLUMN_COUNT

        # SQLite's compile-time ``SQLITE_MAX_COLUMN`` default is 2000.
        # The C server's ``query.c:111-120`` emits the column count
        # as ``uint64`` without cap; ``STMT__MAX_COLUMNS`` in
        # ``stmt.c:10`` is defined but never referenced, so the
        # Python cap is defence-in-depth rather than a wire-protocol
        # mirror.
        assert _MAX_COLUMN_COUNT == 2000

    def test_max_file_count_is_one_hundred(self) -> None:
        from dqlitewire.messages.responses import _MAX_FILE_COUNT

        assert _MAX_FILE_COUNT == 100

    def test_max_node_count_is_ten_thousand(self) -> None:
        from dqlitewire.messages.responses import _MAX_NODE_COUNT

        assert _MAX_NODE_COUNT == 10_000


class TestNamedPrimaryCodeValues:
    """Pin each named SQLite/dqlite primary result code at its
    upstream-spec value.

    The named constants are public re-exports on ``dqlitewire``
    (``SQLITE_BUSY`` etc.) and the wire's ``FailureResponse`` round-
    trip matrix exercises every one of them. The matrix uses the
    constant on both encode and decode sides, so a maintainer who
    silently changes the value (rebase swap, accidental edit) does
    not break the round-trip — the matrix is internally consistent
    even when the value has drifted from the spec. This pin closes
    that gap.

    Reference values:
      - SQLite primary codes: https://www.sqlite.org/rescode.html
      - DQLITE_PROTO=1001    : dqlite-upstream/src/protocol.h:9
      - DQLITE_NOTFOUND=1002 : dqlite-upstream/src/lib/registry.h:13
      - DQLITE_PARSE=1005    : dqlite-upstream/src/lib/serialize.h:14
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
