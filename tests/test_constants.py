"""Tests for protocol constants matching Go reference implementation."""

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


class TestTypeDictCompleteness:
    """Verify REQUEST_TYPES and RESPONSE_TYPES cover all enum members."""

    def test_request_types_covers_all_enum_members(self) -> None:
        from dqlitewire.codec import REQUEST_TYPES

        for member in RequestType:
            assert member.value in REQUEST_TYPES, (
                f"RequestType.{member.name} ({member.value}) has no entry in REQUEST_TYPES"
            )

    def test_response_types_covers_all_enum_members(self) -> None:
        from dqlitewire.codec import RESPONSE_TYPES

        for member in ResponseType:
            assert member.value in RESPONSE_TYPES, (
                f"ResponseType.{member.name} ({member.value}) has no entry in RESPONSE_TYPES"
            )
