"""Every Request/Response type has an explicit entry in the schema-
version table OR is documented as schema-0-only.

``codec.py`` defines ``_REQUEST_MAX_SCHEMA`` and
``_RESPONSE_MAX_SCHEMA`` mapping each known message code to the
maximum supported schema version. Codes not in the map default to
``max_schema=0`` — meaning the decoder rejects schema=1 frames the
server might send for a future request type.

Pin the contract: every value of ``RequestType`` / ``ResponseType``
must either appear in the corresponding ``_*_MAX_SCHEMA`` dict OR
appear in the explicit "schema-0-only" allow-list below. A future
maintainer who adds a new message code without updating one of the
two structures will fail this test loudly — exactly the scenario
that's currently invisible to the suite.

The schema-0-only allow-lists are authored against the C reference
at ``dqlite-upstream/src/`` (request.h, command.h) — every entry has
an authority citation by being declared there with no schema-1
variant.
"""

from __future__ import annotations

from dqlitewire.codec import _REQUEST_MAX_SCHEMA, _RESPONSE_MAX_SCHEMA
from dqlitewire.constants import RequestType, ResponseType

# Request types that are deliberately schema-0-only (no schema-1
# variant in the C reference). If a future schema-1 variant is added
# upstream, move the entry from here to ``_REQUEST_MAX_SCHEMA`` and
# the test will continue to pass.
_REQUEST_SCHEMA_ZERO_ONLY: frozenset[int] = frozenset(
    {
        RequestType.LEADER,
        RequestType.CLIENT,
        RequestType.HEARTBEAT,
        RequestType.OPEN,
        RequestType.FINALIZE,
        RequestType.INTERRUPT,
        RequestType.CONNECT,
        RequestType.ADD,
        RequestType.ASSIGN,
        RequestType.REMOVE,
        RequestType.DUMP,
        RequestType.CLUSTER,
        RequestType.TRANSFER,
        RequestType.DESCRIBE,
        RequestType.WEIGHT,
    }
)

# Response types that are deliberately schema-0-only.
_RESPONSE_SCHEMA_ZERO_ONLY: frozenset[int] = frozenset(
    {
        ResponseType.FAILURE,
        ResponseType.LEADER,
        ResponseType.WELCOME,
        ResponseType.SERVERS,
        ResponseType.DB,
        ResponseType.RESULT,
        ResponseType.ROWS,
        ResponseType.EMPTY,
        ResponseType.FILES,
        ResponseType.METADATA,
    }
)


def test_every_request_type_has_schema_entry_or_explicit_zero() -> None:
    missing = []
    for req in RequestType:
        if req in _REQUEST_MAX_SCHEMA:
            continue
        if req in _REQUEST_SCHEMA_ZERO_ONLY:
            continue
        missing.append(req)
    assert not missing, (
        f"RequestType values missing from both _REQUEST_MAX_SCHEMA and the "
        f"schema-0-only allow-list: {missing!r}. Add the entry to "
        f"_REQUEST_MAX_SCHEMA in codec.py if a schema-1 variant exists "
        f"upstream, or extend _REQUEST_SCHEMA_ZERO_ONLY in this test file "
        f"with the authority citation."
    )


def test_every_response_type_has_schema_entry_or_explicit_zero() -> None:
    missing = []
    for resp in ResponseType:
        if resp in _RESPONSE_MAX_SCHEMA:
            continue
        if resp in _RESPONSE_SCHEMA_ZERO_ONLY:
            continue
        missing.append(resp)
    assert not missing, (
        f"ResponseType values missing from both _RESPONSE_MAX_SCHEMA and the "
        f"schema-0-only allow-list: {missing!r}. Add the entry to "
        f"_RESPONSE_MAX_SCHEMA in codec.py if a schema-1 variant exists "
        f"upstream, or extend _RESPONSE_SCHEMA_ZERO_ONLY in this test file "
        f"with the authority citation."
    )


def test_request_max_schema_does_not_overlap_zero_only_allowlist() -> None:
    """Sanity: a code listed as schema-0-only must NOT also appear in
    ``_REQUEST_MAX_SCHEMA`` with a value > 0. The split lists are
    intentionally disjoint."""
    overlap = set(_REQUEST_MAX_SCHEMA) & _REQUEST_SCHEMA_ZERO_ONLY
    assert not overlap, (
        f"RequestType codes appear in both _REQUEST_MAX_SCHEMA and the "
        f"schema-0-only allow-list: {overlap!r}. Pick one."
    )


def test_response_max_schema_does_not_overlap_zero_only_allowlist() -> None:
    overlap = set(_RESPONSE_MAX_SCHEMA) & _RESPONSE_SCHEMA_ZERO_ONLY
    assert not overlap, (
        f"ResponseType codes appear in both _RESPONSE_MAX_SCHEMA and the "
        f"schema-0-only allow-list: {overlap!r}. Pick one."
    )
