"""Pin: the legacy LeaderResponse dispatch in ``MessageDecoder``
gates on ``not self._is_request`` (the direction guard) rather than
on the structural-coincidence ``msg_class is LeaderResponse`` check.

``RequestType.CLIENT == 1`` collides numerically with
``ResponseType.LEADER == 1``. The identity check on ``msg_class``
incidentally provides the direction discrimination today, but a
future alias/subclass registration in ``RESPONSE_TYPES`` would
break the implicit coupling. The direction guard is the load-
bearing predicate.
"""

from __future__ import annotations

import inspect

from dqlitewire.codec import MessageDecoder


def test_legacy_leader_dispatch_gates_on_direction_not_identity() -> None:
    src = inspect.getsource(MessageDecoder.decode_bytes)
    assert "not self._is_request" in src, (
        "Legacy LeaderResponse dispatch must gate on the direction "
        "guard ``not self._is_request``, not solely on the identity "
        "coincidence ``msg_class is LeaderResponse``."
    )
