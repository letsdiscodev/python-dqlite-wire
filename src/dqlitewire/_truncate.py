"""Back-compat shim — the helper has been promoted to the public
:mod:`dqlitewire.truncate` module. The public names
:data:`dqlitewire.DEFAULT_MAX_RAW_MESSAGE` and
:func:`dqlitewire.cap_raw_message` are the canonical surface.

This shim re-exports the underscore-prefixed pre-promotion names for
one release cycle so downstream consumers can migrate without
lockstep. New code should import from :mod:`dqlitewire` directly.
"""

from dqlitewire.truncate import (
    DEFAULT_MAX_RAW_MESSAGE as _DEFAULT_MAX_RAW_MESSAGE,
)
from dqlitewire.truncate import (
    cap_raw_message as _cap_raw_message,
)

__all__ = ["_DEFAULT_MAX_RAW_MESSAGE", "_cap_raw_message"]
