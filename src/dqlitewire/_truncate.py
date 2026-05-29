"""Back-compat shim re-exporting the pre-promotion underscore names for one
release cycle. New code should import from :mod:`dqlitewire` directly.
"""

from dqlitewire.truncate import (
    DEFAULT_MAX_RAW_MESSAGE as _DEFAULT_MAX_RAW_MESSAGE,
)
from dqlitewire.truncate import (
    cap_raw_message as _cap_raw_message,
)

__all__ = ["_DEFAULT_MAX_RAW_MESSAGE", "_cap_raw_message"]
