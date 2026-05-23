"""Pin: the underscore-prefixed pre-promotion names continue to work
for one release cycle so downstream consumers can migrate at their
own pace.

``_DEFAULT_MAX_RAW_MESSAGE`` and ``_cap_raw_message`` are kept as
back-compat aliases for the now-public ``DEFAULT_MAX_RAW_MESSAGE``
and ``cap_raw_message``. The aliases will be removed when the two
downstream consumers (``dqliteclient.exceptions`` and
``dqlitedbapi.exceptions``) have migrated.
"""

from __future__ import annotations

from dqlitewire import DEFAULT_MAX_RAW_MESSAGE, cap_raw_message
from dqlitewire._truncate import _DEFAULT_MAX_RAW_MESSAGE, _cap_raw_message


def test_value_alias_points_at_public_constant() -> None:
    assert _DEFAULT_MAX_RAW_MESSAGE == DEFAULT_MAX_RAW_MESSAGE


def test_function_alias_is_public_function() -> None:
    assert _cap_raw_message is cap_raw_message
