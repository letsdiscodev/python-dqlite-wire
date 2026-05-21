"""``encode_blob`` / ``decode_blob`` docstrings must cite the actual
``_MAX_BLOB_SIZE`` value (or the "minus framing" qualifier), not the
rounded "64 MiB" figure that drifts 64 bytes from the constant.

The constant sits 64 bytes below ``DEFAULT_MAX_MESSAGE_SIZE`` so a
blob at the documented cap round-trips through the default decoder
(see ``test_blob_at_cap_round_trips_in_default_envelope.py`` for the
arithmetic pin). A docstring claiming the default is exactly "64 MiB"
sends a caller who builds an integration test against the documented
cap into a sharp ``EncodeError`` at the 64-byte shortfall — the
operator-facing trap this pin guards against.
"""

from __future__ import annotations

from dqlitewire.types import _MAX_BLOB_SIZE, decode_blob, encode_blob


def _qualifier_present(text: str) -> bool:
    """The docstring is considered honest if it cites the exact byte
    count, the "minus framing" qualifier, or the "64 MiB - 64" form.
    Bare "64 MiB" alone is the drift this test guards against."""
    return (
        str(_MAX_BLOB_SIZE) in text
        or "67_108_800" in text
        or "minus framing" in text
        or "64 MiB - 64" in text
    )


def test_encode_blob_docstring_cites_actual_cap() -> None:
    doc = encode_blob.__doc__ or ""
    assert _qualifier_present(doc), (
        f"encode_blob docstring must cite the actual _MAX_BLOB_SIZE "
        f"value ({_MAX_BLOB_SIZE}) or a 'minus framing' qualifier — "
        f"bare '64 MiB' drifts 64 bytes from the constant. Docstring:\n{doc}"
    )


def test_decode_blob_docstring_cites_actual_cap() -> None:
    doc = decode_blob.__doc__ or ""
    assert _qualifier_present(doc), (
        f"decode_blob docstring must cite the actual _MAX_BLOB_SIZE "
        f"value ({_MAX_BLOB_SIZE}) or a 'minus framing' qualifier — "
        f"bare '64 MiB' drifts 64 bytes from the constant. Docstring:\n{doc}"
    )


def test_blob_at_documented_cap_actually_encodes() -> None:
    """The number the docstring cites must actually round-trip — guards
    against future drift if either side of the constant/docstring pair
    is edited without the other."""
    encoded = encode_blob(b"\x00" * _MAX_BLOB_SIZE)
    assert isinstance(encoded, bytes)
    decoded, _ = decode_blob(encoded)
    assert len(decoded) == _MAX_BLOB_SIZE
