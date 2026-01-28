"""Exceptions for dqlite wire protocol."""


class ProtocolError(Exception):
    """Base exception for protocol errors."""

    pass


class EncodeError(ProtocolError):
    """Error during message encoding."""

    pass


class DecodeError(ProtocolError):
    """Error during message decoding."""

    pass
