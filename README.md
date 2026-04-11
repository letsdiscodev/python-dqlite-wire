# dqlite-wire

Pure Python wire protocol implementation for [dqlite](https://dqlite.io/), Canonical's distributed SQLite.

## Installation

```bash
pip install dqlite-wire
```

## Usage

```python
from dqlitewire import MessageEncoder, MessageDecoder
from dqlitewire.messages import LeaderRequest, ClientRequest

# Encode a message
encoder = MessageEncoder()
data = encoder.encode(LeaderRequest())

# Decode a message
decoder = MessageDecoder()
decoder.feed(data)
message = decoder.decode()
```

## Thread-safety

`ReadBuffer`, `WriteBuffer`, `MessageEncoder`, and `MessageDecoder`
are **not thread-safe**. Each instance must be owned by a single
thread or a single asyncio coroutine at a time. This matches Go's
`driver.Conn` contract from go-dqlite.

Concurrent use of a single instance from multiple threads produces
**silent data corruption** — not exceptions. The `is_poisoned`
mechanism catches torn state from signal delivery during
single-owner execution, but it **cannot** detect lost-update races
between concurrent threads. Fuzz testing reliably reproduces both
duplicate message delivery and corrupt (garbage) message bytes with
no exception surfacing.

If you need concurrent access, wrap every call site in an
`asyncio.Lock` or `threading.Lock` at the layer that owns the
socket and decoder.

## Protocol Reference

Based on the [dqlite wire protocol specification](https://canonical.com/dqlite/docs/reference/wire-protocol).

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup and contribution guidelines.

## License

MIT
