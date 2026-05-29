# Thread-safety

`ReadBuffer`, `WriteBuffer`, `MessageEncoder`, and `MessageDecoder` are
**not thread-safe**. Each instance must be owned by a single thread or a
single asyncio coroutine at a time. This matches the single-owner
`driver.Conn` contract from [go-dqlite](https://github.com/canonical/go-dqlite).

Concurrent use of one instance from multiple threads produces **silent
data corruption — not exceptions.** Fuzz testing reliably reproduces both
duplicate message delivery and corrupt (garbage) message bytes with no
error raised. The internal `is_poisoned` guard catches torn state from a
signal delivered during single-owner execution, but it **cannot** detect
lost-update races between concurrent threads.

If you need concurrent access, serialize it with an `asyncio.Lock` or
`threading.Lock` at the layer that owns the socket and decoder. In
practice the higher-level packages
([dqlite-client](https://github.com/letsdiscodev/python-dqlite-client) and
above) already do this for you — this constraint matters only if you drive
the codec directly.
