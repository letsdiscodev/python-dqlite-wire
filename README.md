# dqlite-wire

Pure Python wire protocol implementation for [dqlite](https://dqlite.io/), Canonical's distributed SQLite.

## Installation

```bash
pip install dqlite-wire
```

## Usage

```python
from dqlitewire import encode_message, decode_message
from dqlitewire.messages import LeaderRequest

# Encode a message
data = encode_message(LeaderRequest())

# Decode a message
message = decode_message(data, is_request=True)
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

## Deliberate divergences from upstream

This library implements the dqlite wire protocol faithfully but adds a
handful of defensive guards that the upstream C server and the
canonical [go-dqlite](https://github.com/canonical/go-dqlite) client do
not. They protect a Python client running in potentially adversarial
network contexts and are all opt-out-able.

**Python-specific caps** (not present in C or Go; `None` disables):

- `DEFAULT_MAX_ROWS` (`MessageDecoder(max_rows=...)`, default 1,000,000)
  — per-query row cap.
- `DEFAULT_MAX_MESSAGE_SIZE` (`ReadBuffer(max_message_size=...)`, default
  64 MiB) — envelope cap on a single frame.
- `_MAX_PARAM_COUNT` (100,000), `_MAX_COLUMN_COUNT` (32,767 — matches
  SQLite's `SQLITE_MAX_COLUMN`), `_MAX_FILE_COUNT` (100),
  `_MAX_NODE_COUNT` (10,000) — internal sanity bounds on decoded
  tuple / response sizes.

**Stricter-than-Go validations** (match the C server's intent):

- `decode_row_header` requires the full 8-byte marker (C defines
  `DQLITE_RESPONSE_ROWS_DONE = 0xff..ff` / `_PART = 0xee..ee`;
  go-dqlite checks only the first byte).
- `encode_value(value, ValueType.BOOLEAN)` rejects arbitrary ints
  (accepts only `bool` or exactly `0`/`1`).
- `FilesResponse.encode_body` rejects non-8-aligned file content (C's
  `dumpFile` asserts `len % 8 == 0`).
- `encode_params_tuple` rejects `ValueType.UNIXTIME` outbound (C's
  `tuple_decoder__next` cannot decode it on the server side).
- `StmtResponse` rejects a 16-byte body when `schema=1` (C's V1
  response is 24 bytes).

**Not implemented** (valid upstream formats this library chose to skip):

- `ClusterRequest` `format=0` — V0 response with id+address only. We
  only decode the V1 variant that includes node role.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup and contribution guidelines.

## License

MIT
