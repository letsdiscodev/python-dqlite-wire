# dqlite-wire

Pure-Python codec for the [dqlite](https://dqlite.io/) wire protocol —
encode and decode the messages a dqlite server speaks.

[dqlite](https://dqlite.io/) is Canonical's distributed SQLite, built on
Raft. This package implements the bytes-on-the-wire layer: it turns
request/response objects into frames and back, following the
[official wire-protocol specification](https://canonical.com/dqlite/docs/reference/wire-protocol).
It does no networking, pooling, or SQL — just framing.

## Is this the package you want?

Probably not, unless you are building a driver or doing wire-level work
(a proxy, traffic capture/replay, a custom client). If you just want to
run SQL against dqlite from Python, use one of the higher layers — see
[The dqlite Python stack](#the-dqlite-python-stack) below.

## Installation

```bash
pip install dqlite-wire
```

Requires Python 3.13+.

## Usage

```python
from dqlitewire import encode_message, decode_message
from dqlitewire.messages import LeaderRequest

# Encode a request to bytes
data = encode_message(LeaderRequest())

# Decode bytes back into a message object
message = decode_message(data, is_request=True)
```

## The dqlite Python stack

This is the lowest of four layered packages. Each builds on the one below:

| Package | Role |
| --- | --- |
| [sqlalchemy-dqlite](https://github.com/letsdiscodev/sqlalchemy-dqlite) | SQLAlchemy 2.0 dialect |
| [dqlite-dbapi](https://github.com/letsdiscodev/python-dqlite-dbapi) | PEP 249 (DB-API 2.0) driver — sync & async |
| [dqlite-client](https://github.com/letsdiscodev/python-dqlite-client) | Async wire client — pooling, leader discovery |
| **dqlite-wire** — this package | Wire-protocol codec |

**Most applications should use [dqlite-dbapi](https://github.com/letsdiscodev/python-dqlite-dbapi)
or [sqlalchemy-dqlite](https://github.com/letsdiscodev/sqlalchemy-dqlite).**

## Documentation

- [Thread-safety](docs/thread-safety.md) — codec objects are single-owner; read this before sharing one.
- [Divergences from upstream](docs/divergences-from-upstream.md) — the
  defensive caps and stricter validations this codec adds on top of the C
  server and [go-dqlite](https://github.com/canonical/go-dqlite).

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup and contribution guidelines.

## License

MIT
