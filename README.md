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

## Protocol Reference

Based on the [dqlite wire protocol specification](https://canonical.com/dqlite/docs/reference/wire-protocol).

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup and contribution guidelines.

## License

MIT
