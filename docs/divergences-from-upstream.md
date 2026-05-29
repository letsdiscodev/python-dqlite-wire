# Divergences from upstream

This library implements the dqlite wire protocol faithfully, but adds a
handful of defensive guards that the upstream C server and the
[go-dqlite](https://github.com/canonical/go-dqlite) client do not. They
protect a Python client running in potentially adversarial network
contexts. The caps are configurable (pass `None` to disable); the stricter
validations match the C server's intent.

## Python-specific caps

Bounds on how much a single decode will allocate, so a hostile or buggy
peer cannot exhaust memory. All are optional (`None` disables):

- `DEFAULT_MAX_TOTAL_ROWS` (`MessageDecoder(max_total_rows=...)`, default
  10,000,000) — rows accumulated across continuation frames for one query.
- `DEFAULT_MAX_CONTINUATION_FRAMES`
  (`MessageDecoder(max_continuation_frames=...)`, default 100,000) —
  continuation frames for one query.
- `RowsResponse.DEFAULT_MAX_ROWS` (`MessageDecoder(max_rows=...)`, default
  1,000,000) — per-frame row cap.
- `ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE` (`ReadBuffer(max_message_size=...)`,
  default 64 MiB) — envelope cap on a single frame.
- Internal sanity bounds on decoded tuple/response sizes:
  `_MAX_PARAM_COUNT` (32,766 — SQLite's `SQLITE_MAX_VARIABLE_NUMBER`),
  `_MAX_COLUMN_COUNT` (2000 — SQLite's default `SQLITE_MAX_COLUMN`),
  `_MAX_FILE_COUNT` (100), `_MAX_NODE_COUNT` (10,000).

`DEFAULT_MAX_TOTAL_ROWS` and `DEFAULT_MAX_CONTINUATION_FRAMES` are
importable from `dqlitewire`; the others are class-scoped.

## Stricter-than-Go validations

These match the C server's intent more closely than go-dqlite does:

- `decode_row_header` requires the full 8-byte end-of-rows marker (C defines
  `DQLITE_RESPONSE_ROWS_DONE = 0xff…ff` / `_PART = 0xee…ee`; go-dqlite checks
  only the first byte).
- `encode_value(value, ValueType.BOOLEAN)` rejects arbitrary ints (accepts
  only `bool` or exactly `0`/`1`).
- `FilesResponse.encode_body` rejects non-8-aligned file content (C's
  `dumpFile` asserts `len % 8 == 0`).
- `encode_params_tuple` rejects `ValueType.UNIXTIME` outbound (the C server's
  tuple decoder cannot decode it).
- `StmtResponse` rejects a 16-byte body when `schema=1` (C's V1 response is
  24 bytes).

## Asymmetric encode/decode

Some frames are decodable for proxy / recorded-traffic round-trips even
though fresh construction of them is rejected:

- `ClusterRequest` with `format=0` (the V0 response shape: id + address
  only). `ClusterRequest.decode_body` decodes it for proxy/replay use, and a
  *decoded* V0 frame re-encodes byte-identically. *Fresh* construction with
  `format=0` is rejected with `EncodeError`, because production senders
  always emit V1 (id + address + role) and this client only decodes the V1
  `ServersResponse`.
