# Turn floor, first receipt (2026-11-21)

One request, measured for stage 0 of `docs/PREFILL_ANTICIPATION.md` with
`tools/turn_floor.py`. The run was planned over seven models. Not measured:
the fleet was unavailable after this request.

## Configuration

- Release c2b714343 on sixteen Sparks: the production lane of `qwen3.8-max`
  (adapter `qwen38_max-tp16`, driver artifact
  `fa25f65112363e2dd6b1a62628e2c49c4183840ff752d588500650da8b540df9`),
  active for 22 s when the run began, 16 of 16 ranks connected.
- The driver ran on the hub against the engine API with token ids, no chat
  layer, speculation off, streaming.
- The prompt is 20,000 token ids of this repository's text through the
  model's tokenizer, never sent before, so nothing of it was cached.

## Files

| File | What it is |
| --- | --- |
| `qwen3.8-max/requests.jsonl` | the run's start mark and the request, as the driver wrote them |
| `qwen3.8-max/engine.log` | the engine API's log from the start mark, byte 120,651,989 of the lane's log, to the request's `request_measurements` line |
| `qwen3.8-max/parts.jsonl`, `qwen3.8-max/summary.json` | the split of the request's first token |

The driver on the hub was an earlier copy of the tool (the mark's
`tool_sha256`). The split is this branch's:

```sh
python3 tools/turn_floor.py parts qualification/turn-floor-20261121/qwen3.8-max \
    --api-log qualification/turn-floor-20261121/qwen3.8-max/engine.log --log-base 120651989
```

## What it shows

Request 100002, 20,000 tokens cold:

| | |
| --- | ---: |
| First token at the client | 31,830 ms |
| Send | 1.6 ms |
| Queue | 0.6 ms |
| Restore | none |
| Prefill: ten waves, nine of 2,048 rows and one of 1,567 | 31,726 ms |
| Service of a 2,048-row wave | 3,101 to 3,226 ms (635 to 660 tok/s) |
| Between the waves, in all | 3.2 ms |
| The last prompt token's wave, one row | 101 ms (92 ms of service) |
| Reply leg, engine to client | 0.3 ms |
| Decode after the first token | 12.0 tok/s (15 tokens in chains of 8 and 7) |

The log also holds request 100003, which another client sent 2.4 s after
the first was accepted: 5 prompt tokens, 8 output tokens, interactive like
the first, not streamed. Only the engine's side of it is known.

| Request 100003 | Service | Wall |
| --- | ---: | ---: |
| Its prefill wave, 5 rows | 155 ms | 975 ms |
| Its chain of 7 tokens | 570 ms | 3,751 ms |
| First token after acceptance | | 975 ms |
| Last token after acceptance | | 4,726 ms |
| Request closed after acceptance (`request_done`) | | 11,093 ms |

Each of its steps waited for the 2,048-row wave of the other request that
was in flight, about 3.2 s a wave. The same 725 ms of its service shows in
the long request: two of its waves returned 156 and 569 ms late. The log has
no line that says what request 100003 waited for between its last token and
its close; two more waves of the long request ran in that time.
