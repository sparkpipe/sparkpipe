# One rank on one GB10 (2026-11-22)

Two runs of the stage 0 tools against a real rank: Qwen 3.8 27B FP8 on one
GB10, engine API and rank on the same machine. Nothing here stands for a
sixteen-rank lane.

1. **The turn floor**, with the GPU alone
   (`qwen3.8-27b-fp8-one-gb10-alone/`).
2. **The instruments**, earlier the same day with the GPU shared with
   another service (`qwen3.8-27b-fp8-one-gb10/`): a check that they tell the
   truth on a real rank, not a floor.

## The turn floor

One GB10, one rank, GPU alone.

Node files built from `ac1ebd825` (this branch with the node KV root setting
of the stacked PR). Same lane as below. 40 requests in ten minutes; the
GPU's process list was read every 2 s and held only this lane in all 307
samples, so no request is discarded. Another session read the machine's
process list once, for about a second, during the 170-second prefill that
grew the session to 120,000 tokens, and during no turn.
Each row is the median of three requests, except 5,000 new tokens and the
whole-prompt rows, which ran once.

| New tokens | First token at 8,000 tokens of context | First token at 120,000 |
| --- | ---: | ---: |
| 1 | 547 ms | 804 ms |
| 40 | 551 ms | 823 ms |
| 500 | 967 ms | 1,628 ms |
| 5,000 | 5,853 ms | 11,241 ms |
| none, whole prompt cached | 254 ms | 404 ms |

One GB10, one rank, GPU alone.

| Part of the first token | 8,000 tokens of context | 120,000 |
| --- | ---: | ---: |
| the request's first wave, one new token (largest part) | 404 ms | 518 ms |
| the last token's wave | 140 ms | 272 ms |
| send, queue, restore | under 1 ms, none | under 1 ms, none |
| prefill of 5,000 new tokens, five waves | 5,711 ms (875 tok/s) | 10,951 ms (457 tok/s) |

One GB10, one rank, GPU alone.

| Other readings | Result |
| --- | --- |
| decode, plain and as served | 7.3 and 18.5 tok/s at 8,000 tokens; 3.7 and 10.5 at 120,000 |
| a cold prompt of 8,000 tokens | 8.4 s (970 tok/s) |
| growing a session from 14,798 to 120,000 tokens | 170 s in 104 waves (618 tok/s) |
| a second request sharing 45,500 of an opening's 46,000 tokens | resumed at 32,768, prefilled 13,232: 18.5 s (the first took 56.3 s) |
| a second request sharing 35,890 of 37,000 tokens | resumed at 32,768, prefilled 4,232: 6.2 s (the first took 44.3 s) |
| the first turn after an engine stop and start, 127,099 tokens, 300 new | waited 121 ms for the session's state, re-read no page: 1.6 s to its first token |
| saves to NVMe | 3,100 pages, 18.3 GB, at 2.4 GB/s |

A second request on a shared opening resumes at the last multiple of 16,384
tokens below what it shares: this engine keeps state only at those
checkpoints, so it computed 12,700 and 3,100 shared tokens again.

### Finding: a request's first wave has a fixed start cost

One GB10, one rank, GPU alone.

A request's first wave costs about 250 ms more than a plain one-row wave,
at either context length:

- About 105 ms pass between its dispatch and the rank's work on it.
- When it resumes a session with new tokens, the wave itself is about
  140 ms longer than a plain one-row wave.

So a turn with one new token takes two one-row waves plus that cost: 547 ms
at 8,000 tokens, where a one-row wave is 135 ms, and 804 ms at 120,000,
where it is 270 ms. It is the largest part of a small turn on this rank and
does not come from the new tokens. `docs/AGENT_TURN_BENCHMARK.md`,
"Finding: the start cost of a request's first wave", takes it apart: it is
the session's recurrent state, 154.9 MB on this lane, read through the state
store when the request is admitted and written again where the request
keeps a new state, with a SHA-256 over all of it each time.

The evidence is in `qwen3.8-27b-fp8-one-gb10-alone/engine.log` (`service_ms`
is the rank's work, `wall_ms` the time from dispatch to the wave's return):

| Line | Request | Wave | `service_ms` | `wall_ms` |
| ---: | --- | --- | ---: | ---: |
| 19 | 100002, one new token at 8,016 | its first wave, one row | 278.74 | 386.68 |
| 20 | 100002 | the last token's wave | 134.35 | 135.10 |
| 83 | 100012, whole prompt cached, 14,799 | its only wave | 144.76 | 249.98 |
| 232 | 100015, one new token at 120,016 | its first wave, one row | 408.12 | 513.00 |
| 233 | 100015 | the last token's wave | 269.04 | 269.60 |
| 320 | 100025, whole prompt cached, 126,799 | its only wave | 281.48 | 389.66 |

### The wait behind a background wave, after the fix

One GB10, one rank, GPU alone.

| Probe of 2,048 cached tokens | First token | Its wave: `wall_ms`, `service_ms` | `background_wait_ms` |
| --- | ---: | ---: | ---: |
| sent alone, five times | 235 ms (median) | 233 to 289, 127 to 132 | 0 each time |
| sent during a 20,000-token background prefill, once | 271 ms | 268.94, 126.54 | 142.1 |

The engine now reports the wave's time beyond the rank's work, 142 ms. A
probe sent alone spends about 105 ms of that kind too, and the two overlap:
the background wave delayed this probe's first token by about 36 ms.

## The instruments

### Configuration

- One GB10, one rank, adapter `libqwen38_27b_tp1_serving_adapter.so`: 8
  sequences, 1,024 rows a wave, 131,072 positions, 64-token blocks, a KV
  pool of up to 16 GiB. Kernel 7.0.0-1025-nvidia, driver 580.178.04.
- Node files built from `5ef8180ce`: this branch at `a794ef3d7` plus the
  node KV root setting of the stacked PR.
- `tools/turn_floor.py` ran on the same machine against the engine API with
  token ids, speculation off unless a row says as served. The miss ledger
  ran through this branch's chat layer on another machine, over a tunnel.
- The run took four minutes. One other service held about 15 GB of the
  GPU's memory throughout; nothing else ran on the GPU during any step.

### What the run showed

One GB10, one rank, GPU shared with another service, not a floor.

| Instrument | Result |
| --- | --- |
| `POST /v1/cache/state`, a prefix the engine holds | 14,209 of 14,210 tokens held, resident, saved, with state and read, from one rank of one |
| the same prefix after the API and the rank were stopped and started, weight daemon kept | 14,209 held and saved, state 0, read 0. The next request on it resumed at 14,209, waited 115 ms for the state and re-read no page |
| a prefix the engine never saw | every count 0 |
| the `cache` object of a completion | present on all 21 requests; its `restore_ms` on the resumed turn is 115.1 against 116.0 read from the refused waves in the log; `boot` changes with the restart |
| the probes behind a background wave | first token 242 ms sent alone, 246 and 669 ms behind a 20,000-token background prefill |
| `background_wait_ms` of those probes | **wrong: 0 for both.** See below |
| the miss ledger through the chat layer | 10 records on the Messages, Responses and chat completions shapes, all carrying the engine's report: `first_opening` on a session's first request, no cause on its turns (1 token beyond the tail), `restored` with 114.9 and 95.3 ms on the first turn after an engine restart |

### The fault it found

The engine reported no wait behind the background wave. `background_wait_ms`
counted only the time a request could not be dispatched. On this rank a
probe is dispatched at once and then waits at the rank, which serves one
wave at a time: the slower probe's first wave took 667 ms for 130 ms of the
rank's own work. Commits `d2da5475f` and `c7e9233f1` count that time: the part of
each of the request's waves beyond the ranks' own work on it, as they report
it, up to the time a background wave was running. It is tested against a
mock rank that serves one wave at a time and one that serves two side by
side, and the turn floor run above shows it on the real rank.

### First numbers

One GB10, one rank, GPU shared with another service, not a floor.

| Request at 8,000 tokens of context | First token | Waves before the last token | The last token's wave |
| --- | ---: | ---: | ---: |
| 1 new token | 561 ms | 421 ms, one wave | 138 ms |
| 40 new tokens | 552 ms | 412 ms, one wave | 137 ms |
| 500 new tokens | 954 ms | 813 ms, one wave | 139 ms |
| 5,000 new tokens | 5,862 ms | 5,717 ms, five waves | 142 ms |
| nothing new, whole prompt cached | 255 ms | none | 252 ms |
| the first 8,000 tokens, nothing cached | 8,417 ms | 8,282 ms, eight waves | 134 ms |

Decode ran at 7.3 to 7.7 tok/s plain and 8.9 as served. A prefill wave of
1,024 rows took about 1.1 s (875 to 966 tok/s).

### The start cost in this run

One GB10, one rank, GPU shared with another service, not a floor.

A wave of one row is about 135 ms of this rank's work. A request's first
wave costs more, in two ways:

- About 107 ms pass between its dispatch and the rank's work on it. With
  the whole prompt cached, the only wave took 252 ms for 145 ms of service.
- When it resumes a session with new tokens, the wave itself is about
  140 ms longer than a plain one-row wave: 277 ms of service for one row.

So of the 522 ms one of the two one-token turns took to its first token,
about 250 ms is that start cost. It is the first part of a turn's floor that
does not come from the new tokens, and any rule that adds a wave to a turn
adds to it.

The evidence is in `engine.log` (`service_ms` is the rank's work, `wall_ms`
the time from dispatch to the wave's return):

| Line | Request | Wave | `service_ms` | `wall_ms` |
| ---: | --- | --- | ---: | ---: |
| 15 | 100002, a turn with one new token | its first wave, one row | 277.45 | 383.38 |
| 16 | 100002 | the last token's wave, one row | 135.21 | 135.70 |
| 61 | 100009, whole prompt cached | its only wave, one row | 144.93 | 252.28 |

## Files

| File | What it is |
| --- | --- |
| `qwen3.8-27b-fp8-one-gb10/requests.jsonl` | every request and every answer of `/v1/cache/state`, as the driver wrote them |
| `qwen3.8-27b-fp8-one-gb10/engine.log` | the engine API's log lines the split reads, over the run |
| `qwen3.8-27b-fp8-one-gb10/parts.jsonl`, `summary.json` | the split of each first token, and the summary with the cache states |
| `qwen3.8-27b-fp8-one-gb10/unknown.json` | the answer for the prefix the engine never saw |
| `qwen3.8-27b-fp8-one-gb10/ledger.jsonl` | the ledger fields of the ten requests that went through the chat layer |
| `qwen3.8-27b-fp8-one-gb10-alone/requests.jsonl`, `engine.log`, `parts.jsonl`, `summary.json` | the same four files for the turn floor run |
| `qwen3.8-27b-fp8-one-gb10-alone/gpu.json` | the count of GPU process list samples and the requests they would discard (none) |
| `qwen3.8-27b-fp8-one-gb10-alone/rank-phases.log` | the rank's own lines over the run: its state size, its phase times per sixteen prefill waves and its frame times |
