# Instruments on one real rank (2026-11-22)

This folder holds one run of the stage 0 instruments against a real rank:
Qwen 3.8 27B FP8 on one GB10, engine API and rank on the same machine. It
checks that the instruments tell the truth on a real rank. It is not a turn
floor: the GPU was shared with another service for the whole run. Nothing
here stands for a sixteen-rank lane.

## Configuration

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

## What the run showed

| Instrument | Result |
| --- | --- |
| `POST /v1/cache/state`, a prefix the engine holds | 14,209 of 14,210 tokens held, resident, saved, with state and read, from one rank of one |
| the same prefix after the API and the rank were stopped and started, weight daemon kept | 14,209 held and saved, state 0, read 0. The next request on it resumed at 14,209, waited 115 ms for the state and re-read no page |
| a prefix the engine never saw | every count 0 |
| the `cache` object of a completion | present on all 21 requests; its `restore_ms` on the resumed turn is 115.1 against 116.0 read from the refused waves in the log; `boot` changes with the restart |
| the probes behind a background wave | first token 242 ms sent alone, 246 and 669 ms behind a 20,000-token background prefill |
| `background_wait_ms` of those probes | **wrong: 0 for both.** See below |
| the miss ledger through the chat layer | 10 records on the Messages, Responses and chat completions shapes, all carrying the engine's report: `first_opening` on a session's first request, no cause on its turns (1 token beyond the tail), `restored` with 114.9 and 95.3 ms on the first turn after an engine restart |

## The fault it found

The engine reported no wait behind the background wave. `background_wait_ms`
counted only the time a request could not be dispatched. On this rank a
probe is dispatched at once and then waits at the rank, which serves one
wave at a time: the slower probe's first wave took 667 ms for 130 ms of the
rank's own work. Commit `d2da5475f` counts that time. It is tested against a
mock rank that serves one wave at a time and has not yet run on a real rank.

## First numbers (GPU shared, not a floor)

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

Where a small turn's time goes on this rank: a wave of one row is about
135 ms of the rank's work. A request's first wave costs more: about 107 ms
pass between its dispatch and the rank's work on it (252 ms against 145 ms
of service when the whole prompt is cached), and when it resumes a session
with new tokens the wave itself is about 140 ms longer than a plain one-row
wave (277 ms of service for one row). So of the 522 ms one of the two
one-token turns took to its first token, about 250 ms is that start cost.

## Files

| File | What it is |
| --- | --- |
| `qwen3.8-27b-fp8-one-gb10/requests.jsonl` | every request and every answer of `/v1/cache/state`, as the driver wrote them |
| `qwen3.8-27b-fp8-one-gb10/engine.log` | the engine API's log lines the split reads, over the run |
| `qwen3.8-27b-fp8-one-gb10/parts.jsonl`, `summary.json` | the split of each first token, and the summary with the cache states |
| `qwen3.8-27b-fp8-one-gb10/unknown.json` | the answer for the prefix the engine never saw |
| `qwen3.8-27b-fp8-one-gb10/ledger.jsonl` | the ledger fields of the ten requests that went through the chat layer |
