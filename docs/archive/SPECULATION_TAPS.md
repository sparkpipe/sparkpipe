# Speculation taps and the rtx5090 relay link

Owner ruling (2026-09-29): drafters run only on the rtx5090; the fleet runs
exact verification. This document is the contract for the fleet side of that
path: which hidden rows the engine taps, how they reach the rtx5090 draft
daemon, how they are dumped for offline acceptance, and what is measured.
Plan context: [SPECULATION_PLAN.md](SPECULATION_PLAN.md) sections 2.2 and 9.4;
host: [RTX5090_SPECULATION_NODE.md](RTX5090_SPECULATION_NODE.md).

## What is tapped

- A tap is the output of target layer `L` (0-based): the residual after that
  layer's MLP post, taken on one TP rank (the tap rank, default the last rank,
  rank 15 = sparkf). After the all-reduce every rank holds the full residual.
- `mean` reduces the model's residual streams (GLM Flash: 4 hyper-connection
  streams) to one row with the same kernel that feeds the head and the
  in-engine MTP drafter. The `mean` tap of the last layer is therefore
  bit-identical to the MTP layer's hidden input (GPU-tested, below).
  bf16, `hidden` elements (GLM Flash: 8 KiB per row).
- `all` keeps every stream: bf16, `streams x hidden` elements (32 KiB per row),
  for the offline reduction sweep (which reduction and layer offset a drafter
  was trained on).
- Up to 8 layers, strictly increasing. GLM Flash examples:
  - MTP: `mean:44`
  - DFlash2 (incoai config `target_layer_ids` 5/14/24/33/42) plus MTP:
    `mean:5,14,24,33,42,44`. The HF convention indexes layer outputs; the
    offline lane sweeps offset +-1 before pinning it.
- One record per committed position: sequence id, position, the token at that
  position (the row's input), the engine's output token at that row (the
  committed token at position+1 for decode and verify rows and for the last
  row of the final prefill chunk; a prediction for other prefill rows), then
  one row per tap in configured order.
  Prefill emits every prompt position; a plain step emits its rows; a verify
  round emits rows 0..accepted (the bonus token has no row yet). Rejected
  draft rows are never emitted.

## Exactness

Taps only read the residual after the layer finished; nothing downstream reads
the tap buffers, and a disabled tap set adds no graph node or launch.
Records are emitted only after the stream that wrote them finished: graph
steps and inner steps of a multi-step chain already wait for it, and with
taps on the tap rank also waits for the stream at the end of a prefill or
eager chain before reading the rows (an idle-stream query when the graph
already waited).

- GPU (sparkf, `make -C modules/glm5_next_resident_decode_stage
  validate_mtp_parity`): 88 steps with every layer tapped (`mean` and `all`)
  give the taps-off tokens byte-exact with identical KDA state, conv windows,
  KV and index caches; the last-layer `mean` tap equals `hc_mean` (the MTP
  input) and the `all` tap equals the residual streams.
- Host (`tests/test_glm5_next_stage_context.py`): oracle and adversary verify
  frames with a tap dump open serve the same tokens and dump each committed
  position exactly once, in order, each verify row from its wave slot.
  Prefill chunks and multi-step decode frames through the module's execute
  path dump and relay every committed position once, in order, with its
  input token, output token, flags and each tap's row; B1 rounds send the
  REQUEST after the round's taps, anchored at the committed token, and a late
  draft is scored in shadow. The graph walk adds one capture after each
  tapped layer's MLP post and nothing with taps off.
- The relay runs in shadow mode only: a received draft is scored, never
  verified. Serving drafts through the relay needs the root-rank broadcast of
  each round's draft to all ranks (not built), so configuration refuses
  anything but `SPARK_GLM5_NEXT_TAP_RELAY_SHADOW=1`.

## Configuration (glm5_next)

| Variable | Meaning |
| --- | --- |
| `SPARK_GLM5_NEXT_TAPS` | `mean:L[,L...]` or `all:L[,L...]`; unset = taps off. Every other tap variable requires it |
| `SPARK_GLM5_NEXT_TAP_RANK` | tap rank, default `tp_degree - 1`; it must own the tap layers, the embedding and the head |
| `SPARK_GLM5_NEXT_TAP_DUMP` | dump path; the file must not exist |
| `SPARK_GLM5_NEXT_TAP_DUMP_MAX_BYTES` | required with a dump; at the limit the dump stops (`TAP-DUMP-STOPPED`, header flag truncated), serving continues |
| `SPARK_GLM5_NEXT_TAP_RELAY_LOCAL` / `_PEER` | `ip:port` of the tap rank and of draftd (sparkf `10.10.250.1:0`, rtx5090 `10.10.250.2:<port>`) |
| `SPARK_GLM5_NEXT_TAP_RELAY_SHADOW` | must be `1` |
| `SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US` | 0..100000. 0 = send the request and score the draft whenever it lands; N = busy-wait up to N us for the draft, the cost a serving round would pay |
| `SPARK_GLM5_NEXT_TAP_RELAY_DEPTH` | draft tokens requested per round, 1..16 |

Taps need a consumer (a dump, the relay, or both). The legacy MTP chain
(`mtp_enabled`) is refused. The tap rank logs `TAP-STATS` at powers of two of
emissions and `TAP-SUMMARY` at shutdown: records, dump records and refusals,
relay records sent and unsent, requests, in-time and late answers, deadline
misses, shadow rounds, proposed, accepted, accepted-length histogram
(`accepted_at=a0,a1,...`) and round-trip p50/p99/max.

## Wire (UDP on the sparkf-rtx5090 /30, MTU 9000)

All frames are little-endian and carry the engine generation.

`SPT1` tap fragment, 72-byte header plus up to 8192 payload bytes (one
datagram of at most 8264 bytes):

| Offset | Field |
| --- | --- |
| 0 | magic `SPT1` (0x31545053) |
| 4 | version 1 (u16), kind 3 (u16) |
| 8 | engine generation (u64) |
| 16 | sequence id (u64) |
| 24 | position (u64) |
| 32 | serial (u64), increasing per record on the tap rank |
| 40 | tap-set fingerprint (u64) |
| 48 | token id (u32) |
| 52 | next token id (u32) |
| 56 | flags (u32): 1 prefill, 2 decode, 4 verify |
| 60 | record bytes (u32) = taps x row bytes |
| 64 | offset of this slice in the record (u32) |
| 68 | reserved (u32), zero |
| 72 | payload slice |

Fragments of a record are sent in offset order. The assembler keeps one open
record; a missing or out-of-order slice abandons it (counted), a fingerprint
or size mismatch is refused. A drafter detects lost records by position
continuity per sequence.

After a round's taps, the tap rank sends the existing `SPR1` REQUEST (the
committed tokens since the last request, ending at the anchor) and draftd
answers with an `SPR1` DRAFT for that engine generation, round, sequence and
anchor (layout in `include/sparkpipe/spark_speculation_relay_draft.h`). A
draft for an older round is counted stale; in shadow mode it is still scored.

## Dump (`SPTD`)

128-byte header, then records of a 40-byte record header and the rows.

| Offset | Header field |
| --- | --- |
| 0 | magic `SPTD` (0x44545053) |
| 4 / 8 / 12 | version 1, header bytes 128, record header bytes 40 |
| 16 / 20 / 24 / 28 / 32 | reduction (1 mean, 2 all), tap count, streams, hidden, model layers |
| 36 / 40 / 44 | row elements, row bytes, dtype (1 = bf16) |
| 48 | tap layers, 8 x u32 |
| 80 / 88 | tap-set fingerprint, engine generation (u64) |
| 96 / 100 | tp rank, flags (1 truncated, 2 closed) |
| 104 | record count (u64, written at close) |
| 112 | model tag, 16 bytes |

Record header: sequence id (u64), position (u64), token id (u32), next token
id (u32), flags (u32), reserved (u32, zero), serial (u64).

`tools/spec_tap_dump.py`:
- `summary|verify PATH`: header, per-sequence coverage, gaps and repeats,
  and stale reads: a record whose next token differs from the following
  position's token (the final prefill row, decode and verify rows), or tap
  rows equal to another position's rows (identical rows at the same position
  of two sequences pass only when both share every token up to it). A tap
  rank that reads its host rows before the stream finished shows up as
  either; `verify` exits 1 on any problem;
- `export-offline PATH OUT --layer L --model M --firmware F [--classes JSON]`:
  one tapped layer as a `spark-tapdump-1` directory, the format
  `tools/spec_offline/tapdump.py` reads (tokens = the record tokens plus the
  last record's next token; the final token's row is zero because the engine
  never computes it; sequences that do not start at position 0, such as
  prefix-cache hits, are skipped and listed);
- `iter_records` and `tap_rows` for direct use.

## Tools

- `build/spark_speculation_relay_probe draftd LOCAL TAPS LAYERS STREAMS HIDDEN SECONDS`
  answers every REQUEST at once and assembles taps;
  `... engine LOCAL PEER TAPS LAYERS STREAMS HIDDEN ROUNDS ROWS AWAIT_US PACE_US DEPTH`
  plays the tap rank and prints `RELAY-PROBE-ENGINE` with tap send time and
  round-trip percentiles. Model-neutral; no GPU.

## Measured (2026-09-29, sparkf to rtx5090, probe, perf window held)

Round trip = REQUEST sent (after the round's taps) to DRAFT received, draftd
answering at once; await 2 ms, depth 7.

| Pacing between rounds | Taps per round | Tap send p50/p99 (us) | RTT p50/p99/max (us) | Misses |
| --- | --- | --- | --- | --- |
| 0 (back to back) | none | - | 40 / 570 / 806 | 0/2000 |
| 0 | 6 x mean (48 KiB) | 8 / 16 | 168 / 536 / 826 | 0/2000 |
| 20 ms (a B1 step) | none | - | 266 / 722 / 780 | 0/500 |
| 20 ms | 6 x mean (48 KiB) | 43 / 97 | 284 / 718 / 954 | 0/500 |
| 20 ms | 4 rows x 6 x mean (192 KiB) | 82 / 162 | 350 / 904 / 1069 | 0/500 |

At a real step cadence the idle link adds about 230 us per round over the
back-to-back figure. Both hosts run with `net.core.busy_poll=0` and
`busy_read=0`, so receive still waits for an interrupt on a possibly idle
core. Round overhead at 20 ms cadence is about 0.3 ms p50 per round
(1.4% of a 21 ms GLM Flash B1 step) before draft time.
