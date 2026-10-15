# SparkPipe Performance Status

This file is the only current performance ledger. Measurements include exact
scope and identity. Projections are kept in a separate section and never count
as achieved results. A number without a committed receipt names its source.

Measurements of deprecated models and of the residentd transport crossovers
(DeepSeek V4 Flash TP4, GLM 5.2 TP8, 2026-08-13 to 2026-08-16) are archived in
[`docs/archive/PERFORMANCE_HISTORY_2026-08.md`](docs/archive/PERFORMANCE_HISTORY_2026-08.md).

## Measured fabric

### Eight pairwise direct links

On 2026-08-13 all eight direct pairs ran simultaneous traffic in both
directions. Every pair passed the 80 Gb/s-per-direction gate.

| Metric | Result |
| --- | ---: |
| Slowest observed direction | 91.669 Gb/s |
| Fastest observed direction | 105.907 Gb/s |
| Combined sixteen-direction throughput | 1.643 Tb/s |

These useful rates explain why the nominal 200 Gb/s direct links behave like
approximately 100 Gb/s links after the GB10 PCIe limit. A 100 Gb/s switched
port therefore does not reduce practical per-port throughput.

### Combined direct and switched rails

The isolated TP4 two-port characterization measured:

| Metric | Result |
| --- | ---: |
| Simultaneous two-port aggregate ceiling | 213.687 Gb/s |
| Best sustained 14 MiB split ring | 193.018 Gb/s |
| Ceiling utilization | 90.328% |

The sustained result used two counter-rotating rings, four streams, four
credits, and one verbs QP per route, on the residentd transport that only
k3's runner still creates. It is an isolated transport measurement, not model
decode throughput.

### Weightd mesh all-reduce

On the fleet after PR #1208 (2026-09-25), an 8 KiB 16-rank all-reduce on the
weightd mesh took 167 µs p50, 472 µs p90, 3 ms p99 and 12 ms at most
([`docs/GLM5_NEXT_ROOFLINE.md`](docs/GLM5_NEXT_ROOFLINE.md)). Every
hardware-wait round crosses weightd's CPU relay twice
([`TECHDEBT.md`](TECHDEBT.md), Mesh collectives). At the 92 all-reduces per
GLM 5.3 Flash step used in the planning table below, the p50 alone is about
15.4 ms per token when nothing overlaps (arithmetic). No other single cost in
the GLM B1 step is that large.

### Hub link

On 2026-09-28 a busy-polled 64-byte UDP round trip between `sparkf`
(`enP7s7`, 10.10.250.1) and the rtx5090 hub (`eno1`, 10.10.250.2) over the
direct 10 GbE link took 22.6 µs p50 and 25.2 µs p99. From another Spark,
routed through `sparkf`, the p50 was about 300 µs. Source: lead-dev
measurement; no receipt is committed. A draft model hosted on the rtx5090
would use this link.

## Measured model performance

### Release qualification, all 16 models (2026-10-14)

Release `1d89e0377` (weightd `9e9618df`) on sixteen Sparks at TP16, measured
by `scripts/release_rollout.sh qualify` with `api_serving_perf.py` against
each model's engine endpoint: decode is one greedy request without
speculation generating 256 tokens (`b1_o256`), prefill is one cold prompt of
about 7,400 tokens (`ttft6000`). Swap-in times are medians of the swaps the
controller logged on 2026-10-13 and 2026-10-14. Receipts:
[`qualification/releases/1d89e0377/`](qualification/releases/1d89e0377/)
(one `perf.jsonl` per model, `swaps.jsonl`, `summary.json`).

| Served name | Model | Family | Context | Swap in | Decode, B1 | Prefill, 7K prompt |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `glm-5.3` | GLM 5.3 Full | `glm52` | 256K | 38 s | 24.3 tok/s | 984 tok/s |
| `glm-5.3-flash` | GLM 5.3 Flash | `glm5_next` | 128K | 24 s | 42.1 tok/s | 804 tok/s |
| `kimi-k3` | Kimi K3 | `k3` | 128K | 46 s | 10.0 tok/s | 469 tok/s |
| `deepseek-v4.1-flash` | DeepSeek V4.1 Flash | `dsv41_flash` | 16K | 24 s | 25.1 tok/s | 167 tok/s |
| `mimo-v2.6-pro` | MiMo V2.6 Pro | `mimo26` | 32K | 24 s | 19.0 tok/s | 55 tok/s |
| `mimo-v2.6-flash` | MiMo V2.6 Flash | `mimo26` | 32K | 19 s | 40.9 tok/s | 157 tok/s |
| `hy4-preview` | Hunyuan HY4 preview | `hy4` | 32K | 28 s | 12.0 tok/s | 50 tok/s |
| `gemma-4-31b` | Gemma 4 31B | `gemma4` | 32K | 17 s | 27.9 tok/s | 734 tok/s |
| `gemma-4-26b-a4b` | Gemma 4 26B A4B | `gemma4` | 32K | 16 s | 75.2 tok/s | 1,705 tok/s |
| `ling-3.0-flash` | Ling 3.0 Flash | `ling` | 32K | 20 s | 67.6 tok/s | 1,361 tok/s |
| `ling-3.0-flash-fin` | Ling 3.0 Flash, finance fine-tune | `ling` | 32K | 20 s | 80.0 tok/s | 1,524 tok/s |
| `laguna-s-2.1` | Laguna S 2.1 | `laguna` | 32K | 20 s | 42.1 tok/s | 700 tok/s |
| `muse-glimmer-30b` | Muse Glimmer 30B | `muse_glimmer` | 32K | 16 s | 32.1 tok/s | 1,152 tok/s |
| `qwen3.8-max` | Qwen 3.8 Max (internal) | `qwen38` | 32K | 35 s | 8.2 tok/s | 34 tok/s |
| `qwen3.8-27b` | Qwen 3.8 27B (internal) | `qwen38` | 256K | 19 s | 48.3 tok/s | 1,721 tok/s |
| `qwen3.8-flash` | Qwen 3.8 Flash (internal) | `qwen38` | 32K | 20 s | 38.0 tok/s | 344 tok/s |

Conditions and limits:

- A CPU tenant kept about 16 cores busy on every node, and no other GPU work
  ran. On a quiet fleet on 2026-10-08 (older build) K3 decoded 16.3 tok/s at
  B1, against 10.0 here.
- The same build on the quiet fleet, 2026-10-14 12:45Z after the tenant
  stopped (exclusive tp16 hold, `b1_o256` x 9): K3 median 16.17 tok/s; seven
  runs read 16.10 to 16.25, and the first two (10.74, 11.93) overlapped chat
  requests. The release's 10.0 was the CPU load slowing every collective
  round, not a regression
  (`qualification/releases/1d89e0377/kimi-k3.quiet-20261014.perf.jsonl`).
- Each number is one run. Cross-hold B1 swings of 5 to 10% are normal; the
  previous release, `e05394f94`, measured the same day, read for example 9.2
  tok/s for Qwen 3.8 Max and 11.4 for K3.
- All 16 models produced the old build's tokens on the six release prompts
  (`EXACT 6/6`), with no error lines in any rank log.

### Speculation: long verify frames on code edits, K3 (2026-10-15)

K3 TP16 now verifies lookup drafts too: verify steps run with commit off and
the kept rows are folded into the KDA state with the forward's own kernels
([`docs/SPECULATION_LONG_FRAMES.md`](docs/SPECULATION_LONG_FRAMES.md)). One
exclusive hold, 2026-10-15 12:55-13:41Z, arms C (`36f24360e`: same build,
verify removed from the K3 descriptor) and P (`1134081e3`: frames up to 255
drafts that cross cache blocks, priced frame policy), order C1 P1 P2 C2, each
after a KV snapshot and prefix-index wipe. Same 20-task code-edit corpus as
GLM, greedy, thinking off; decode tok/s excludes the time to the first token.
Receipts: [`qualification/k3/spec-long-frames-20261015/`](qualification/k3/spec-long-frames-20261015/).

| Edit style | C tok/s | P tok/s | P per task |
| --- | ---: | ---: | ---: |
| whole file (15.9K tokens) | 15.3 | 126.4 (8.3x) | 77-161 |
| SEARCH/REPLACE (2.0K tokens) | 15.0 | 28.0 (1.9x) | 13-34 |

- All 20 P streams are token-identical to C: on this corpus the fold of the
  kept rows is exact on hardware.
- 384 verify frames, 22,200 drafts, 16,902 accepted (76%).
- Story prompt, 256 tokens, medians of 4 (greedy, T=1 top-p 0.95) or 2 (T=1):

| Case | C1 / C2 tok/s | P1 / P2 tok/s |
| --- | ---: | ---: |
| greedy | 16.07 / 16.18 | 18.71 / 18.84 |
| T=1, top-p 0.95 (chat) | 15.60 / 15.64 | 15.57 / 15.61 |
| T=1 | 15.33 / 15.59 | 16.46 / 16.45 |

- Sampled verify keeps the sampled stream: 5 of 6 sampled P streams equal C1
  token for token (one diverges at token 178 into a coherent alternative); C2
  equals C1 10/10. Greedy story streams diverge at near-ties (tokens 81-209)
  into coherent continuations.
- Enabling verify switches K3 decode publication from deferred to inline; chat
  decode speed did not move.

### Speculation: long verify frames on code edits, GLM Full (2026-10-15)

Lookup speculation drafts from the request's own history and verifies up to
62 drafts per frame (frames still inside one 64-token cache block); mechanism
in [`docs/SPECULATION_LONG_FRAMES.md`](docs/SPECULATION_LONG_FRAMES.md). One
exclusive GLM Full TP16 hold, 2026-10-15 11:27-12:15Z, arms N (same build,
verify removed from the GLM descriptor) and S (`a161d0de7`), order N1 S1 S2
N2, each after a KV snapshot and prefix-index wipe. Workload: the 20-task
code-edit corpus (ten repository files, one function rename each, replied as
the whole file or as SEARCH/REPLACE blocks; `tools/spec_edit_corpus.py`),
greedy, reasoning effort low. Decode tok/s is output tokens over the time
after the first token.

| Edit style | N tok/s | S tok/s | S per-task range |
| --- | ---: | ---: | ---: |
| whole file (15.8K tokens) | 27.6 | 124.2 (4.5x) | 94-132 |
| SEARCH/REPLACE | 27.5 | 36.0 (1.3x) | 23-41 |

- 882 verify frames, 21,421 drafts, 17,719 accepted (83%). S1 and S2 are
  token-identical and within 0.2% in speed; so are N1 and N2.
- The logprob goldens (which never draft) are bit-identical on every arm.
- Hold 2 (12:19-12:55Z, frames crossing blocks with the priced policy,
  `1134081e3` against its control `36f24360e`): whole file 27.5 -> 120.7
  tok/s, SEARCH/REPLACE 27.5 -> 34.7; prose (256 tokens, greedy) 28.3-28.6
  against 28.3-28.5 with speculation on; 16 of 20 streams equal the control,
  goldens bit-identical. Crossing blocks did not beat the in-block frames of
  hold 1 on whole-file edits; per-frame costs are being measured
  (receipts under `hold2/`).
- Hold 3 (13:42-14:11Z, tip `a397d67b0`, which prices frames by node
  service time): whole file 121.4 and 120.6 tok/s in two runs, SEARCH/REPLACE
  37.8 and 37.4. Frames per row bucket from the engine log: 2 rows 62 ms
  (9 frames), 3 rows 73 ms (7), 12.6 rows 176 ms, 25 rows 238 ms, 42 rows
  284 ms (435 of 525 frames), 68 rows 307 ms, 256 rows 529 ms; the chained
  token costs 36 ms and a single step 48 ms. Long-context corpus (the same
  edits behind about 40K tokens of context): control `d2e0fda7b` 24.8 tok/s
  whole file and 24.1 SEARCH/REPLACE, tip 82.4 and 32.7, the tip without the
  scatter choice (`0d9319a79`) 24.9 and 24.6: with gather attention a long-
  context frame costs more than it saves, so the policy drafts almost nothing.
- Why frames stayed near 42 rows: a cap meant for chat-memory background work
  (frame cost at most eight chained tokens for priority 0) applied to every
  request, because requests without a priority field carry 0. `2b2fa2a44`
  applies it only below the highest priority the engine has served.
- Hold 4 (14:11-14:33Z, same node build, hub API `a397d67b0` as arm T
  against `2b2fa2a44` as arm P, order P1 T3 PL P2):

  | Edit style | T3 tok/s | P1 tok/s | P2 tok/s |
  | --- | ---: | ---: | ---: |
  | whole file (15.8K tokens) | 120.8 | 232.9 | 223.3 |
  | SEARCH/REPLACE | 37.4 | 35.3 | 36.2 |
  | whole file, long context | | 103.0 | |
  | SEARCH/REPLACE, long context | | 30.2 | |

  P frames: 109 of 193 near 241 rows keeping 138 tokens each (540 ms), 24
  near 88 rows (377 ms). Whole-file edits gain 1.9x over T and about 8.4x
  over no speculation. SEARCH/REPLACE loses 4%: on the 17 streams equal to
  T3, short replies (45-224 tokens) run 5-14% slower, because the pooled
  acceptance statistics of whole-file copies make frames on short copies too
  long (39-59% of rows kept). 17 of 20 P streams equal T3; editsr_01, 05 and
  09 diverge in both P runs.
- Near-tie probe (hold 3, control build): at the first divergence of
  editsr_01, 05 and 09 the multi-row path serves the spec stream's token, by
  1.63, 0.54 and 0.31 nats over the control's token. These are not tiny
  ties: the single-row decode path and the multi-row path disagree by at
  least that much at those positions. The decode path's own logprobs there
  are the next probe.
- 13 of 20 S streams equal N; 7 diverge, two at the reply's last token and
  five early in SEARCH/REPLACE replies, all on formatting tokens. At four of
  them the control's own prefill path at the same position serves the spec
  stream's token, so they are near-ties between numerics paths. Frames above
  8 rows run gather attention and wide GEMMs, so their logits differ slightly
  from chained decode; the near-tie check at each divergence is in the next
  hold. The diverged SEARCH/REPLACE replies are longer (6.2K against 3.4K
  tokens), so that row compares partly different text.

### K3 sampled decode: chat sampling at greedy speed (2026-10-15)

The chat layer samples at the publishers' defaults (temperature 1, top-p
0.95). Before this change K3 decoded such requests 16% slower than greedy:
the engine ran top-p and top-k lanes one token per submission, and the stage
runner's distribution head (the vocabulary-slice logits recomputed for
sampled rows) launched only 10 blocks on GB10's 48 SMs, 5.5 ms per token at
27 GB/s (single-GB10 microbench; 128-token tiles: 0.96 ms, bit-identical
logits). Top-p sampling itself takes 2.25 ms per token in its one-block
kernel; that is left.

One exclusive K3 TP16 hold, 2026-10-15 07:53-08:20Z, quiet fleet, arms
A (release adapter and API), H (128-token distribution head), HC (H plus
chaining for top-p and top-k lanes), run order A1 H1 HC1 HC2 H2 A2, each
after a KV snapshot and prefix-index wipe; per arm the STORY prompt with
256 tokens: greedy x4, T=1 top-p 0.95 with seeds 101-104, T=1 with seeds
201-202. Receipts: [`qualification/k3/sampled-decode-20261015/`](qualification/k3/sampled-decode-20261015/).

| Case | A tok/s | H tok/s | HC tok/s | Steps per submission A / HC |
| --- | ---: | ---: | ---: | ---: |
| greedy | 16.05 | 15.80 | 15.91 | 7.4 / 7.4 |
| T=1, top-p 0.95 (chat) | 13.40 | 14.38 | 15.52 | 1.0 / 7.5 |
| T=1 | 14.34 | 15.65 | 15.61 | 6.1 / 7.5 |

- Medians, n=8 per arm and case (n=4 for T=1). Every greedy and seeded
  sampled run of every arm produced exactly the tokens of A1: the Gumbel
  draw is keyed by seed, position and token, so chaining changes no sample.
- Production K3 runs HC since 08:22Z (adapter `f0fc74dc`, API `44650c56`).
  Logprob lanes still decode one token per submission. Other models get the
  engine change with the next fleet release.

### GLM 5.3 Full prefill: row shard and wide FP8 tiles (2026-10-15)

One exclusive TP16 hold on a quiet fleet (2026-10-15 06:17-07:22Z), GLM Full
lane 6, adapters built from release `1d89e0377` plus each lever, run order A1
I1 R1 C1 F1 RF1 CF1 CF2 RF2 F2 C2 R2 I2 A2. Each run wiped the KV snapshot
tier and the hub prefix index, then ran the 8 golden prompts (128 tokens,
4 logprobs) against A1, the chained no-logprobs check, streams8, `b1_o256`
x5 and `ttft1650` and `ttft6500` x2, so n=10 for B1 and n=4 for each TTFT.
Receipts: [`qualification/glm53_full/prefill-rf-20261015/`](qualification/glm53_full/prefill-rf-20261015/)
(`speed.txt`, `cmp.txt`, `hold.log`, `perf.jsonl`, and the GB10 kernel
check `fp8gemm-gb10-summary.txt`).

| Arm | Change | B1 tok/s | 2K TTFT ms | 8K TTFT ms |
| --- | --- | ---: | ---: | ---: |
| A | release lane adapter | 28.40 | 1974 | 7361 |
| I | release rebuilt by the arm build path | 28.35 | 1959 | 7332 |
| R | row shard: RS, own-row norm, AG for 256+ row waves | 28.44 | 1889 | 7067 |
| C | lane comm schedule: long all-reduces in pieces | 28.34 | 1953 | 6921 |
| F | wide FP8 warp tiles, ungrouped GEMMs | 28.30 | 1897 | 7295 |
| RF | R + F | 28.46 | 1814 | 6895 |
| CF | C + F | 28.39 | 1901 | 6886 |

- Every arm's goldens were bit-identical to A1 (8/8, chained 4/4) in both
  rounds, A2 equalled A1, streams8 had no errors, and no rank logged a mesh
  failure.
- RF and CF tie at 8K (-6.3% and -6.5%, n=4); RF is ahead at 2K (-8.1%
  against -3.7%) and leaves the shared stage runner unchanged, so lane 6 runs
  RF (adapter `09b782cf`) and RF is merged. R and C do not stack: R removes
  the all-reduces that C splits.
- On one GB10 the wide FP8 tiles run the dense GLM linears 1.9-2.2x faster
  with equal output hashes on all 21 shapes; the grouped expert GEMMs keep
  their tiles.
- B1 is unchanged; 28.4 tok/s is the quiet-fleet GLM Full number (the
  release table's 24.3 was measured under the CPU tenant).

### GLM 5.3 Flash TP16 B1 (2026-09-28)

One request on sixteen Sparks (`spark0` to `sparkf`, TP16), greedy, no
speculation: 128 generated tokens in 3.53 s end to end, reported as 36 tok/s
(arithmetic: 128 / 3.53 s = 36.3). Decode ran as resident graph chains of 8
steps at 23 to 24.5 ms per token. The TP16 memory floor for this model is
8.0 ms per token, 125 tok/s
([`docs/GLM5_NEXT_ROOFLINE.md`](docs/GLM5_NEXT_ROOFLINE.md)), so a step takes
2.9 to 3.1 times the floor (arithmetic: 23 / 8.0 and 24.5 / 8.0).

In context (all arithmetic):

- End to end works out to 27.6 ms per token (3.53 s / 128), which also counts
  the time spent outside the chain steps.
- 36 tok/s is 29% of the 125 tok/s ceiling.
- The best public 4-Spark B1 result for GLM 5.3 Flash FP8 without speculation
  is 23.2 tok/s (lead-dev survey, 2026-09-28). 36 tok/s is 1.55 times that.
  The TP16 target of 3 to 3.5 times it is 70-81 tok/s.
- The 167 µs mesh all-reduce above, times 92 per step, is about 15.4 ms of a
  23-24.5 ms step.

Source: lead-dev measurement. The raw client stream is not committed; the
next run should retain one, with three unprofiled runs.

| Field | Configuration |
| --- | --- |
| Model | `zai-org/GLM-5.3-Flash`, revision `84c6a6aa9497188e15a635ba793b0f95a79b1033` |
| Runtime root | `glm53flash.fp8.tp16` |
| Engine source | main at PR #1243 (`dd3526b`), before the i21-i30 bundles |
| Serving environment | fleet-agent drop-in `20-serving.conf`: `G5_GRAPH_PATH=1`, `G5_PIN_EXPERTS=1` (all 12096 routed experts pinned), `SPARK_TP_WAIT_MODE=hardware`, `G5_API_DISABLED=1`, `G5_WARMUP=0` |
| API | `g53-api` on the rtx5090 hub, port 8433, an x86 build of `dd3526b` |

The deployment's binaries, as recorded for the COMPSEC-17 run below on the
same day:

| Component | SHA-256 |
| --- | --- |
| residentd | `0afa3721f377e980b431d394ad544f198a839f18257ef9101d0e232f01feebaa` |
| `model_driver.so` | `0e15456abc55f1f8b0ec7f71995909f473cecf1a6d93841f470d6070da2f14f2` |
| serving adapter (ranks) | `90209fb4c8ba439ff6f7724e45b0d15647794bacaa9c6f795c1fbf760ca60125` |
| `hidden_transport.so` | `538aa5b1aed7a4270427425a25020a710ecc29d46fc9b11f24842f7367cef8f5` |
| weightd (prefix, `core/WEIGHTSD_BIN`) | `3da98597a88b60b5` |
| fleet agent (prefix) | `a05e207588da2cef` |
| API (x86) | `2acff5135b69411aa5eb20e4f4e50c0a29717203b4a39026d6ea751abbb598b4` |
| API serving adapter (x86) | `e0dac318cb8bef804f41766c738906e3378bdc064c39108463dd468bf11abbdc` |
| `tokenizer.json` | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| rank 0 pack | `318dd18ad18dc748181f22687be2a482c53f76d13c23c3056895128fa550ac80` |
| `20-serving.conf` (prefix) | `8324336487eecc38` |

Known-bad configuration: earlier the same day the fleet ran eager chains with
spin wait and unpinned experts at 5.9 tok/s B1 (lead-dev measurement). The
configuration lives only in the untracked drop-in
([`TECHDEBT.md`](TECHDEBT.md), Production qualification). The previous B1
figure, 27.6 to 29.8 tok/s after PR #1208 (2026-09-25), is in
[`docs/GLM5_NEXT_ROOFLINE.md`](docs/GLM5_NEXT_ROOFLINE.md).

### GLM 5.3 Flash TP16 COMPSEC-17 (2026-09-28)

The functional receipt for the deployment above: 14 of 17, the pass
threshold. The harness is `tools/glm5_next_compsec17.py --thinking off`: GLM
chat template with an empty think block, temperature 0, 512 tokens, one
request at a time, graded by `qualification/ds4_eval/compare_runs.py` on the
last `Answer:` line. compsec-079, compsec-080 and compsec-090 were wrong. The
completions were byte-identical to an eager, spin-wait run earlier the same
day. Earlier harness runs without the chat template scored 0/17 on the same
model; they measured the harness, not the model.

Sent concurrently four times (`--concurrency 17`), the runs scored 13 or 14
and each differed from the sequential completions in one or two cases:
batched serving is not batch-invariant ([`TECHDEBT.md`](TECHDEBT.md),
Dynamic batching).

| Receipt | SHA-256 |
| --- | --- |
| [`REPORT.md`](qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/REPORT.md) | protocol, identities and the concurrent runs |
| [`summary.json`](qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/summary.json) | `4f575d33afa48aca860b5bc6d9566985a63df7b4f8878d21072883cccc168485` |
| response stream ([`INTEGRITY.json`](qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/INTEGRITY.json)) | `e60c1a6a68abd18bf1ca8e6f5dbd116a2c4f3a8c07f3fe20af4814493f4ed512` |

### GLM 5.3 Flash TP16 B8 8-stream aggregate (2026-09-26 / 2026-09-27)

Scope: sixteen Spark nodes (spark0-f, MESH0005) serving the i17 stack
(`glm-serving-b108de0c` driver, station core c0d54638, B8 profile, 128-row
prefill); API host rtx5090. Load: 8 concurrent streams, reference prompt,
512 output tokens per stream, greedy, four consecutive whole-station sweeps
per session.

| Session | Sweep tok/s | Best |
| --- | --- | ---: |
| 2026-09-26, #1228 qualification | 114.8 / 126.3 / 125.6 / 129.9 | 129.9 |
| 2026-09-27, after full-station restart | 120.7 / 125.8 / 120.3 / 125.3 | 125.8 |

The 2026-09-26 session is receipted in the #1228 thread (summarized outputs;
identical load and stack). The 2026-09-27 session re-ran the same load after a
full station restart and is committed below. Its rank-0 window shows zero
`LINEAR-CHAIN-FAILED`, `MESH-DEFERRED-ROUNDS-FAILED` or `GRAPH-FAILED`; one
interval recorded transient `busy[CHAIN]` retries (235) during ramp that
self-recovered within the window — the persistent-freeze wedge under
investigation on #1229 is a separate issue and did not occur here.

| Receipt | SHA-256 |
| --- | --- |
| [client-8stream-sweeps.txt](qualification/glm5next/performance/tp16_b8_20260927/client-8stream-sweeps.txt) | 8f78436bc4734dd1f3f6d49ea67dd9c9283d7e75acbfd3c8b7b23b050bb223bd |
| [rank0-wave-window.log](qualification/glm5next/performance/tp16_b8_20260927/rank0-wave-window.log) | dcb332ecc624198ffc52f202c6521fa4466192230f9d67e346addf4a0d2b76f3 |

### Earlier points carried over from the performance ledger

These rows were folded in from
[`docs/archive/PERFORMANCE_LEDGER.md`](docs/archive/PERFORMANCE_LEDGER.md)
when it was archived on 2026-09-28. Only rows with a measurement identity
are kept. Its projections, status notes and the retracted Qwen 3.8 27B batch
curve are left out.

| Model | Cell | Result | Identity | Date |
| --- | --- | ---: | --- | --- |
| Qwen 3.8 27B FP8 | TP1 B1, DFlash2 k=8, 512 output tokens | 24.5 tok/s wall | merged main; stream `d7f798801a6e43a6`, 77 rounds, E≈5.66 | 2026-08-28 |
| Qwen 3.8 27B FP8 | TP1 B1, no speculation, 512 output tokens | 7.7 tok/s | merged main; stream `5d6ee525deb999f5` | 2026-08-28 |
| Qwen 3.8 27B | TP1 B1, no speculation, high-water mark | 8.03 tok/s | mixed/FP8 29.9 GB pack; dashboard reading | 2026-08-28 |
| Qwen 3.8 Max FP8 | TP4 x PP4, B1 per request | 1.29 tok/s | measured anchors | 2026-08 |
| K3 MXFP4 | warm B1 step of stage 0 (24 layers) of a TP4 rank pack, one Spark | 55.5 ms per stage step | [`docs/K3_PERF.md`](docs/K3_PERF.md) | 2026-08-16 |
| DSV4 Flash FP8 | TP4 B1, 128 output tokens, main plus PR #731 | 40.46 / 40.35 / 40.19 tok/s | exact token hash `211462f2…` | 2026-08-28 |
| DSV4 Flash FP8 | TP4 B1, exact 32K cell, lean branch | 29.37 tok/s | ledger scoreboard only | 2026-08-28 |

No Qwen 3.8 27B point has a frozen qualification bundle; the ledger kept
only the stream hashes. The DSV4 rows exist only as ledger scoreboard
entries, with no committed receipt. K3's only measured point is that one
stage step, not a fleet run. The 18.0 tok/s TP4xPP4 pipelined rate and about
4.5 tok/s for one sequence through four stages are arithmetic from it, and the
20.6 tok/s roofline is calculated ([`docs/K3_PERF.md`](docs/K3_PERF.md)).

Public comparables are other stacks' numbers, graded as their authors
announced them. None of them is a SparkPipe measurement:

| Stack | Point | Speculation | Grading |
| --- | ---: | --- | --- |
| gb10-vllm v5-prd, K3, 16 Sparks, TP16+DCP8 | C1 29.81 tok/s | DSpark nst6 | announced |
| gb10-vllm v5-prd, K3, llama tg2048 | 23.59 t/s at depth 4000 | none | announced |
| Best public GLM 5.3 Flash FP8, 4 Sparks, B1 | 23.2 tok/s | none | lead-dev survey, 2026-09-28 |

The ledger's K3 row of 29.0 tok/s, which claimed to supersede SparkPipe's
18.0, was the gb10-vllm author's speculative result
([`docs/K3_VS_GB10_VLLM.md`](docs/K3_VS_GB10_VLLM.md)). It is not a SparkPipe
number.

## Planning projections

The following values guide architecture choices and are not measurements.

### GLM 5.3 Flash TP16 decode

The site's throughput figures come from this table. It assumes that the M1
and M2 work lands, but short of their exit targets: kernels at 75% of memory
bandwidth where M1 asks for 80%, and all-reduces that do not overlap compute.
None of the projected values is measured.

A step takes the bytes each rank reads, at 75% of 273 GB/s, plus 92
all-reduces, plus 0.5 ms of host work:

- **Kernels at 75% of memory bandwidth.** Today the batch harness reaches
  128 GB/s (47%) at B1 and 151 GB/s (55%) at B8, and its last recorded B256
  step, 302 ms, ran at about 30%. The skinny BF16 GEMV already runs at
  234 GB/s, and the grouped expert kernel at 260 GB/s on its own.
- **All-reduces on GPU-initiated RDMA**, at 25 µs per round: the "tens of
  microseconds" that [`docs/GLM5_NEXT_ROOFLINE.md`](docs/GLM5_NEXT_ROOFLINE.md)
  expects once the CPU relay leaves the critical path. B1 takes one direct
  round. From B8 up, a reduce-scatter plus all-gather is cheaper: two rounds,
  with 1.875 times the payload on the wire, 14/15 of it through the rank's
  100 Gb/s switched port.
- **0.5 ms of host work per step**, with several decode steps per submission.
- 1K context, and no speculative decoding.

| Streams | Bytes per rank | Kernels | All-reduces | Step | Projected tok/s | Per stream | Ceiling | Measured |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.18 GB | 10.7 ms | 3.2 ms | 14.4 ms | 70 | 70 | 125 | 36 |
| 8 | 5.6 GB | 27.4 ms | 5.4 ms | 33.3 ms | 240 | 30 | 390 | 129.9 |
| 64 | 18.6 GB | 90.8 ms | 11.4 ms | 102.7 ms | 623 | 9.7 | 940 | - |
| 128 | 22.3 GB | 108.9 ms | 18.1 ms | 127.5 ms | 1,004 | 7.8 | 1,560 | - |
| 256 | 25.1 GB | 122.6 ms | 31.6 ms | 154.7 ms | 1,655 | 6.5 | 2,780 | - |

The bytes and the ceiling are from the batch roofline in
`docs/GLM5_NEXT_ROOFLINE.md`. The measured B1 value is the 2026-09-28 run
above: `dd3526b` engines with `G5_GRAPH_PATH=1`, `G5_PIN_EXPERTS=1` and
`SPARK_TP_WAIT_MODE=hardware`, 36 tok/s at 23 to 24.5 ms per step of 8-step
graph chains. Before that the fleet measured 27.6-29.8 tok/s after PR #1208
(2026-09-25). The measured B8 value is the 8-stream load reported with #1228
(2026-09-26), receipted only in the PR thread; the committed re-run's best is
125.8 tok/s. The current breakdown is "Where it stands" in
[`docs/GLM5_NEXT_ROOFLINE.md`](docs/GLM5_NEXT_ROOFLINE.md).

At 70% or 80% of bandwidth the projections move by about 5% either way: 66 to
73 tok/s at B1, and 1,570 to 1,740 at B256. The M1 and M2 exit criteria in
[`docs/ROADMAP.md`](docs/ROADMAP.md) (100, 300, 750 and 2,200 tok/s) are 80% of
the ceiling for the whole step, collectives included, so they sit above this
projection. The site rounds it to 70, 240, 620, 1,000 and 1,650.

## Target gates

These are architecture requirements, not measured results:

| Gate | Target |
| --- | ---: |
| Promote a configured nonresident model to ready | at most 60 s |
| Read model shards from the external pooled tier | at least 20 Gb/s useful |
| Internal hot KV allocation | 2.5 TB per Spark |
| Internal active model-shard allocation | 1.0 TB per Spark |
| External direct model tier | at least 1.0 TB per Spark |
| Mixed fleet: Spark prefill with Mac Studio decode | faster end to end than either pool alone |

Promotion timing includes shard access, verification, rank-local placement,
driver and communicator binding, prewarm, all-rank agreement, and atomic ready
publication. A copy-only storage benchmark does not close the model-promotion
gate.

The mixed-fleet comparison runs the same checkpoint, precision, request shape,
context, output length, batching policy and timing boundary on the Sparks
alone, the Studios alone and the mixed fleet. No analytical bandwidth ratio
closes that gate.
