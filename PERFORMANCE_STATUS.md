# SparkPipe Performance Status

This file is the only current performance ledger. It states what the served
builds measure today, how each figure is measured, and where its receipt is.
Projections are kept in a separate section and never count as achieved
results. A number without a committed receipt names its source. The dated
holds behind these numbers are in
[`docs/archive/PERFORMANCE_HISTORY_2026-10.md`](docs/archive/PERFORMANCE_HISTORY_2026-10.md)
and [`docs/archive/PERFORMANCE_HISTORY_2026-08.md`](docs/archive/PERFORMANCE_HISTORY_2026-08.md).

All figures: sixteen DGX Spark nodes at TP16, GPU clocks capped at
2200 MHz, current as of 2026-10-18.

## The fleet

| | |
| --- | --- |
| Nodes | 16 NVIDIA DGX Spark (GB10), 128 GB unified memory each: 2 TB in all |
| Memory bandwidth | 273 GB/s per node by specification; 243-247 GB/s measured streaming read |
| Compute | about 210 TFLOPS FP8 and 90 TFLOPS BF16 per node (measured cuBLASLt peaks) |
| Network | per node, one 100 Gb/s switch port and one direct link to a pair partner (7 pairs); 98-109 Gb/s useful per link |
| Collective latency | 16-rank all-gather 16 µs; 7 KB all-reduce 33-38 µs |
| Hub | one x86 host with an RTX 5090: API, tokenizer and the draft model |

The collective figures are from `mesh_round_bench` on PR #1490 (`af0cc3f8d`,
2026-10-08); no receipt is committed.

## Core models

Kimi K3, GLM 5.3 Full and GLM 5.3 Flash get the performance work.

### Decode without speculation

| | Kimi K3 | GLM 5.3 Full | GLM 5.3 Flash |
| --- | ---: | ---: | ---: |
| B1 | 16.2 tok/s | 28.9 tok/s | 55.0 tok/s |
| B1 at long context | 15.1 tok/s at 73K | 24.1 tok/s at 197K | 46 tok/s at 71K |
| B8, all streams | 63.7-63.9 tok/s | 73.2-74.2 tok/s | 173.7-176.1 tok/s |
| B16, all streams | not served | not served | 225.8-227.1 tok/s |
| Serving cap | 8 sequences | 8 sequences | 16 sequences |

- B1 is one request, greedy, 256 tokens (`b1_o256`). K3 is the median of 9
  runs on 2026-10-14
  ([`kimi-k3.quiet-20261014.perf.jsonl`](qualification/releases/1d89e0377/kimi-k3.quiet-20261014.perf.jsonl));
  the GLM models are medians of 3 in hold BW.
- B8 and B16 are the aggregate rate over the window in which every stream is
  between its first and last token (8 or 16 streams of 512 tokens, two runs
  each; receipts in
  [`qualification/batch-sweep-20261018/`](qualification/batch-sweep-20261018/)).
  K3 ran with lookup on and its model drafts off: accepted drafts were at
  most 3.1% of its tokens. The GLM models proposed no drafts.
- Long context: K3 at 72.9K, plain decode (65-67 ms per chained step); GLM
  5.3 Full at 197K; GLM 5.3 Flash at 71K.
- Earlier builds ran larger batches: GLM 5.3 Flash B32 230, B64 290 and
  B128 347 tok/s (with B64 token-identical across runs), and GLM 5.3 Full
  B16 109-116 tok/s by step time, on the modules the common runner replaced.

### Prefill

| | Kimi K3 | GLM 5.3 Full | GLM 5.3 Flash |
| --- | ---: | ---: | ---: |
| 8K prompt | 590 tok/s | 1,162 tok/s | 876 tok/s |
| Long prompt | 137.7K in 299 s (460 tok/s) | 197K at about 680 tok/s | 71K at 736 tok/s |
| Of the compute floor | 3.7% | 2.8% | 0.9% |

GLM 5.3 Full's 8K prefill takes 6.9 s against a network floor of about 2.9 s:
42% of the floor that binds first at TP16.

### Floors

Two floors bound decode:

- **The ideal memory floor:** each rank reads 1/16 of the model's active
  weights once per token at 273 GB/s. It leaves out replication, cache and
  state.
- **The floor for the bytes a build reads:** the bytes per rank per token of
  the deployed pack, counted from its kernel shapes, at the measured 243-247
  GB/s.

| | Kimi K3 | GLM 5.3 Full | GLM 5.3 Flash |
| --- | ---: | ---: | ---: |
| Ideal floor | 32 tok/s (8.56 GB per rank) | 109 tok/s (2.51 GB) | 243 tok/s (1.12 GB) |
| B1 of the ideal floor | 51% | 27% | 23% |
| Floor for the bytes read | 27.7 tok/s (8.9 GB at 247 GB/s) | about 59 tok/s (about 4.1 GB) | 94 tok/s (2.58 GB at 243 GB/s) |
| B1 of that floor | 58% | about 50% | 59% |

K3 reads within 4% of its ideal bytes. GLM 5.3 Full and Flash read 1.6-2.3
times theirs; the difference is tensors every rank still replicates, plus
cache and state reads. The compute floor for prefill is 16
nodes at the format's peak over 2 FLOPs per active parameter per token:
16,123 tok/s for K3, 41,791 for GLM 5.3 Full and 93,333 for GLM 5.3 Flash.

### Speculation

Two draft sources serve, and the target model verifies every draft. Lookup
copies drafts from the request's own context by suffix match and verifies up
to 255 of them in one frame (K3 and GLM 5.3 Full). Model drafts come from a
DSpark block drafter on the hub's RTX 5090, fed by hidden states from five
K3 layers, when lookup has no match of at least 8 tokens (K3).

| Workload, B1 | Kimi K3 | GLM 5.3 Full |
| --- | --- | --- |
| Whole-file code rewrite | 15.3 → 126.4 tok/s (8.3×); best 140.6 | 27.6 → 124.2 tok/s (4.5×); 232.9 with uncapped frames, best 238.7 |
| SEARCH/REPLACE edit | 15.0 → 28.0 tok/s (1.9×) | 27.5 → 36.0 tok/s (1.3×) |
| Short edits, model drafts added to lookup | 1.39× per token sampled, 1.6× greedy | not ported |
| Reasoning prose at 137.7K, model drafts added | 1.07× | not ported |
| Prose, lookup only | 1.16× greedy, 1.0× sampled | about 1.0× |
| Greedy story prompt | 26.4 tok/s with both sources, 21.2 with lookup alone | no drafts found |
| Whole-file rewrite, long prompt | 76.8 tok/s at 73K, 41.8 at 337K | 105 tok/s at 46K, 86.2 at 198K |

Ratios are against no speculation in the same hold, except the greedy story
prompt (baseline 16.2 tok/s, 2026-10-14) and GLM 5.3 Full's 232.9 (baseline
from an earlier hold). Receipts:
[`qualification/k3/spec-long-frames-20261015/`](qualification/k3/spec-long-frames-20261015/),
the GLM 5.3 Full speculation section of the
[history](docs/archive/PERFORMANCE_HISTORY_2026-10.md), and
[`qualification/batch-sweep-20261018/`](qualification/batch-sweep-20261018/).
At B8, K3's model drafts change nothing measurable (62.5-64.8 against
63.7-63.9 tok/s).

## All served models

| Served name | Context | Decode, B1 | Of ideal floor | Prefill, 7-8K prompt | COMPSEC-17 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `glm-5.3` | 256K | 28.9 tok/s | 27% | 1,162 tok/s | 15 |
| `glm-5.3-flash` | 128K | 55.0 tok/s | 23% | 876 tok/s | 13 |
| `kimi-k3` | 720K | 16.2 tok/s | 51% | 590 tok/s | 17 |
| `deepseek-v4.1-flash` | 16K | 29.5 tok/s | – | 182 tok/s | 16 |
| `mimo-v2.6-pro` | 32K | 19.0 tok/s † | – | 55 tok/s † | 17 |
| `mimo-v2.6-flash` | 32K | 40.9 tok/s † | – | 157 tok/s † | 16 |
| `hy4-preview` | 32K | 14.2 tok/s | 16% | 53 tok/s | 17 |
| `gemma-4-31b` | 32K | 34.4 tok/s | 48% | 834 tok/s | 16 |
| `gemma-4-26b-a4b` | 32K | 89.5 tok/s | 16% | 1,848 tok/s | 16 |
| `ling-3.0-flash` | 32K | 80.4 tok/s | 19% | 1,514 tok/s | 13 |
| `ling-3.0-flash-fin` | 32K | 79.2 tok/s | 18% | 1,550 tok/s | 13 |
| `laguna-s-2.1` | 32K | 51.0 tok/s | – | 773 tok/s | 13 |
| `muse-glimmer-30b` | 32K | 37.9 tok/s | 51% | 1,186 tok/s | 17 |
| `qwen3.8-max` | 32K | 10.4 tok/s | – | 35 tok/s | 5 |
| `qwen3.8-27b` | 256K | 57.4 tok/s | 35% | 1,852 tok/s | 14 |
| `qwen3.8-flash` | 32K | 45.0 tok/s | – | 358 tok/s | 15 |

- Non-core rows are the release `91c0e6be3` qualification on a quiet fleet
  ([`qualification/releases/91c0e6be3/`](qualification/releases/91c0e6be3/)),
  except DeepSeek V4.1 Flash: 29.5 tok/s and 182 tok/s prefill after its
  shared-expert split on 2026-10-13.
- † MiMo was measured only under a CPU tenant that kept 16 cores busy per
  node ([`qualification/releases/1d89e0377/`](qualification/releases/1d89e0377/)),
  so its numbers are lower bounds; that load cut K3 from 16.2 to 10.0 tok/s.
- COMPSEC-17 is the release qualification's 17-question security set. Qwen
  3.8 Max always thinks and ran with a 2,048-token budget.
- No floor is shown where the split of active parameters between formats is
  not verified (DeepSeek V4.1 Flash, the Qwen 3.8 MoE models) or not
  computed (MiMo, Laguna). The Qwen models are internal only, by their
  license.

Ideal floors, from the active parameters each rank reads per token (K3:
104.2B active, of which 48.6B are routed experts, per its model contract):

| Model | Formats | Non-expert params | Active expert params | GB per rank per token | Ideal floor |
| --- | --- | ---: | ---: | ---: | ---: |
| `glm-5.3` | FP8 | 17.5B × 1 B | 22.7B × 1 B | 2.51 | 109 tok/s |
| `glm-5.3-flash` | FP8 | 7B × 1 B | 11B × 1 B | 1.12 | 243 tok/s |
| `kimi-k3` | BF16 spine, MXFP4 experts | 55.6B × 2 B | 48.6B × 0.53 B | 8.56 | 32 tok/s |
| `hy4-preview` | MXFP8 | 24B × 1.03 B | 25B × 1.03 B | 3.15 | 87 tok/s |
| `gemma-4-31b` | BF16, dense | 30.7B × 2 B | – | 3.84 | 71 tok/s |
| `gemma-4-26b-a4b` | BF16 | 1.9B × 2 B | 1.9B × 2 B | 0.47 | 575 tok/s |
| `ling-3.0-flash` | BF16 | 2.6B × 2 B | 2.5B × 2 B | 0.64 | 428 tok/s |
| `muse-glimmer-30b` | BF16, dense | 29.2B × 2 B | – | 3.65 | 75 tok/s |
| `qwen3.8-27b` | FP8, dense | 27B × 1 B | – | 1.69 | 162 tok/s |

## Accuracy

- **Release gate:** every release reproduces the previous build's tokens on
  six fixed prompts for all 16 models (last: `1d89e0377`).
- **ds4_eval, 92 questions** (GPQA Diamond, SuperGPQA, AIME 2025, COMPSEC),
  served by SparkPipe: Qwen 3.8 Max 82/92 and Qwen 3.8 Flash 79/92 (greedy,
  16,000-token budget;
  [`qualification/ds4_eval/runs/`](qualification/ds4_eval/runs/)). The core
  models run the 92 once their prefill and decode reach 50% of roofline. For
  reference, Kimi K3 on Kimi's own API scores 81/92 (sampled).
- **Number formats** are accepted only by a teacher-forced A/B against a
  same-precision floor. GLM 5.3 Flash FP8 experts against BF16: KL 0.0087,
  96.8% top-1 agreement, no NLL change: accepted
  ([`qualification/ab/flash-w1/REPORT.md`](qualification/ab/flash-w1/REPORT.md)).
  K3 attention with a single bf16 probability matrix: KL 0.0042 against
  floors of 0.0013-0.0037 and NLL change −0.0009 [−0.0046, +0.0038] nats over
  1,713 positions up to 137K context: no measurable loss.

## KV cache and long context

| | |
| --- | --- |
| Sharding | each node holds 1/16 of a sequence's MLA latent, split by context |
| K3 window | 737,280 tokens on a 1.36 GB KV pool per node; measured end to end to 337K |
| Pages | lanes share one physical page pool; pages beyond it park on NVMe |
| K3 follow-up turn, cached prefix | 4.7-5.1 s at 137.7K (299-316 s cold); 8.0 s for 1,872 new tokens at 337K (982 s cold) |
| Prefix restored after a model swap | 137,600 tokens in 2.7 s |
| Short chat turn, cached | 0.3-0.9 s (6.6 s cold) |

Limits today: K3's recurrent-state store fills after about 977 checkpointed
requests in one conversation, after which the request fails, and the
snapshot store's chain links grow with the square of the chain length.

## How the numbers are measured

- Tool: [`tools/api_serving_perf.py`](tools/api_serving_perf.py) against each
  model's engine endpoint. `b1_o256`: one greedy request, 256 tokens,
  decode rate after the first token. `ttft6000`: one cold prompt of about
  7,400 tokens, prompt tokens over time to first token. `streamsNxM`: N
  concurrent requests of M tokens, aggregate over the window in which every
  stream is decoding.
- Holds: one exclusive fleet lock per measurement, quiet fleet, arms
  interleaved, the KV snapshot tier wiped between arms of an A/B. B1 varies
  5-10% between holds, so ratios come from one hold.
- Speculation is always reported separately from plain decode, per workload.

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
| 1 | 2.18 GB | 10.7 ms | 3.2 ms | 14.4 ms | 70 | 70 | 125 | 55.0 |
| 8 | 5.6 GB | 27.4 ms | 5.4 ms | 33.3 ms | 240 | 30 | 390 | 174-176 |
| 64 | 18.6 GB | 90.8 ms | 11.4 ms | 102.7 ms | 623 | 9.7 | 940 | - |
| 128 | 22.3 GB | 108.9 ms | 18.1 ms | 127.5 ms | 1,004 | 7.8 | 1,560 | - |
| 256 | 25.1 GB | 122.6 ms | 31.6 ms | 154.7 ms | 1,655 | 6.5 | 2,780 | - |

The bytes and the ceiling are from the batch roofline in
`docs/GLM5_NEXT_ROOFLINE.md`. The measured column is the current build
(hold BW, 2026-10-18, below); B16 measures 226-227 tok/s.

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
