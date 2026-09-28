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
