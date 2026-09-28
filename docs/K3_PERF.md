# K3 performance record

SparkPipe's own Kimi K3 numbers live here and nowhere else. External stacks
are compared in [K3_VS_GB10_VLLM.md](K3_VS_GB10_VLLM.md), which takes
SparkPipe's side of the comparison from this file.

## Numbers

| Point | Value | Grade | Source |
| --- | --- | --- | --- |
| Warm B1 step: stage 0 (24 layers) of a TP4 rank pack on one Spark (sparka) | 55.5 ms, 2.3 ms/layer | measured 2026-08-16, one stage | `MEASURED_STAGE_MS` in `tools/k3_tp4pp4_perf_estimate.py` (a40946c) |
| TP4xPP4 pipelined throughput, one token per stage step | 18.0 tok/s | arithmetic: 1000 / 55.5 | same tool |
| One B1 sequence through the four PP stages | ~4.5 tok/s | arithmetic: 1000 / (4 x 55.5), collectives excluded | same tool |
| TP4xPP4 decode roofline, slowest stage | 48.6 ms (20.6 tok/s pipelined) | analytical | tool output |
| TP16 PP1 decode | 49.5 ms (20.2 tok/s) | analytical, 15 us per all-reduce | tool output |
| TP16 PP1 decode at GLM's measured all-reduce latency | ~78 ms (~12.9 tok/s) | arithmetic, below | this file |
| End-to-end K3 decode on the fleet | none | not measured | |

The 55.5 ms step was measured before a29ea53 (2026-09-10) moved every K3
weight behind weightd, and has not been measured since.

SparkPipe has no K3 fleet measurement. The 29 tok/s K3 point announced by
@ciprianveg on 2026-09-19 (published C1 29.81) comes from his gb10-vllm stack
(vLLM, TP16+DCP8, DSpark nst6 speculation), not from SparkPipe. It is listed
with the other external numbers in K3_VS_GB10_VLLM.md.
[PERFORMANCE_STATUS.md](../PERFORMANCE_STATUS.md#earlier-points-carried-over-from-the-performance-ledger)
carries the 55.5 ms stage step among the points taken over from the archived
performance ledger, and gb10-vllm's point in its public-comparables table.
The ledger's claim that 29 tok/s superseded SparkPipe's 18.0 was wrong: 18.0
is arithmetic from the one measured stage step.

### TP16 collective arithmetic

The estimate tool charges 15 us per 16-rank all-reduce (`tp16_ar_ms`). The
measured GLM 5.3 Flash 8 KiB 16-rank all-reduce is 167 us p50
([GLM5_NEXT_ROOFLINE.md](GLM5_NEXT_ROOFLINE.md), fleet, 2026-09-25). K3 runs
two all-reduces per layer on the B1 critical path (after attention and after
the MLP), so at that latency (arithmetic):

- 93 layers x 2 x 167 us = 31.1 ms per token, against the tool's 2.8 ms;
- 49.5 - 2.8 + 31.1 = 77.8 ms per token, ~12.9 tok/s.

This assumes K3's collective matches GLM's weightd-mesh latency; the K3 runner
uses its own `device_collective` tier (NCCL or hidden transport). K3's phase
payloads are larger than 8 KiB, so ~12.9 tok/s is optimistic. At TP16 the
collective, not weight bandwidth, is the dominant cost, as it is for GLM.

### Prefill estimate (analytical)

`tools/k3_tp4pp4_perf_estimate.py` gives TP4xPP4 prefill of 92 tok/s at B8
rising to 1537 tok/s at B1024 steady state (single-prompt latency 0.35 s to
2.66 s). The expert stream saturates at B56 (896 experts / top-16) and the
fp32 KDA state becomes the dominant term at large batch. TP16 gives
1511 tok/s at B1024 with 0.68 s latency. Both use the tool's 8 us (TP4) and
15 us (TP16) all-reduce assumptions.

## Current code state

- **Only a TP4xPP4 descriptor exists.** `K3ServingDescriptor`
  (`modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c`) is
  `k3-tp4pp4`: hybrid TP/PP, 16 stages, `parallel_group_size` 4 and PP4 stage
  layer counts. residentd derives pipeline geometry from the descriptor
  (`SparkPipelineRuntimeDeriveStageGeometry` in `runtime/pipeline_runtime.c`).
  The adapter config can set `tp_degree` 16, but without a TP16 descriptor a
  TP16 PP1 deployment is not deployable as written.
- **weightd is mandatory.** The runner fails closed when
  `SparkWeightdAttachRequested` is not granted (a29ea53: the family no longer
  opens or maps weight bytes).
- **CUDA graphs were removed.** a29ea53 deleted the capture machinery
  (`graphs[4]`, `LaunchSliceGraph`, `CAPTURE_GRAPHS`, `capture_graphs`)
  because exclusive lazy loading makes capture ineligible. The 2026-08-16
  replay result (54.2 ms vs 55.5 ms eager) no longer applies. The
  `capture_graphs` keys still emitted by `tools/k3_gen_adapter_configs.sh`,
  `tools/k3_multidev_lane.py` and `modules/k3_resident_decode_stage/configs/`
  are not read by the adapter. GLM regained graphs by pinning every expert
  (78c2c21); K3 needs the same, or relocatable graphs.
- **TILE_K=32 landed** (837fe89, 2026-08-16): `layer.cuh` takes the
  INTERLEAVED_B path when `expert_tile_k == 32u`; `tools/k3_pack.py` packs
  experts at tile_k 128 by default and 32 on request.
- **Expert sharding** (`tools/k3_shard.py`): w1 (`_expert_gate_up`) splits
  the input on whole k-tiles and keeps the gate|up output whole, because the
  layer all-reduces before the SiTU. w2 (`_expert_down`) splits the output on
  whole 16-neuron cells and keeps the input whole. The sharder refuses a
  degree the tile or cell counts do not divide.
- **Lifecycle callbacks are no-ops.** `K3ServingPrefetch`,
  `K3ServingResolvePrefetch`, `K3ServingProgress`, `K3ServingQuiesce` and
  `K3ServingReset` return success without doing anything, which
  [DRIVER_ACCEPTANCE.md](DRIVER_ACCEPTANCE.md) forbids.
- **Landed collective work** (2026-08-16): one fused, stream-ordered
  all-reduce per layer phase over the `device_collective` tier (`nccl` or
  `hidden_transport`); the host TCP tier serves at most 4 ranks
  (`SPARK_TP_COLLECTIVE_MAX_STEPS`). The phase after attention is required
  for correctness: the MLP-side AttnRes retrieval reads the post-attention
  partial.

## TP16 divisibility (degree 16)

Extents the sharder splits; the expert rows use `inference/llms/kimi_k3/config.h`
(`K3_ROUTED_EXPERT_HIDDEN` 3584, `K3_EXPERT_INTERMEDIATE` 3072).

| tensor | full | /16 | status |
| --- | --- | --- | --- |
| heads (KDA + MLA) | 96 | 6 | ok |
| vocab (embed/lm_head) | 163840 | 10240 | ok |
| kda_out, mla_out input (96 heads x 128) | 12288 | 768 | ok |
| expert w1 input k-tiles at tile_k 32 | 3584 / 32 = 112 | 7 | ok (tile_k 128 gives 28 tiles, refused) |
| expert w2 output cells | 3584 / 16 = 224 | 14 | ok, input 3072 kept whole |
| shared w1 halves | 1024 | 64 | ok |
| shared w2 input | 2048 | 128 | ok |
| dense gate_up halves | 16896 | 1056 | ok |
| dense down input | 33792 | 2112 | ok |

## Open work, in dependency order

1. A TP16 adapter descriptor, so TP16 PP1 can load.
2. Real prefetch, resolve, progress, quiesce and reset callbacks.
3. Collective latency, the dominant TP16 term (186 all-reduces per token).
   Recorded in the 2026-08-16 audit and not re-verified since: the
   slot-encoded full-width all-reduce moves 4x the minimal bytes
   (reduce-scatter plus all-gather would halve per-rank wire traffic), the
   head exchange still uses the host tier, and the hidden-transport tier
   cannot narrow its pre-registered frame.
4. Graph replay through pinned or relocatable experts.
5. A first end-to-end fleet B1 number, then the levers in
   [K3_VS_GB10_VLLM.md](K3_VS_GB10_VLLM.md).

## Fixed defects (record)

- 2026-08-17, tail nondeterminism: the KDA o_proj ran its weight-only GEMM
  with output == input, so second-wave tiles read first-wave stores (about
  14 ULP on hidden[6144..7167], across runs and across TP4 vs full stage).
  o_proj now lands in `hidden_bf16`; fresh-run determinism holds at 4 ULP and
  the TP4 4-rank sum equals the full-stage run to bf16 rounding.
- The earlier "plain BF16 GEMM breaks at K >= 512" was a test artifact: the
  probe's `(((int32_t)x % 11u) - 5)` subtracts in unsigned space and wraps
  negatives to ~4.29e9.
