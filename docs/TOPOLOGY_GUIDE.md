# Topology Guide

Why TP4×PP4 was the first candidate, what the alternatives cost, and how to
hill-climb to the right topology per model. Written 2026-08-28 after the
pack-agent fleet made multi-topology packs practical.

Updated 2026-09-28. GLM 5.3 Flash production runs TP16; its anchor is below.
No topology is settled for any model. The deliverables are tok/s and latency,
so hill-climb TP16, PP16, TP4×PP4 and the hybrids on measured numbers, B1
through the largest batch that still scales.

## The "center": TP4 × PP4

**16 sparks, 4 pipeline stages × 4 tensor-parallel ranks.** The first
candidate tried for large models (DSV4 Pro, Qwen Max, K3). It is not a
default: the placement that measures best on tok/s and latency serves.

Why it was the center:
- TP4 divides every supported model's dimensions cleanly (heads, experts,
  FFN intermediate all divide by 4).
- PP4 splits layers into 4 stages — small enough that pipeline bubbles are
  amortized at any batch, large enough that per-stage packs fit a single
  spark's memory for every model we support.
- KV cache is sharded 4× by TP within each stage → each node holds ~1/16
  of the total KV (the sharding invariant).
- Measured aggregate: ~3.5× single-spark throughput and 3.5× capacity. Not
  the ultimate, but not horrible.

## The axes

### TP (tensor parallelism) — splits each weight matrix across ranks
- **Cost:** an all-reduce per layer (2 per layer with the gated residual).
  All-reduce data = batch_rows × hidden_dim × 2 bytes per rank.
- **Scaling:** TP4→TP8 halves per-rank weight traffic but doubles the
  all-reduce participant count. In the 08-28 projections, TP8→TP16 showed
  little gain (docs/archive/QWEN38_MAX_PERF.md: TP16 B=16 ≈ 190 tok/s
  projected vs TP8's similar aggregate) because the all-reduce grows linearly
  with rank count while per-rank compute halves. The measured GLM 5.3 Flash
  TP16 anchor is in the tables below.
- **When to raise:** memory-bound single-spark models that don't fit, or
  when the model's MoE expert count divides better (K3's 896 experts = 8×112).

### PP (pipeline parallelism) — splits layers across stages
- **Cost:** pipeline bubbles. With B requests in flight and S stages, the
  bubble fraction is roughly (S-1)/(B+S-1). At B=16, PP4 bubbles ≈ 19%;
  at PP8 ≈ 35%; at PP16 ≈ 60% unless B ≥ 16k microbatches.
- **Scaling:** PP reduces per-stage memory linearly and adds no
  communication within a step (only activation sends between stages).
- **When to raise:** model too big for fewer stages; B1024 throughput
  (needs PP16 = 16k microbatches to keep bubbles low).
- **PP2×TP8:** splits the bubble cost between fewer stages and wider TP.
  Interesting untested hybrid for 2-node-pair deployments.

### EP (expert parallelism) — splits MoE experts across ranks
- **Cost:** all-to-all dispatch/combine per MoE layer (tokens → their
  top-k experts' ranks). Data = active tokens × top-k × hidden_dim.
- **When:** expert count is huge relative to TP width (K3: 896 experts —
  TP16 gives 56/rank with EP, vs 224/rank with pure TP).
- **Status:** the module ABI has expert_weight_codec + the MoE kernels
  support grouped experts; EP dispatch is NOT yet wired in the serving
  adapter. This is real engineering, not a config flip.

### DP (data parallelism) — replicates the whole model, splits requests
- **Cost:** none within a replica; the cost is N× the weight memory.
- **When:** model fits on one spark AND throughput scales with replicas
  (embarrassingly parallel). For the 27B (28.5 GB FP8), a single spark
  holds one replica; two sparks = 2 replicas = ~2× aggregate.
- **Status:** the residentd already runs one deployment per spark; DP is
  just "launch N independent deployments + a request router". The router
  (LiteLLM → SparkPipe router) is the missing piece.

## Measured anchors (GB10, sm_121a)

| Config | Metric | Value | Source |
|---|---|---|---|
| 27B TP1 | B1 spec decode | 24.5 tok/s | spark2 prod, d7f79880 |
| 27B TP1 | B1 no-spec | 7.7–8.03 tok/s | spark2 prod |
| GLM 5.3 Flash TP16 | B1 no-spec | 36 tok/s; 23-24.5 ms/token in 8-step graph chains, all 12096 experts pinned | lead-dev measurement, 2026-09-28 |
| GLM 5.3 Flash TP16 | B1 memory roofline | 7.99 ms/token (2181 MB per rank at 273 GB/s) | GLM5_NEXT_ROOFLINE.md |
| GLM 5.3 Flash, 4 Sparks, public best | B1 no-spec | 23.2 tok/s | lead-dev facts, 2026-09-28 |
| GLM 5.2 TP8 (model deprecated) | B16 aggregate | 75.55 tok/s | PERFORMANCE_STATUS (pre-audit) |
| Qwen Max TP4xPP4 | B1 | 1.29 tok/s | measured anchors |
| Qwen Max TP4xPP4 | B256 aggregate | ~39 tok/s | measured anchors |
| K3 single-stage | B1 | 18.0 tok/s | measured |
| 27B prefill p256×B16 | PFR=8 | 189s wall | spark3 events, 2026-08-28 |

## All-reduce reality check

The 08-28 derivation of the TP8→TP16 "no gain" report: at TP16, each all-reduce moves
batch × hidden × 2 bytes × (ranks-1)/ranks per rank. For B=8, hidden=8192:
8 × 8192 × 2 × 15/16 ≈ 123 KB per rank per reduce. With 2 reduces per
layer × 92 layers = 184 reduces/step ≈ 22.6 MB of all-reduce traffic per
token per rank. At ~100 Gbps line rate (12.5 GB/s) with recursive
doubling (log2(16)=4 hops), the wire time is microseconds — BUT the
kernel-launch + sync overhead per reduce is ~20-50 µs, so 184 reduces ≈
4-9 ms per step JUST in launch overhead. That overhead, not bandwidth,
is why TP16 didn't beat TP8.

**Fix direction:** frame graphs should also capture the all-reduce
launches (they capture kernels; the collective path's device kernels are
already in the capture scope for the hidden transport backend). If the
collectives get graph-captured with everything else, TP16's launch
overhead disappears and the bandwidth term (~2 µs/reduce) becomes noise.

**Measured since (2026-09-28).** With CUDA-graph chains and hardware waits
(the fleet drop-in in [FLEET_RELEASE_RUNBOOK.md](FLEET_RELEASE_RUNBOOK.md)),
GLM 5.3 Flash TP16 runs 36 tok/s at B1, 23-24.5 ms/token against a 7.99 ms
roofline. Collective latency is the dominant cost (lead-dev facts): an
isolated 8 KiB 16-rank all-reduce through the weightd mesh measures 167 µs
p50 ([GLM5_NEXT_ROOFLINE.md](GLM5_NEXT_ROOFLINE.md)), far above the wire term.

## Hill-climbing procedure

Per model, after the TP4×PP4 baseline:

1. **Measure the baseline**: B1, B8, B64, B256 at TP4×PP4. Record
   prefill tok/s, decode tok/s, aggregate, and the per-phase GPU times
   (the profile counters).
2. **Try TP±4**: TP8×PP2 (if PP4 stage packs fit 2-wide), or TP2×PP8.
   Same 16 sparks, different split. The recipe compiler emits the packs.
3. **Try EP for MoE-heavy models** (K3, Qwen Max): if expert count / TP
   width > 100, EP likely beats pure TP on the MoE layers.
4. **Try DP for small models**: 27B TP1 with 2-4 replicas + request
   router. Compare aggregate vs TP4's aggregate.
5. **Try PP16 for B1024**: only when the batch scheduler can sustain
   16k microbatches. Otherwise PP8×TP2.
6. **The right topology may be dynamic**: light queue → TP1 replicas
   (DP); heavy queue → TP4×PP4. The router switches by queue depth.
   This is the "compute island" model from the biz plan.

## Rule of thumb (first guess per model)

| Model | Params | First guess | Why |
|---|---|---|---|
| 27B dense FP8 | 28.5 GB | TP1 (single spark) | fits, no comm cost |
| GLM 5.3 Flash FP8 | 328.3 GB checkpoint / 20.45 GB per rank at TP16 (contract `source`) | TP16 (production); climb PP16 and TP4×PP4 for aggregate | 45 layers (34 KDA, 11 DSA), 288 routed experts, top-8; 42 MoE layers (45 − 3 dense; the contract's `expert_layer_count` 43 adds the MTP layer), 288 × 42 = 12096 experts (arithmetic) |
| DSV4 Flash FP8+MXFP4 | ~149 GB / ~37 GB/rank | TP4 (single stage) or TP4×PP2 | 43 layers, fits |
| DSV4 Pro FP8+MXFP4 | 832 GB / ~52 GB/rank | TP4×PP4 | 61 layers ÷ 4 stages |
| Qwen Max FP8 | 2.3 TB / ~144 GB/rank | TP4×PP4 minimum, TP8×PP2 to test | 92 layers, huge MoE |
| K3 MXFP4 | 1.5 TB / ~94 GB/rank TP16 | TP4×PP4 or TP8×PP2 | 93 layers, 896 experts (EP candidate) |

## Dense model topology: TP16 vs DP (the 27B case study)

For DENSE models (no MoE), the math is different from MoE models.
Every token streams the same weights — batching amortizes but doesn't
eliminate the weight traffic.

| Configuration | Single-stream | Aggregate | Aggregate spec |
|---|---|---|---|
| TP16 (1 instance, 16 sparks) | 130 tok/s (16×) | 2,351 tok/s | 8,229 tok/s |
| TP1 (1 spark) | 8.8 tok/s | 926 tok/s | 3,241 tok/s |
| 16× TP1 replicas (DP) | 8.8 tok/s each | **14,815 tok/s** | **51,852 tok/s** |

**TP16 is a latency play. DP is a throughput play.**

- TP16: 16× faster per token (each rank streams 1/16 of weights → 7 ms
  instead of 114 ms). Single-stream spec: 455 tok/s. But the all-reduce
  scales with batch size and caps aggregate at ~2,351 tok/s (reached at
  B=17, where all-reduce wire time = weight stream time).

- DP (16× TP1 replicas): each spark runs an independent deployment.
  Aggregate = 16 × 926 = 14,815 tok/s. No all-reduce. The request
  router (LiteLLM → SparkPipe router) distributes pending requests.

- The compute-bound crossover B*=106 is the same for both (TP divides
  weights AND compute by the same factor). Below B*: memory-bound.
  Above: compute-bound. The aggregate plateau is the same per-GPU-group.

**The right deployment**: some sparks run TP16 for interactive/low-latency,
the rest run TP1 replicas for batch throughput. The router switches by
request class. This IS the compute-island model.

### Why TP16 showed no gain historically

This is the 08-28 reasoning; the 2026-09-28 measurement is at the end of
"All-reduce reality check".

128 all-reduce launches per decode step × 20-50 µs launch overhead =
2.6-6.4 ms per step. At TP16, the weight stream is 7.1 ms — so the
launch overhead ADDS 90% on top, erasing the 16× gain. With graph-
captured collectives (0.1 ms), TP16 delivers its theoretical 16×.

The fix is the same one as the MoE expert-grouping: capture the
collectives into the frame graph. The collective kernels are already
in the capture scope for the hidden transport backend.

## The compute-bound knee B*

For a dense model there is a batch size B* where weight streaming time equals
compute time. Below B* a step is memory-bound; above it, compute-bound. A step
streams `params × bytes_per_param / memory_BW` and computes
`B × 2 × params / compute_rate`, so (arithmetic):

```
B* = compute_rate × bytes_per_param / (2 × memory_BW)
```

The 08-28 version divided by `bytes_per_param`; the two agree only at one
byte per parameter.

GB10 memory is LPDDR5X at 273 GB/s, the figure
[GLM5_NEXT_ROOFLINE.md](GLM5_NEXT_ROOFLINE.md) uses; 242-247.5 GB/s was
measured on the fleet (09-15 record, archived `FLEET_RUNBOOK.md`). The compute
rate is not established here. The 08-28 version assumed 50 TFLOPS at FP8
without a source, and EXO rates a Spark at about 100 FP16 TFLOPS
([HARDWARE_TOPOLOGY.md](HARDWARE_TOPOLOGY.md#mac-studio-pool)). The two
assumptions give (arithmetic):

| Assumption | B* on GB10 |
|---|---|
| 50 TFLOPS at FP8, 1 byte per parameter | 50e12 × 1 / (2 × 273e9) ≈ 92 |
| 100 TFLOPS at FP16 or BF16, 2 bytes per parameter | 100e12 × 2 / (2 × 273e9) ≈ 366 |

When compute rate doubles each time bytes per parameter halve, B* does not
depend on precision. For an MoE model a step streams every expert the batch
touches, and that set grows with B until every expert is touched, so the knee
in B lies above the dense B*. Measure it on the fleet: sweep B until tok/s
stops scaling.

**The maximum sustainable aggregate throughput** at B >> B*:

```
max_tok_s = total_compute_rate / (2 × active_params_per_token)
```

For MoE models, use ACTIVE params (top_k × expert_size + attention),
not total params — the whole point of MoE is that most experts are idle.
The table is arithmetic from the 08-28 compute assumptions (50 TFLOPS per
GB10 at FP8, 25 at MXFP4 for K3), not a measurement.

| Model | Active params/token | FLOPs/token | Max aggregate (16× GB10) |
|---|---|---|---|
| Qwen 27B (dense) FP8 | 27B (all active) | 54 GFLOP | 14,815 tok/s |
| K3 MXFP4 (896 experts, top-16) | ~144B active | 288 GFLOP | 1,389 tok/s |
| Qwen Max FP8 (512+1 experts, top-10) | ~60B active | 120 GFLOP | 6,667 tok/s |

ACTIVE params = attention + router + top_k experts + shared experts per
token. NOT total params. K3 has 1.5T total params but only ~144B active
per token — still 5.3× more than the 27B dense, and at MXFP4 half-rate
compute, giving 10.6× more FLOPs per token than the 27B.

**CORRECTION (2026-08-28):** the previously published 1.9M tok/s figure
for K3 was an arithmetic error (confused total and active params, and
used the wrong FLOPs divisor). The correct max aggregate for K3 TP16 is
1,389 tok/s — slower than the 27B dense, because K3 trades compute for
capacity (more knowledge per model, more compute per token).

**For inference services**: running at B < B* wastes hardware. Most
production deployments run at B=1-8, far below B*. The expert-grouped
continuous batching (or weight amortization for dense models) to reach
B > B* is the difference between 3% and 90%+ hardware utilization.

**For SparkPipe**: this is the core innovation for the MoE models. The
pack agents, the route counting sort, and the grouped GEMM already exist.
The batch engine's continuous-batching admission is the remaining work.
