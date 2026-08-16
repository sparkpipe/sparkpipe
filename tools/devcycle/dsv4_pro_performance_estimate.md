# DSV4 Pro GA (0813) TP4xPP4 performance estimate

Analytical estimate for the 16-spark ring (GB10, sm_121a, 1 GPU/host,
MXFP4-E2M1 experts, FP8 non-expert weights, BF16 activations/KV, 100G RDMA
all-to-all + 200G paired rails). Anchors: the GB10 memory bandwidth the
kernel library prices against (273 GB/s, model-families/common/.../
spark_lm_kernels.cuh MXFP4 comment), and the Flash TP4 control measurement
(40.46 tok/s, h4096/l43/e256) as the scaling-law cross-check.

## Weight traffic (the binding constraint)

Per token per layer, TP4 (per-rank reads):

| Component | Bytes | Note |
| --- | --- | --- |
| Top-6 routed experts (w1+w2+w3, MXFP4 0.5 B/elem + E8M0 scales) | ~198 MB | 6 x 33 MB; the router changes the set every token, so no cross-token reuse |
| Attention q/kv/o projections (FP8, rank-sharded) | ~75 MB | wq_b 25 MB + wo_a/b 46 MB dominate |
| Shared experts + gate + HC + compressor/indexer | ~30 MB | |
| **Total per layer** | **~303 MB** | |
| **Total per token (61 layers)** | **~18.5 GB** | |

At 273 GB/s effective DRAM bandwidth: **~68 ms/token** weight streaming.
The cross-check: Flash at the same per-layer structure reads ~6.9 GB/token
(h4096/l43/e256 scales everything down ~2.7x) -> ~25 ms/token -> 40 tok/s,
matching the measured 40.46 tok/s control. The model therefore scales the
Pro estimate the same way.

Non-weight overheads per token: 122 TP all-reduces (14 KB each, mapped-host
zero-copy + progress thread) ~2.5-5 ms; 3 PP boundary hops ~0.2 ms; control
plane ~1-2 ms. Total ~4-7 ms — second order vs 68 ms.

## Decode (output) estimate

- **Main-model-only decode: ~68-75 ms/token -> 13-15 tok/s.** This is what
  the first GA ring run will measure (the DSpark draft execution is not yet
  wired). The O128 benchmark's 128 output tokens -> ~9-10 s of decode.
- **With the DSpark speculative stage (the GA's design):** the draft block
  costs ~0.5-1 ms (3 draft layers x ~250 MB reads amortized over 5 draft
  tokens + one head pass), so each main step emits 1 + 5 x acceptance
  tokens. At 50-70% acceptance (typical for draft depth 5): effective
  ~3.5-4.5 tokens/step -> **~45-60 tok/s**, roughly the GA's intended
  regime.

## Prefill estimate (128-row submission, the shipped batch)

Weights amortize across the 128 rows. The router's per-row top-6 union over
128 rows covers ~330 of 384 experts, so the expert reads grow to
~1.8 GB/layer but serve all 128 rows:

- Expert weights: ~110 GB total batch
- Dense weights: ~6.4 GB total batch
- **~117 GB / 273 GB/s ~ 430 ms**, plus 122 collectives of 1.8 MB
  (~7 ms), attention waves (~1-2 ms), pipeline fill (~10-20 ms)

**Prefill: ~0.45-0.6 s for the 128-token prompt -> ~210-280 tok/s
prefill throughput; TTFT ~0.5-0.7 s.**

## What moves these numbers

1. DSpark acceptance (biggest): 0.5 -> 0.7 acceptance doubles the decode
   gain (13 -> 45-60+ tok/s).
2. FP8-expert variant: doubles expert DRAM traffic (accuracy knob, NOT a
   speedup - confirmed by the codec audit).
3. FP8 KV: shrinks cache reads, ~neutral on the weight-bound path.
4. Nothing else matters much until the weight streaming is addressed
   (expert selection clustering / persistent expert residency in a
   larger unified-memory cache tier are the real long-term levers).

All numbers are estimates pending the first measured ring receipt
(--profile-stages + the collective profiling are wired into the runbook).
