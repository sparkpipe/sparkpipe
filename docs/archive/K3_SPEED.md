# Kimi K3 on the sparkring - the speed model, and the validation it rests on

## The numerical foundation (sparkdev, 2026-07-29, CPU end-to-end)
All 93 transformer layers traversed with actual weights on a coherent
1,024-token code prompt; NumPy and Torch agree on next token 646
(" The"); transformer reference errors 0.005-0.011% relative L2; final
163,840-token lm_head logits differ by 1.42e-5% relative L2. Exact
cache tests cover every layer, two-sequence batching, chunked
execution, 2,048-token KDA and 4,096-token MLA. Disclosed, not hidden:
batch-shaped BLAS rounding can flip near-tied MoE routing (focused
layer-0 difference 0.0458%, recurrent state 5.3e-7); CUDA batching
untested on the GPU-less validation host.
Ledger: k3-cpu-end-to-end-ledger.json,
SHA-256 66083b4d67a6074fc6e43679cb6e7444f7b526f7741e4be163f6a9ddbda407bb.
The math is proven. Everything below is about carrying it at speed.

## What K3 is, from inference/llms/kimi_k3/config.h
93 layers (24 MLA + 69 KDA), hidden 7168, vocab 163,840, 1M context.
LatentMoE from layer 1: router at 7168, latent 3584, 896 experts at
intermediate 3072, top-16 plus 2 shared, latent up/down projections.
MLA: 96 heads, kv-lora 512 + 64 unrotated (NoPE - position lives in
KDA decay). KDA: 96 heads x 128 x 128, fp32 state, conv-4 window.

Counted: experts 33.0M each; MLA layer ~232M attention; KDA layer
~440M; **total ~2.77T parameters**.

## The fit fact that decides everything
BF16 5.55 TB - impossible. FP8 2.77 TB - exceeds the ring's 1.66 TB.
**MXFP4 (group-32, ~4.25 bits/weight): 1.47 TB - the only resident
option on thirteen GB10s**, with ~190 GB left for KV, KDA state and
activations. K3_MXFP4_GROUP in the config is not an option, it is the
plan.

## Per-step streamed bytes
- Weights: MoE batching obeys the expert-union curve
  E[distinct] = 896(1-(1-16/896)^B): B=16 -> 224 experts, B=64 -> 614,
  B=256 -> 887 of 896 - **large batches stream essentially the whole
  pool**, so MoE sparsity helps latency, not saturated throughput.
- KDA state: fp32, 96x128x128 + conv window = 6.5 MB/layer/seq,
  read+write across 69 layers = **895 MB per sequence per token** -
  the dominant batch term, context-independent.
- MLA KV: (512+64) x 2 B x 24 layers = 27 KB per context token per
  sequence - the 1M-context story is cheap because only 24 layers grow.

## tok/s at bus saturation, 13-ring (3.55 TB/s), MXFP4 weights
| ctx    | B=1  | B=16 | B=64 | B=256 |
|--------|------|------|------|-------|
| 4k     | 68   | 141  | 210  | 530   |
| 32k    | 67   | 136  | 201  | 474   |
| 256k   | 60   | 110  | 148  | 257   |

Single-stream floor ~15 ms/token (68 tok/s). FP8 weights would halve
these if they fit; they do not.

Readings:
- The KDA fp32 state is the throughput ceiling at batch: at B=64 it
  streams ~57 GB/step against ~0.8 TB of weights - already 7% and
  growing linearly with B. **State-precision (bf16 state, 2x) is the
  single biggest K3 throughput lever after residency**, and it is a
  numerics question for sparkdev's harness, not a systems question.
- 256k context costs only ~7 GB/seq/step in MLA reads - the
  1M-context design holds up; long context is not the bottleneck,
  batch state is.
- These are ceilings under the cohort-13 pipeline keeping the bus
  busy; audit F1-F3 remain the risks to that assumption.

## Runtime readiness, established this session
- KV tier: layout-parameterized and MLA-correct - the capacity
  estimator reproduces K3's 1,152 B/token/layer from the family
  header, gated by test_hybrid_kv_arithmetic. A generality defect fell
  on the way: the validator demanded the MLA latent/rope pair from
  every layout, rejecting full-KV families with correct zeros;
  requirements are layout-conditional now.
- JIT KV prefetch: block-index machinery, model-free; a hybrid wires
  layer_count = its full-attention count (24 for K3, 16 for qwen).
- Recurrent state (KDA here, GDN for qwen): non-growing, so it is not
  a cache-tier concept at all - it is a lane-indexed arena at the
  execute rung, allocated once per lane. The convention is written
  here so K3-A implements it that way and no host machinery grows.
- Admission: the uniform-estimated profile carries K3 until measured
  tables exist (test_uniform_profile_admit).
What remains is the execute rung - and the chunk-path, phase-clock and
transport-overlap measurements the bandwidth audit already queued.
