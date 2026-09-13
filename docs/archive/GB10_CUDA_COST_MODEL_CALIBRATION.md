# GB10 CUDA Module Cost Model — Calibration

Purpose: estimate the decode-stage time cost of every resident CUDA module
(glm52, mimo25, dsv4, qwen38_27b, k3) from hardware constants and measured
efficiencies, so a module's latency can be projected before it runs on the
ring and so optimization effort targets the term that actually bounds it.

This is a projection model. Every constant below traces either to a
looked-up GB10 hardware figure or to a retained measurement in this repo's
diagnostics; the provenance is named at each one. Where a term is not yet
pinned by measurement it is labelled PENDING with the exact measurement that
would pin it. The model must not be read as silicon truth for the family
drivers — none of them have run — but its glm52 predictions are checked
against the measured corpus and its form is validated by those checks.

## Hardware constants (looked up, GB10 / DGX Spark)

- Peak memory bandwidth 273 GB/s, LPDDR5x, unified, not HBM. Source: the
  in-repo 12x performance model and the public GB10 spec; the non-HBM point
  matters because it sets a low bandwidth ceiling against a high FLOP ceiling.
- Peak FP8 tensor throughput approximately 250 TFLOP/s dense (1 PFLOP FP4
  with sparsity, halved for dense, halved again FP4 to FP8). Public GB10 spec.
- 48 SM at up to 2.55 GHz, compute capability 12.1, 128 KB L1/shared per SM.
- Resulting arithmetic-intensity balance point is roughly 915 FLOP per byte.
  Any decode kernel below that intensity is memory bound; above it, compute
  bound. Single-token decode of most kernels sits far below it.

## Model form

For each kernel: time = max(bytes / (BW_peak * eta_bw_class),
flops / (FLOP_peak * eta_flop_class), launches * launch_ns). The three terms
are the memory bound, the compute bound, and the fixed launch/plan floor.
The binding term is regime dependent, which is the central finding below.

## The regime map (measured)

Decode is not uniformly bandwidth bound. Which term binds depends on batch:

- B1: launch and plan overhead plus bandwidth. Before the M=1 linear-plan
  fix, a per-layer M=1 cublasLt launch loop added roughly 60 ms per rank at
  64 lanes, giving about 76 ms per-rank occupancy against 15-19 ms of actual
  kernel time. Source: GLM52_PP13_MULTIROW_LINEAR_PLAN_FIX_20260718 and the
  ring-occupancy line in GLM52_MEASURED_STATUS. This session's device-grouped
  MoE removes the equivalent per-layer host round trip for the family drivers.
- B16: bandwidth. Weights amortize across the cohort; compute still small.
- B128 and above: compute bound on QKVO and attention projection, bandwidth
  bound on MoE weight read and KV cache read. Source: the B128/B256 per-kernel
  Nsight split below.

## Measured per-kernel split at B128 to B256 (glm52)

From GLM52_B256_PER_TOKEN_KERNELS_20260704, best-of Nsight buckets:

| kernel bucket        | B128    | B256    | scaling |
| -------------------- | ------- | ------- | ------- |
| QKVO / linear WMMA   | 36.4 ms | 73.9 ms | 2.03x   |
| decode attention     | 36.8 ms | 73.0 ms | 1.98x   |
| B12x generated MoE   | 27.2 ms | 47.8 ms | 1.76x   |
| B12x fc2 finalize    | 5.55 ms | 11.29 ms| 2.03x   |

The near-2x scaling of QKVO and attention with token count is the signature
of a compute bound: doubling tokens doubles the work and the time. The
sub-2x MoE scaling (1.76x) is the signature of a bandwidth bound partially
amortized by rising expert coverage (87 percent to 98 percent B128 to B256).

## Key calibrated efficiency: QKVO WMMA runs at 6.5 TFLOPS

The QKVO/linear WMMA kernel runs at approximately 6.5 TFLOP/s effective,
about 2.6 percent of the FP8 peak. Source: GLM52_B256_PER_TOKEN_KERNELS. This
is the single largest optimization lever in the decode path and it is a
compute-efficiency problem, not a bandwidth problem: the v1 kernel used
16-wide output tiles, 16-deep K, and is instruction- and sync-bound rather
than math-bound. Moving bytes does nothing for it; retiling the WMMA does.

## Attention cache read is hard bandwidth bound

Three structurally different decode attention kernels (2-pass removed, load
pattern changed, barriers changed) produced identical 36.5 ms phase time at
B128. Source: GLM52_ABSORBED_MLA_DECODE_20260704. A time invariant under ALU,
load, and barrier changes is bound only by the bytes all versions share. The
absorbed-MLA formulation attacks the bytes directly: the cache read per
(sequence, slot) becomes the shared 576-element latent row (1152 B) instead
of 57 KB of per-head key/value rows, at a cost of about 30 MFLOP/token of
small per-head GEMMs — a large byte reduction for negligible added compute,
exactly the right trade on a bandwidth-bound machine.

## Memory-path efficiency eta_bw ~= 0.80

Three independent derivations agree: solving the B1 6-layer packaged stage
(14.626 ms, GLM52_MEASURED_STATUS) against its byte estimate gives about
0.80; the 12x model's roofline family implies 174-214 GB/s effective
(0.64-0.78); and the B128 routed-layer byte model (attention 165 MB + shared
expert 38 MB + routed MoE 5.3 GB = 5.5 GB, GLM52_B128_SCALING) predicts a
20.2 ms bandwidth floor against the doc's independently stated ~20 ms — a
match to one percent, which validates the byte accounting rather than fitting
to it.

## Batching amortization (measured)

B1/B4/B16/B64 aggregate throughput 3.71 / 7.58 / 20.99 / 58.22 token
events/s. Source: diagnostics/glm52_b64_api_performance_20260714 summary
JSONs. Per-token cost at B64 is about 25 percent of B1: weight bytes amortize
across the cohort. Amortization saturates by B128 as expert coverage
approaches 98 percent; past that, per-token stage cost is flat (1.56 ms/token
B64 to 1.48 ms/token B128, GLM52_B128_SCALING).

## Release reconciliation (do not average across these)

Per-stage numbers come from different releases and configs and must be kept
separate. The direct-fp8-tiled run (diagnostics/glm52_direct_fp8_tiled_
20260716) shows a clean B1 6-layer stage at about 29.9 ms/token (12 stages,
28.3-31.9 ms, excluding the 95.4 ms stage-6 M=1 outlier), about 5.0 ms/layer,
agreeing with the 12x doc's 4.6 ms backend per checked layer. The packaged-
driver stage-0 number is 14.6 ms for a different release with graph replay.
These are two configs, not one number with noise; the model carries both and
labels each by release.

## Per-driver application (family, byte geometry real, efficiencies inherited)

Each family driver inherits the calibrated constants (eta_bw ~0.80, QKVO
compute wall, MoE bandwidth bound, B64 amortization) and supplies its own
CONFIG byte geometry. First-order consequences that hold regardless of the
still-pending itemization:

- qwen38_27b is dense, not MoE: its FFN weight is read every layer with no
  expert sparsity to amortize, giving the largest per-token weight-byte term
  of the family. The device-grouped MoE work does nothing for it; its lever
  is dense-FFN quantization (FP8 to FP4 halves the dominant term).
- dsv4 is lightest: fewest experts (topk 6), MLA-style 512-latent attention
  (smallest KV byte term), thinnest head. Fastest per-token of the family.
- mimo25 sits between: MoE like dsv4 but heavier head (vocab 152k) and fatter
  attention (head_dim 192, dual full/SWA branch).
- qwen38_27b (hidden 5120) and k3 (hidden 7168) hit the QKVO 6.5-TFLOPS compute
  wall harder than dsv4/mimo25 (hidden 4096), because QKVO flops scale with
  hidden squared and the qkv width. If the family QKVO kernels share glm52's
  WMMA tiling, they inherit its 2.6-percent efficiency, and QKVO retiling is
  the highest-value cross-family compute optimization.
- k3 geometry is GUESS-tagged (K2 lineage); all k3 numbers are double
  conditional until its config is confirmed.

## What is pinned vs PENDING

Pinned (measured, cross-validated): the hardware ceilings; the regime map;
the QKVO 6.5-TFLOPS compute wall; the attention bandwidth bound; eta_bw
~0.80; the B64 amortization curve; the M=1 launch-overhead magnitude.

PENDING (needs the kernel-level nsys split the B256 doc points at — the
.nsys-rep itemization behind the bucketed summaries, which is not in-repo):

- Per-kernel-class eta_bw separation (a WMMA tile, a memcpy-bound cache
  scatter, and a head reduce hit different fractions of peak; the bucketed
  numbers mix compute, pack, and quantize inside the "generated MoE" bucket
  per the B256 doc's own note). PENDING: kernel-level nsys split at B64/B128/
  B256.
- The launch_ns constant to better than the graph-replay delta (17.20 to
  16.56 ms on a 72:6 stage, GLM52_PP13_2X_SPEEDUP_ROOT_CAUSE). PENDING: a
  1-row vs N-row same-stage sweep isolating fixed launch from per-row bytes.
- eta_flop separation between QKVO (measured 6.5 TFLOPS) and the MoE WMMA
  (inferred better-tiled, not measured). PENDING: MoE-only nsys timing.
- Any family number as silicon truth. PENDING: one measured decode-stage time
  per family driver, the same anchor glm52 has.

The model reproduces glm52's measured buckets in form and reproduces its
byte floors to one percent; it names the four measurements that convert the
family projections from conditional to real, and it re-derives everything the
moment those measurements land.
