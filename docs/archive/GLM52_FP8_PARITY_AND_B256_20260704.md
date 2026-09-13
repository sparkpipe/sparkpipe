# FP8 Model Parity and the B256 Re-Analysis - 2026-07-04

## B256: the fix that wasn't, and what the numbers now prove

Measured on main@1369d1d: B128 worst stage 1173 tok/s (109.1 ms), B256
worst-so-far 1211 tok/s (211.4 ms) - still 1.94x time for 2x tokens. Two
facts now pin the mechanism:

1. B128 at ~109 ms/stage puts the NVFP4 expert sweep at roughly the LPDDR5x
   roofline (~280-300 GB/s effective on ~30 GB of touched weights). A stage
   already at the bandwidth wall cannot be compute-bound, so "the primitive
   tops out at ~7 TFLOPS" is dead as an explanation.
2. The static kernel's work decomposition (moe_static_kernel.py,
   `_compact_static_get_work_tile`: work = per-expert ceil(rows/tile_m) x
   n-tiles, weights loaded per work tile) re-reads an expert's weights only
   across its own m-tiles. At B256, avg rows/expert = 8, so every expert has
   ONE m-tile - zero intra-launch re-read.

Therefore the 1.94x can only be TWO LAUNCHES: `SparkGlm52B12xLaunchChunked`
split B256 into 2 x 128 because the loaded bundle's largest bucket is 128.
No commit on main between the sweeps changed bucket generation, so nothing
that ran was a one-sweep B256.

Dev action (minutes): dump the generated manifest's bucket
token_upper_bound list from the loaded bundle (or Nsight the MoE launch
count for one B256 stage pass). Then regenerate the AOT bundle with
`tools/glm52_b12x_aot_compile.py --tokens ...,256,512,1024`.

Capacity blocker for big buckets: the dispatch's dense per-expert workspace
(`packed_input [state_E, max_rows, k//2]`, moe_dispatch.py:274-329) with
`maximum_routed_rows = max(buckets) * 8` (aot tool line 903) means a
1024-token bucket allocates 8192 rows x 256 experts x 3 KB = ~6.4 GB. Real
per-expert max at B1024 is ~60 rows (birthday bound), so max_rows can be
capped at 128-256 with a loud overflow path - that is the design change
that makes 512/1024 buckets practical. Projections if one-sweep is
restored: B512 ~4100 tok/s, B1024 ~7000 tok/s.

## FP8 model support: what has parity, what didn't, what changed

Model quantization modes (firmware.h:492): AUTO, NVFP4_4BIT, FP8_E4M3_8BIT.

### Already at parity (verified in source, no change needed)

- **Projections (q_a/q_b/kv_a/kv_b/o, dense MLP, heads):** the shared
  weight-element dequant used by both Wmma linear kernels branches on
  runtime `weight_format` and implements FP8_E4M3, NVFP4_E2M1 (ue4m3
  scales), and E2M1+E8M0 - so the weight-stationary batch kernel added for
  B>16 serves FP8 weights identically to NVFP4. Additionally, when a linear
  plan carries an `Fp8ScaledGemmBackend`, FP8 projections route to the
  compiled scaled-GEMM backend before any in-CU kernel.
- **FP8 activation plumbing:** quantize/dequant/amax kernels, fused
  RmsNorm->fp8 and SiluMul->fp8 are elementwise and fine as-is.

### The gap: FP8 MoE ran one block per output element

The FP8 MoE (the production path for FP8_E4M3_8BIT; NVFP4 uses the compiled
b12x backend instead) launched `Fp8MoePackedReferenceW1Kernel` with
grid = (intermediate_dim, routed_rows, 2): a 6144-wide scalar dot per
block, re-reading the expert's weight row for every routed row, W2 the
same shape. At B128 that is ~230 GB/stage of weight traffic (~7-8x over a
weight-amortized sweep) on CUDA cores.

Replaced with `Fp8MoePackedWmmaGroupKernel` - one kernel serving both W1
and W2 (output split routes rows to up/gate for W1; single output for W2):
grid = (weight-row 16-tiles, expert). Per expert segment (the packed route
build already produces expert-contiguous rows plus offsets/counts), the
block walks 128-row groups; per 16-wide K slab it dequantizes ONE 16x16
weight tile to shared bf16 (amortized over all row tiles) and each warp
dequantizes its own 16x16 activation tile, then wmma. Weight traffic
becomes coverage x weights once per launch (~35 GB/stage at B128, est
~7x). A deterministic `Fp8MoePackedFinalizeKernel` (token x hidden grid,
fixed route order, pre-normalized topk weights) replaces the fused W2
finalize; `packed_down_bf16` (route_count x hidden) was added to the
grouped workspace. Both element-grid kernels are deleted; the non-packed
`Fp8MoeReference*` kernels remain as the numeric oracle.

### FP8 KV cache prefill: now on tensor cores

The WMMA paged prefill attention kernel is now templated on the KV cache
format: `<0u>` gathers bf16 caches, `<1u>` gathers fp8 caches through
`Fp8ScaledRowLoad` with the exact extents the scalar kernel used (key nope
row = slot over 64x192, rope from the mla row at latent offset, value over
64x256) and converts to bf16 in shared memory; everything after the gather
is one code path. The scalar fp8-KV tiled prefill kernel is deleted. The
paged-prefill plan ABI constants now state the real geometry
(QUERY_TILE_TOKENS 4->16, KEY_TILE_TOKENS 2->16) with a static_assert
tying the kernel to them; hosts passing 0 (don't-care) are unaffected,
hosts asserting the old 4/2 geometry fail loudly at bind.

### Flagged, intentionally not changed

- **Decode fp8-KV attention kernels** stay scalar: context-length bound,
  irrelevant at short contexts; revisit alongside the bf16 decode online
  kernels for >= 8k serving.
- **`Fp8LinearKernel` grid(N, B)** is the plan-missing fallback only (one
  block per output element per sequence). Production always binds plans;
  left as the debug path.
- **`BlackwellNativeFp8TensorCoreLinearKernel`** is dead scaffolding:
  compiled out by a default-off flag and has no launch site. Either delete
  it or finish it as the fp8-MMA route below.

### FP8-specific optimization worth building (after spark2 numbers)

In the MoE, BOTH operands are already fp8 e4m3 (weights and quantized
activations). A native fp8 MMA variant of the WmmaGroup kernel
(mma.sync m16n8k32 e4m3) skips the dequant stage entirely and runs at 2x
the bf16 MMA rate. Only worth it if the WMMA path measures compute-limited
on spark2; bandwidth math says it likely will not at current batch sizes.

## Honest FP8-vs-NVFP4 expectation

FP8 expert weights are ~1.8x the bytes of NVFP4 (1 B/param vs 0.5625
incl. scales). B128 NVFP4 measures at ~the bandwidth roofline, so once the
FP8 MoE is weight-amortized its sweep should land near 1.8-2x the NVFP4
sweep time at B128 - "close to 4bit" holds at smaller batches where the
sweep is not saturated, not at the roofline. Let the B64/B128 FP8 sweeps
decide; before this change the FP8 MoE was ~7-8x off, so the comparison
was not even meaningful.

## Validation required on spark2 (nothing here is numerically verified)

1. FP8 MoE: run the grouped-vs-reference compare (non-packed
   `Fp8MoeReference*` kernels are the oracle) across batch sizes including
   segments > 128 rows per expert (forces the row-group loop).
2. Prefill: tests/test_glm52_exact_pp13_prefill_hidden.py for the bf16
   template instantiation, then the same with the fp8 KV plan bound.
3. FP8 stage sweep B1-B256 after (1) passes.
