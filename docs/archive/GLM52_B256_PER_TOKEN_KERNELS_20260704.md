# B256: Per-Token Kernels Are the Wall - 2026-07-04

## Retraction first

Two claims from GLM52_FP8_PARITY_AND_B256_20260704.md are falsified by the
spark2 discriminator run (main@365c913, stage 18:6):

1. **"The 1.94x is two chunked launches" - wrong.** The manifest has a real
   token_upper_bound=256 bucket, the validator logged b12x_moe_launches=6
   for 6 layers, and Nsight counted 6 generated-MoE launches per pass. One
   launch per layer at B256, and 1.94x persisted (107.2 -> 208.0 ms
   graph-on; 111.1 -> 214.1 ms graph-off).
2. **"B128 is at the bandwidth roofline" - wrong.** The Nsight
   decomposition shows where the time actually is (per timed pass):

```text
                       B128       B256     ratio
QKVO/linear WMMA     36.4 ms    73.9 ms    2.03x
decode attention     36.8 ms    73.0 ms    1.98x
B12x generated MoE   27.2 ms    47.8 ms    1.76x
B12x fc2 finalize    5.55 ms   11.29 ms    2.03x
```

The stage is not bandwidth-bound: ~70% of B128 time is per-token compute
running at ~6.5 TFLOPS effective, and it scales exactly linearly with
batch. That is the whole B256 story - tokens/s plateaus because doubling
tokens doubles the dominant kernels. The fix is kernel efficiency in QKVO
and decode attention, not MoE launch plumbing.

## QKVO batch kernel: why 6.5 TFLOPS, and the rewrite

The v1 weight-stationary kernel used 16-wide output tiles and 16-deep K
slabs. Per (block, slab) it performed 2048 global activation loads with a
bf16->float->bf16 round trip, 2048 per-element weight dequants, and two
__syncthreads - for eight 16x16x16 wmmas. Worse, grid.x = out/16 means the
full activation matrix is re-read from global once per 16 output columns:
384x for o_proj, 1024x for q_b. L2 absorbs much of it; the instruction and
sync overhead it cannot.

v2 (same signature, same format-generic dequant helper):
- output tile 16 -> 64 per block (4 accumulator fragments per warp):
  activation re-reads drop 4x (grid.x/4)
- K slab 16 -> 32: half the syncthreads; per sync pair each warp now runs
  8 mmas instead of 1
- activation fill is a single uint4 raw copy per thread per slab (no float
  round trip; elementwise fallback kept for input_dimension % 32 != 0,
  which no current projection hits)
- 20 KB static shared (8 input + 4 weight + 8 stage), launch_bounds(256,4)
- per-warp epilogue staging, no cross-warp syncs in the epilogue

Weight-dequant ALU per output element is unchanged (already minimal per
block); the removed waste is activation traffic, sync count, and mma
density. Expect >= 2x on the QKVO bucket; spark2 decides.

## Decode attention: same money, two structural cuts

Both bf16 decode kernels (AttentionKernel: two-pass over
SELECTED_TOKEN_COUNT=2048; AttentionOnlineKernel: 8-slot online rounds;
launcher picks by attention_execution_mode) share the fixes:

1. **WarpDotProduct** went from strided uint32-pair loops (4+ iterations
   per 256-dim dot) to exactly one uint4 load per lane: lanes 0-23 cover
   the 192 nope dims, lanes 24-31 the 64 rope dims (row bases are
   8-element aligned in both caches). Note: this regroups the warp-sum
   partials, so scores differ by FP reassociation - same class of change
   as the MoE kernel swaps; validator tolerance applies.
2. **Online kernel softmax-rescale hoist:** previously all 256 dim-threads
   redundantly recomputed the identical per-tile online-softmax chain -
   2 __expf + max/branch logic per element per tile. Thread 0 now walks
   the 8 tiles once per round (bit-identical operation order), publishing
   (old_scale, value_scale) per tile plus the running denominator to
   shared; dim threads apply at most 2 FMA per (dim, tile). Costs one
   extra __syncthreads per 8 slots.
3. **Base kernel:** probabilities are normalized in shared once after the
   block reduce, removing a per-(dim x candidate) division across 2048
   candidates.

The fp8 decode pair has the same two shapes and should get the mirrored
treatment once the bf16 versions validate; not changed in this pass.

## MoE 1.76x: open question, no blind fix

27.2 -> 47.8 ms with weight bytes ~flat (coverage 98.2% -> 99.97%) is not
explained by the weight sweep. Candidates: per-n-tile re-reads of the A
operand and scale-factor tensors (scale with routed rows), or the
pack/quantize/route stages being inside the "generated MoE" Nsight bucket.
The existing .nsys-rep files can answer this with a kernel-level split of
that bucket before anyone touches codegen.

## Expectations if the kernels deliver

At 2x QKVO and 1.5x decode attention: B256 ~ 37 + 49 + 47.8 + 11.3 ~ 145
ms (~1770 tok/s), B128 ~ 78 ms (~1640 tok/s). Batch scaling past that
requires the MoE sub-scaling answer and, eventually, decode-attention
traffic physics (per-sequence KV is irreducibly linear in B).

## Validation on spark2

1. Correctness: full validator token-match at B>16 (QKVO batch kernel
   serves all decode projections; decode attention changes are
   tolerance-level only, reassociation note above).
2. tests/test_glm52_exact_pp13_prefill_hidden.py unchanged paths sanity.
3. Stage sweep B64/B128/B256 with the same Nsight decomposition to
   attribute the delta per bucket.
