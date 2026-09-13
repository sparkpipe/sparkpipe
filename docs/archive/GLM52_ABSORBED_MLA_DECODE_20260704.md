# Absorbed-MLA Decode: Attacking the Bytes, Not the Kernel - 2026-07-04

## Diagnosis: three null results equal a traffic bound

Three structurally different optimizations of the decode attention kernels
produced identical phase times (36.5 / 36.5 / 36.5 ms at B128): removing
2 expf + branch logic per element per tile, cutting load instructions 4x
with uint4 gathers, and eliminating 764 of 768 block barriers via the
2-pass swap. A kernel whose time is invariant under ALU, load-instruction,
and barrier changes is bound by the one thing all versions share: the
bytes. Decode reads per-head keys and values - 64 heads x (192 nope + 256
value) x 2 B ~ 57 KB of unique cache bytes per (sequence, slot) - which is
irreducibly linear in batch AND in context length. At production context
(>= 2048 selected candidates live) this phase alone would be several
hundred ms per stage at B128. The kernel cannot be optimized out of that;
the bytes have to go.

## The absorbed formulation (this PR, ATTENTION_EXECUTION_ABSORBED_LATENT = 2)

MLA's whole point: per-head keys and values are linear images of one
shared 512-dim latent (kv_b weight rows, per head h: W_UK = rows
[h*448, h*448+192), W_UV = rows [h*448+192, h*448+448) - layout grounded
from the cache-write kernel's raw_kv_b indexing). Therefore:

  score[h, slot] = (W_UK[h]^T q_nope[h]) . latent[slot, 0:512)
                 + q_rope[h] . latent[slot, 512:576)
  out[h] = W_UV[h] . softmax-AV over latent[slot, 0:512)

identical attention math, but the cache read per (sequence, slot) is the
576-element mla row (1152 B) shared by ALL heads, instead of 57 KB of
per-head rows. Extra compute is ~30 MFLOP/token of small per-head GEMMs -
noise. Everything needed already lives in mla_cache_bf16.

## Implementation (zero new buffers, zero ABI change)

Four kernels plus an orchestrator behind the new execution mode; the old
paths are untouched and remain the default:

1. AbsorbedQueryProjectKernel - weight-stationary per-head GEMM
   q_absorbed[h] = W_UK[h]^T q_nope[h], dequantizing kv_b tiles through
   the standard runtime-format helper (NVFP4 and FP8 both served). Output
   columns [192, 512) land directly in query_latent_bf16 (rows are
   LATENT_DIMENSION wide with only [0, 192) used by the legacy path);
   columns [0, 192) would collide with the input, so they park in the
   raw_kv_b_bf16 slot buffer, which is dead after the rope/KV-write phase.
2. AbsorbedQueryCommitKernel - copies the parked [0, 192) columns into
   query_latent after all project blocks finish (kernel boundary is the
   fence).
3. AbsorbedAttentionKernel - grid (sequence, 4 head groups), 16 warps =
   16 heads per block. Per 16-slot tile: block-cooperative slot resolve
   and one gather of the 576-wide mla rows into shared memory; each warp
   scores its head from shared and runs a per-warp online softmax with
   the 512-dim latent accumulator held in registers (16 f32 per lane),
   tile-batched rescaling (one rescale per tile, not per slot). Cache
   traffic per (sequence, slot): 4 x 1152 B (one read per head group) -
   a 12x byte reduction versus ~57 KB; the 64-heads-per-block variant
   with dynamic shared memory doubles that to ~25x and is the noted v2.
   Each warp finally overwrites its own query_latent row with the
   normalized latent output (the row is dead after the shared-memory
   staging).
4. AbsorbedValueApplyKernel - weight-stationary per-head GEMM
   out[h] = W_UV[h] . latent_out[h] into attention_output_latent_bf16 in
   the exact legacy layout, so the output projection is untouched.

Rope handling is bit-compatible in structure: shared query rows are
[q_absorbed 512 | q_rope 64], latent tiles are the raw mla rows
[latent 512 | rope 64], the score is the full 576-dot times qk_scale -
the same nope-dot + rope-dot the legacy kernel computes, with the nope
part algebraically re-associated through W_UK.

## Numerics and gating

q_absorbed is rounded to bf16 after the W_UK transform, so logit deltas
are larger than pure re-association - small but real. That is why this
ships as mode 2 with the legacy path intact: flip
attention_execution_mode to 2 in the sweep config to test (note: the
EXECUTION_REQUIRE_TILED_ONLINE_ATTENTION flag must not be set in that
run; the bind-time contract rejects the combination). v1 requires the
bf16 mla cache (absorbed + fp8-KV returns INVALID_ARGUMENT; the fp8
variant is a straightforward follow-up once bf16 validates).

## Expectations

At the sweep's context the MLA attention phase should drop from
36.5 / 72.5 ms (B128 / B256) to the single-digit range, and - the real
prize - the phase becomes ~50x flatter in context length, which is what
production decode at long context needs. If it does NOT drop, the
traffic diagnosis is wrong too and the nsys kernel-level numbers
(achieved DRAM throughput for the attention kernel) become mandatory
before anything else.

## Validation on spark2

1. Token-match B128/B256 with attention_execution_mode = 2 versus the
   mode-1 baseline (expect exact token match; logit deltas above
   association level but well under tolerance).
2. tests/test_glm52_exact_pp13_prefill_hidden.py (prefill path untouched).
3. Stage sweep + phase split, both modes, same buckets.
4. LOCAL_MOE remains the other half of the wall; the per-kernel split of
   phase 8 from the existing nsys reports is still the required artifact
   there.
