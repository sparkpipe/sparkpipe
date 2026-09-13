# Evidence - S7 step-4 L3 island (layer.cache_transition)

Date: 2026-08-25. Work item: docs/coord/plan_amd_gfx950_mi350p.md section
5.2 step 4 ("L2/L3 last ... ring-page emission"), L3 leg, against the
frozen split verdict of hwiface_v1.md section 3.1: page addresses, ring
indices and emit counters are C2 integer-pure; emitted payloads are a
verbatim bf16 copy on the direct path (C2-eligible) or quantized sim output
on the emission path (C3, per the recipe-time emitted-field table).

## What landed

- source/spark_dsv4_rocm_islands.h - three symbols share the frozen L3
  id+name prefix because their lifecycles are disjoint while remaining one
  semantic island (R1: kernel count is target-free):
  - spark_dsv4_rocm_l3_cache_transition  - per-frame transition
  - spark_dsv4_rocm_l3_initialize_pages  - allocation-time page setup
  - spark_dsv4_rocm_l3_update_page_table - scattered table commits
- source/spark_dsv4_rocm_islands.hip - five kernels + entries:
  - CompressStepKernel: per-lane recurrent ring advance; INTEGER emission
    decision boundary = ((position+1) % ratio == 0); softmax pool over the
    group window (plain and 2x overlapped forms); overlap window shift;
    emitted flags; NEW: optional per-lane emit counter via integer atomics
    (totals exact and run-invariant - C2).
  - KvEmissionKernel: RMS norm -> RoPE -> optional Hadamard rotate ->
    fp8/fp4 block sim-quant -> ring scatter, every bf16 materialization
    boundary of the CUDA kernel retained in order.
  - CacheScatterKernel: verbatim bf16 rows at slot = base +
    ((position % ring_slots)/ratio); optional per-row ring-slot output
    (explicit C2 observable).
  - InitializePagesKernel / UpdatePageTableKernel: byte-copy page init with
    parent inheritance + -inf score spans; scattered table writes.
- Proof-mode additions: __shfl_sync broadcast shim, rsqrtf shim, and
  SPARK_DSV4_ROCM_SHARED_DYNAMIC/FIXED macros rendering extern __shared__
  as function-scoped statics under a host parse (hipcc never compiles that
  branch).

## What ran where

| Proof | Where | Proves | Does NOT prove |
|---|---|---|---|
| ISLAND_TU_HOST_PROOF (full TU as plain C++, clang++ -std=c++17 -Wall -Wextra -Werror) | authoring workstation, exit 0 | all kernels/entries parse warning-clean; launch plumbing and validation complete | device codegen, wave64 shuffles, shared-memory layout |
| ENTRY_HEADER_C_PROOF (C11 twice) | same box, exit 0 | three new declarations valid C11 | link-time symbol presence |
| Primitive family regression (5 TUs) | same box, exit 0 | S6/S7 work undisturbed | nothing new |
| E4M3 encoder probe (tmp/e4m3_encoder_probe.cpp): fixed-point property over all 250 non-NaN codes, monotonicity sweep across the positive float domain (stride 97 bit patterns), tie/saturation spots | same box, 22,052,796 checks, 0 failures | the software cvt.rn.satfinite mirror is exactly RNE saturating-finite | nothing about speed |
| MXFP4 byte-vs-word decode equivalence (previous step, standing) | same box | unchanged | - |

## Defect found and fixed during this step

The first E4M3 encoder draft corrupted the exponent for inputs just below
a binade boundary: when the normal-path 3-bit rounding carried (q == 16),
the incremented exponent leaked into the subnormal branch's grid selection,
so values like 0x3affffda (= 2^-9 minus 2^-33) rounded to 0x02 (2^-8)
instead of 0x01 (2^-9). Found by the monotonicity sweep, confirmed by an
instrumented boundary trace, fixed by branching on the ORIGINAL unbiased
exponent before any carry. Post-fix: 22,052,796 checks green. The sweep
itself initially reported two false positives (NaN samples leaking past a
numeric guard, and two mislabeled spot expectations around the
subnormal/normal boundary); both were probe defects, corrected in place.

## Numerics parity notes (target-owned decisions)

1. Pool arithmetic (softmax pooling, expf, sentinel, slot walk order) is a
   verbatim port - bit-matched to CUDA given identical inputs.
2. Hadamard rotate touches disjoint pairs with pure add/sub per stage:
   bit-exact regardless of wave width or scheduling.
3. RoPE pairs are lane-local adjacent-element rotations: bit-exact.
4. Sim-quant amax reduction is order-free (max): identical results on the
   wider wavefront. Groups stride across four wave64 warps where CUDA used
   eight 32-lane warps - coverage identical, groups self-contained.
5. The ONE deliberate tree difference: RMS-norm block reduce runs the
   archive's documented wave64 butterfly instead of CUDA's width-32 tree,
   inside the C3 tolerance contract for the quantized emission path; fixed
   per shape bucket (B3).
6. Emit counters are integer atomicAdd totals - order-free exact (C2).
7. Direct-path payload bytes are never re-rounded (verbatim u16 moves):
   C2-eligible per the unrounded-copy clause.

## Open items owned by later steps

- F1 skeleton exists from step 2; L1/L2 landed separately per commit
  53c2bf1 (not present in this module's sources on disk at writing time -
  reconciled by whoever owns that workstream; disk content is authoritative).
- Golden frame list per bucket and the L3 emitted-field table (which
  payload bytes are C2-unrounded vs C3-quantized per field) remain
  recipe-time preconditions of S7 acceptance (plan section 5.1).
- First on-hardware actions: tools/island_compile_check.sh hipcc, then the
  synthetic-position C2 comparison (offsets/ring slots/counters vs golden)
  before any C3 tolerance run.
