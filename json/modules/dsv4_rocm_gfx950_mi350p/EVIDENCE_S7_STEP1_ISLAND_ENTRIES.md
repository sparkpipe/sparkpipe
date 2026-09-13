# Evidence - S7 step-1 island entries (E0 `prologue.embed`, L5 `layer.moe_shared`)

Date: 2026-08-25. Work item: docs/coord/plan_amd_gfx950_mi350p.md section
5.2 step 1 ("E0 + L5 first") against the S6 HIP backend already in this
module. Ran-where discipline per plan section 2 box limitation: every claim
classifies where it ran.

## What landed

- `source/spark_dsv4_rocm_islands.h` - core-facing entry surface: C linkage,
  geometry-free signatures, target-private linear view mirroring the
  portable SparkDsv4LinearView shape, weight-format codes numerically equal
  to the shared logical enum. No spark_lm_kernels.cuh inclusion anywhere
  (freeze F4 landmine rule; that header self-errors under non-GB10 targets,
  measured at lm_kernels.cuh:39).
- `source/spark_dsv4_rocm_islands.hip` - kernels + entries:
  - E0: verbatim-byte gather + hc stream expand, optional read-ahead kick.
  - L5: gate/up dot products, clamped swiglu, down projection accumulated
    into the FFN accumulator; fork/join events honored when provided;
    fixed reduction tree documented in the TU banner.
- `tools/island_compile_check.sh` - repeatable proof harness.

## What ran where

| Proof | Where it ran | What it proves | What it does NOT prove |
|---|---|---|---|
| ISLAND_TU_HOST_PROOF: full .hip TU as plain C++ (`clang++ -x c++ -std=c++17 -Wall -Wextra -Werror`) against vendored ROCm 7.14 headers | authoring workstation (Apple clang 16.0.0), 2026-08-25, exit 0 | every kernel body, helper, entry signature, launch call and status mapping parses; surface is complete and warning-clean | device codegen; wave64 shuffle semantics; LDS behavior |
| ENTRY_HEADER_C_PROOF: header consumed twice as C11 by plain clang | same workstation, exit 0 | core-facing header is valid C, guard-idempotent, self-contained | link-time symbol presence (owt by static archive build) |
| Primitive family regression: all five spark_hw_rocm_*.c recompiled clean via both compile_check.sh and island_compile_check.sh | same workstation, exit 0 | island work did not disturb the S6 backend | nothing new |
| hipcc --offload-arch=gfx950 device compile | NOT RUN - no ROCm toolchain on this box (plan section 2) | - | everything device-side |

The TU carries the proof-mode shims under `#if !defined(__HIP__)`; hipcc
never compiles that branch. First on-hardware action: run
`tools/island_compile_check.sh hipcc` on the MI350P node, then the C2/C3
comparisons of plan section 5.3.

## Decisions recorded (target-owned, R1/F4 sweep grant)

1. Entry symbols carry frozen id + name (`spark_dsv4_rocm_e0_prologue_embed`,
   `spark_dsv4_rocm_l5_moe_shared`) - no mapping table at link time (R2).
2. L5 reduction tree FIXED for all buckets: per-lane sequential fp32
   k-stride accumulation -> log2(64)-step shuffle-down butterfly -> one RNE
   bf16 store. Never retuned per bucket (B3/C3 stability requirement).
3. L5 adds its contribution into the FFN accumulator (AccumAdd semantic);
   callers zero-init at slot creation; add order fixed by queue/event order.
4. v1 island weight formats: BF16 scale-free + FP8_E4M3 with per-block E8M0
   scales (scale_group must divide columns). MXFP4 stays L4/S8 scope.
5. E0 requires hidden_dimension % 8 == 0 and an 8-byte-aligned packet buffer
   so the 16-byte vector path can never fault on a misaligned row base
   (pack rule: model dimensions are multiples of 64).
6. Launch mechanism is hipLaunchKernel (runtime API) rather than the
   hipLaunchKernelGGL macro - one less header dependency for host proofs;
   revisit only if hardware profiling says otherwise.
7. Shuffle spelling: __shfl_down_sync with a 64-bit mask and explicit width
   64. ROCm's portability layer takes uint64 masks; verify the exact
   signature on hardware with the hipcc mode before first C3 comparison.

## Addendum - L4 `layer.moe_routed` dispatch/routing round (same day)

A concurrently-running session landed the full L4 island into this same TU
(single-CTA deterministic route-group kernel, grouped W13+swiglu, grouped
W2, weighted pair reduce) on top of the step-1 work above. This office then
audited the dispatch/routing half against the canonical producer contract
(inference/kernels/route.cuh: count -> serial exclusive scan -> scatter;
"the order within an expert is whatever the atomics say" - here strengthened
to a run-invariant ascending rank, a superset of CUDA's guarantee) and the
split-verdict comparison mechanics (hwiface_v1.md section 3.5: expert
offsets is THE C2 probe surface; intra-group slot order reaches no
observable because every consumer reads through the inverse map in fixed
ascending-k order). Audit fixes applied:

1. VALIDATOR GAP (functional): SparkDsv4RocmCheckLinearView still accepted
   only BF16/FP8_E4M3, so MXFP4_E2M1 views - the routed-expert headline
   format - could never pass validation and the entry's MXFP4 branch was
   dead code. The validator now accepts MXFP4 with the stagepack block-scale
   rule plus the 8-element straddle rule (moved there so every island entry
   enforces it identically); the duplicated entry-side check was removed.
2. OVERFLOW GUARD: the pair reduce flattens row x hidden into one uint32
   lane index with no guard. row_count x hidden_dimension > UINT32_MAX now
   refuses SPARK_HW_EXHAUSTED (capacity, not argument error).
3. LAUNCH CAPACITY: grid-Y (W13/W2/L5 projection grids) exceeded the HIP
   65535 ceiling unchecked for large intermediates. Added
   SparkDsv4RocmCheckGridY; violations refuse EXHAUSTED.
4. FOLD-ORDER PARITY: the pair reduce summed weights from zero and added
   the accumulator last; CUDA folds pairs starting FROM the accumulator.
   Reordered to accumulator-first so the documented fixed tree matches the
   golden behavior element-for-element in association order.

Post-fix verification: ISLAND_TU_HOST_PROOF + ENTRY_HEADER_C_PROOF + all
five primitive regressions green (exit 0). Device-code verification on
MI350P remains owed exactly as classified above.

## Open items owned by later steps

- F1 skeleton (S7 step 2), L1 (step 3), L2/L3 (step 4).
- Golden frame list per bucket + L3 emitted-field table remain recipe-time
  preconditions of S7 acceptance (plan section 5.1) - not started here.
- Synthetic sealed-route batch driver for L4 C2/C3 comparison (plan section
  5.2 step 3 drives L4 with synthetic batches) - not started here.
- Note: a concurrently-running session commits this workspace aggressively;
  these files may have entered history inside that session's commits. Disk
  content is authoritative for this evidence note.
