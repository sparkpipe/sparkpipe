# Evidence - S7 step-3 L4 island (grouped MoE experts + FFN down projection)

Date: 2026-08-25. Work item: docs/coord/plan_amd_gfx950_mi350p.md section
5.2 step 3, L4 leg ("L1 then L4", "L4 driven with synthetic sealed-route
batches over the slot-workspace views"). Naming note: the user-facing ask
"L3 FFN down projection" maps to this island - frozen L3 is
\`layer.cache_transition\` (hwiface_v1.md sections 3.1-3.2: FFN ~ L4+L5);
the FFN down projection is the W2 phase of L4/L5. No cache_transition entry
is claimed by this step.

## What landed

- \`source/spark_dsv4_rocm_islands.h\` - L4 entry declared:
  \`spark_dsv4_rocm_l4_moe_routed\` (C linkage, geometry-free). New weight
  format code SPARK_DSV4_ROCM_WEIGHT_FORMAT_MXFP4_E2M1 = 3u mirroring the
  shared logical enum value; view contract documents stacked-expert row
  addressing and the block-scale rules.
- \`source/spark_dsv4_rocm_islands.hip\` - four kernels + entry:
  - RouteGroupKernel (THE C2 half): single-CTA route realization - zero,
    histogram (order-free atomics), thread-0 in-place descending exclusive
    scan publishing canonical expert_offsets + total, then counting-rank
    scatter. Grouped assignment is a pure function of core-sealed element
    order: run-invariant, not merely usually-stable.
  - ExpertW13SwigluKernel (C3): grouped W13 gate/up over stacked experts,
    fixed L5 tree per lane subset, clamp semantics identical to L5,
    routing weight NOT folded (folded once at reduce - CUDA MoePairReduce
    parity).
  - ExpertDownKernel (C3): the routed FFN down projection - grouped W2,
    unweighted output rows.
  - MoePairReduceKernel (C3): weighted pair reduce in ascending k order,
    AccumAdd semantics into the shared FFN accumulator, race-free through
    the inverse map (one writer per output element, no atomics).
  - SparkDsv4RocmWeightElement gained the MXFP4 branch: byte-addressed
    nibble decode (low nibble first) + per-block E8M0 scale.
- Proof-mode shims extended for the new device constructs (__syncthreads,
  atomicAdd); hipcc never compiles that branch.

## What ran where

| Proof | Where it ran | What it proves | What it does NOT prove |
|---|---|---|---|
| ISLAND_TU_HOST_PROOF: full .hip TU as plain C++ (clang++ -x c++ -std=c++17 -Wall -Wextra -Werror) against vendored ROCm 7.14 headers | authoring workstation, 2026-08-25, exit 0 | all six kernels, both entries, every launch call parses warning-clean; surface complete | device codegen; wave64 shuffles; atomic throughput |
| ENTRY_HEADER_C_PROOF: header as C11 twice | same workstation, exit 0 | L4 declaration valid C11, self-contained, guard-idempotent | link-time symbol presence |
| Primitive family regression: five spark_hw_rocm_*.c recompiled clean | same workstation, exit 0 | island work did not disturb the S6 backend | nothing new |
| MXFP4 byte-view vs shared word-view decode equivalence probe (tmp/mxfp4_byteorder_probe.cpp, plain host C++; 8,008,192 element comparisons: every byte value at every word position x8 elements + 10^6 random words) | same workstation, exit 0, EQUIVALENT | the island's scalar nibble addressing decodes EXACTLY what SparkLmDecodeE2m1x8Half2 decodes - the riskiest below-seam claim of this step | nothing about speed; fp16 round-trip in the shared path is exact for E2M1 so equality holds through it |
| hipcc --offload-arch=gfx950 device compile | NOT RUN - no ROCm toolchain on this box (plan section 2 box limitation) | - | everything device-side |

First on-hardware actions: \`tools/island_compile_check.sh hipcc\`, then a
synthetic sealed-route batch through the entry and the plan section 5.3 C2
halves (expert offsets buffer vs golden hashes) before any C3 comparison.

## Decisions recorded (target-owned, R1/F4 sweep grant)

1. Sealed-route input shape: flat row-major route_indices_u32 +
   route_weights_f32 [row_count x experts_per_token] - the slot-workspace
   content below the opaque handle (freeze F2/F3); aggregation policy stays
   core-side, this entry only realizes (freeze section 3.4).
2. Deterministic grouping without tile prefixes: CUDA's group_tile_prefix_*
   buffers are scheduling artifacts of its persistent-tile kernels; this
   target launches one CTA per pair directly from offsets, so those views
   do not exist here. Contents below the island entry are target-defined
   (freeze F2(3)). The DECLARED C2 observable - expert_offsets with
   canonical ascending-expert exclusive prefix + total - is produced
   exactly once, in expert order.
3. Within-expert packing order: ascending flat route element index
   (row*K+k). Fixed once, documented in the header banner (B3 discipline).
4. Routing weights fold ONCE, at the pair reduce (unweighted down rows) -
   mirrors module.c:3357 LaunchMoePairReduce(pair_weights_f32), not the
   legacy fold-at-swiglu SwigluClampKernel path.
5. Trust boundary: out-of-range route indices are a core contract
   violation and are NOT device-guarded - identical trust to the CUDA
   GateRouteBuildShared atomicAdd on route_expert[index]. Documented at the
   entry, not silently handled.
6. Bring-up O(n^2) counting rank in phase 4: correctness first; S7
   explicitly excludes perf (plan section 5.3). A scan-based rank pass is
   the obvious later optimization inside the same fixed ordering.
7. Empty frame still publishes canonical zeroed offsets (+total=0) so the
   C2 probe surface is always coherent; GEMM/reduce grids are skipped.

## Open items owned by later steps

- F1 skeleton (S7 step 2) landed separately; L1 (step 3 first half),
  L2/L3 = layer.cache_transition (step 4) remain.
- Golden frame list per bucket + L3 emitted-field table remain recipe-time
  preconditions of S7 acceptance (plan section 5.1) - not started here.
- Note: a concurrently-running session commits this workspace aggressively;
  these files may have entered history inside that session's commits. Disk
  content is authoritative for this evidence note.
