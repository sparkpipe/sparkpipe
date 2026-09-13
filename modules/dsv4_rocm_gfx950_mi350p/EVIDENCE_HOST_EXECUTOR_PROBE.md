# Evidence - host-executor probe for the island C2 surfaces

Date: 2026-08-25. Improvement pass (queue empty, charter area): until now
every island kernel carried PARSE evidence only - nothing on this workstation
had ever EXECUTED one. This pass adds execution-grade evidence for every C2
integer surface, which is exactly the class the frozen contract demands be
bit-exact.

## Mechanism

The coordinate shims gained a second host rendering gated by
SPARK_DSV4_ROCM_HOST_EXECUTOR: grid/block/threadDim become three DISTINCT
executor globals instead of one aliased zero object. Under blockDim = 1 the
lone thread walks every blockDim-strided loop end to end, so any kernel whose
elements are wholly computed by one thread reproduces the device final state
bit for bit. Kernels whose results depend on cross-lane reductions (W13/down
grouped GEMM, pair reduce, emission post's RMS/quant) are executor-excluded
by design - their arithmetic is covered by the codec/equivalence probes and,
ultimately, hardware.

## What is now executed per run of tools/island_compile_check.sh

| Kernel | Checks |
|---|---|
| RouteGroupKernel | canonical exclusive-scan offsets + total; grouped source/expert arrays; inverse map; within-expert ascending packing; identical results under two different garbage-poisoned buffer starts (run-invariance) |
| CacheScatterKernel | ratio-0 and ratio+base_slot addressing; verbatim payload bytes; emitted predicate skips; last-writer-wins collisions; optional ring-slot observable |
| InitializePagesKernel | root zero-fill; span poison to -inf bits; child copies parent bytes |
| UpdatePageTableKernel | scattered index/value commits; untouched entries intact |
| CompressStepKernel | recurrent state ring writes + APE add; integer boundary decisions; emitted flags; per-lane emit counters via atomic totals; pooled emit rows bit-equal to a mirrored fp32 reference (weights expf(0)); non-boundary rows emit zeros; non-overlap shift skipped |

## Defects found by building the probe (all in the PROBE, none in kernels)

1. Executor threading model: first draft kept real blockDim (256) while
   walking only thread 0 - strided loops then covered just their first
   element and one-element-per-thread kernels saw one update. Fixed by
   blockDim 1 + grid-sized-to-count for element-per-thread kernels.
2. Stale fill bounds after resizing a test buffer left words 32..127 as raw
   stack garbage, producing phantom poison mismatches that moved between
   runs. ASan confirmed zero out-of-bounds accesses; the fix was completing
   the initialization, not touching kernels.
3. Hand-transcribed expectation tables for route/scatter duplicated the flat
   input sequence instead of the slot-ordered result and missed documented
   last-writer-wins collisions. Rewritten slot-by-slot from counts
   e0:4 e1:3 e2:6 e3:3 e4:4 e5:4.

The kernels themselves needed no changes this pass - consistent with their
design goal of thread-count-invariant C2 semantics.

## Ran-where

Authoring workstation, clang++ -std=c++17 -Wall -Wextra -Werror -O2,
vendored ROCm 7.14 headers, executed natively (arm64). Probe wired into
tools/island_compile_check.sh as L3_C2_HOST_EXEC_PROBE and green in the
standard suite. Device codegen verification remains owed on MI350P hardware
(hipcc mode + C2/C3 oracle runs), exactly as classified before.
