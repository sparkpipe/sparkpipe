# GLM 5.3 Flash hill-climbing log

Operator scope, 2026-09-08: concentrate on GLM Flash until it meets and exceeds
the hardware-normalized performance target. Defer other model integrations
and optimization lanes. Use their existing code as potential donors, but
prove the GLM result first. This log becomes the migration handbook for other
drivers after the path is qualified; unfinished ideas are not recipes.

## Acceptance and measurement

Maintain separate functional and performance verdicts. Full functionality
includes arbitrary continuous batching, complete JIT/prefix state, lifecycle
and numerical correctness. Performance compares matched precision/traffic,
context, batch occupancy, speculative mode and hardware count. Target TP16
at 3.5x qualified TP4 throughput and filled TP4xPP4 at approximately 4x TP4
capacity. See [driver acceptance](DRIVER_ACCEPTANCE.md) for common-code
boundaries and [measurement gates](GLM_PERFORMANCE_GATES.md) for receipts.

For every optimization record: trigger/profile, underlying error or cost,
shared-code placement, before/after source and build identities, correctness
result, repeated unprofiled performance, regressions and next decision.
Never infer batching from output count, prefix restore from a descriptor,
correct math from repeatability, or full serving from a component probe.

## Changes and lessons so far

| Change | Evidence and current limit | Reusable lesson |
| --- | --- | --- |
| Strict lazy expert loading, PR #842 | Missing/corrupt configured expert manifests must fail; common weight daemon and per-consumer mappings replace eager fallback. Full working-set/performance qualification remains separate. | One shared manifest/loader contract, bounded residency and in-flight protection; do not put fallback loaders in each driver. |
| Real driver probe, PR #854 | Executes the actual module on the real rank0 pack; distinguishes resident and lazy paths. Four fixed-input steps with collectives disabled are component tests. | Test the real consumer and record scope; a mock cannot prove weight mappings or CUDA safety. |
| Single-page KV transfer capacity, PR #855 | Configured transfer capacity exceeded a one-page pool. Bound transfer slots to the actual page count. | Validate pool geometry in common cache configuration, including the smallest legal shape. |
| Index-KV stride, PR #856 | Per-layer addressing was combined with a page size already multiplied by layer count, producing invalid accesses. Correct the per-layer page stride. | Trace allocation, layer offset, page index and byte stride together; a compiling field access can still have wrong units. |
| True batch waves, PR #858 / main `93c8f0d` | Removed an unconditional one-row clamp. True B3 memcheck returned zero errors; resident/lazy parity and two concurrent lazy consumers passed. This does not establish distributed model accuracy. | Audit actual launch shapes. A B3/B7 request can still execute as repeated B1. Diagnostic clamps require DEBUG and visible diagnostics. |
| Separate decode window, PR #859 / main `b8f20ae` | Wrapper retains per-sequence traces and measures the interval after every first token and before the first sequence finishes. | Separate prefill and batch tail effects. Client arrival timestamps are not device execution timing; this is not yet a required-hit prefix-cache benchmark. |
| Shared row policy, PR #860 / main `5500665` | GLM wrappers use common row validation and wave selection with indexed lane callbacks. Host harness passes widths 1–101, ragged waves, invalid order and released claims. Merged-main B3 memcheck and resident/lazy parity pass. | The qsort pattern applies directly: common algorithm, narrow ordinal callback, opaque context. Replace O(rows × lanes) searches with indexed lookups. |
| Mandatory serving contract, draft PR #861 | ABI 21 removes seven opt-out bits, makes callbacks/cache geometry mandatory and removes zero-cache scheduling paths. Common host tests pass; GLM cache/reset integration remains incomplete. | Required means fail explicitly when missing. Callback presence is only structural validation; stubs and flags cannot establish behavior. |
| Accurate ABI probe, draft PR #861 | Replaced copied, incorrect structs/flag values with the public header and common loader. | Diagnostic tools must consume the same contract as production, or they can report misleading capability results. |

## Baseline that must not be misinterpreted

On `a53ca6f`, the unprofiled TP16 client rates were approximately 14.13
aggregate output token/s at B1, 13.76 at B3 and 13.19 at B7. B3/B7 rows were
serialized by the clamp later removed in #858. The outputs were incoherent;
these are diagnostic timing receipts, not accepted model performance.
Artifacts: `/private/tmp/ds4_glm_tp16_a53ca6f/` on the controller, including
commands, provenance, per-rank logs, token traces and the rank0 Nsight export.

The B1 decode-only rank0 profile attributed about 58.18 ms/token of summed GPU
kernel time: expert FP8 indirect 13.91 ms, HC mix 12.23 ms, dense BF16 GEMM
12.20 ms, accumulation-add 5.62 ms, HC Sinkhorn 4.29 ms and direct expert FP8
2.70 ms. These are B1 profiling observations, not a critical-path decomposition
for overlapped batching and not speedup promises. Reprofile after real batch
execution and after each material scheduling change.

True B3 component receipts on `93c8f0d`:

- `glm-batch-memcheck-93c8f0d`, attempt
  `1e649ffaad2a4f798486cff0d234f792`: CUDA memcheck zero errors.
- `glm-batch-compare-93c8f0d`, attempt
  `0358bcbd01ee4d3e87152c60b5ffc79b`: resident/lazy parity, concurrent lazy
  consumers; rank0, collectives disabled, four fixed-input steps.

The same component checks passed after the shared-row refactor on `5500665`:
`glm-common-build-5500665` (attempt `31fb02910dcd47e6b382fcd43402af28`),
`glm-common-memcheck-5500665` (`28ba044fffb8454388d75cb067675a73`, zero errors),
and `glm-common-compare-5500665` (`4a2f9ae0165f403b8e79bb24e2a0201e`, parity).
All participant cgroups stopped and the assigned Spark was released. These
checks do not provide a new distributed throughput result.

## Current next steps, not completed work

1. Complete GLM integration with the shared cache: dynamic logical/physical
   mapping and full KV/index/KDA/convolution/continuity state restoration.
   Prove prefix-hit execution matches uninterrupted computation. No fixed
   identity page mapping or KV-only snapshot may masquerade as this result.
   The allocation trace also shows a layout mismatch: arena page bytes cover
   all KV layers, while CUDA pools are layer-major; the current page-copy hook
   performs one contiguous transfer. Total-allocation-size checks do not prove
   that any individual page contains the right layer slices. Resolve this
   payload-layout contract with a multi-layer round-trip test before enabling
   cache reuse. Include index state in the payload contract, not only main KV.
2. Qualify true batched distributed computation and clean release/reconnect.
   A continuation-lease teardown failure was observed on the old baseline;
   preserve the safety guard and fix ownership rather than suppressing it.
3. Restore a qualified prefix once for repeated warm-decode benchmarks, require
   cache hits, and measure arbitrary occupancies and continuous traffic.
4. Investigate the one-CTA HC mix bottleneck. A split-K HC implementation
   already exists in the DSV4 CUDA source. Inspect its geometry, reduction
   order and qualification history before extracting a parameterized shared
   mathematical operation for GLM. Do not paste family-specific constants.
5. Profile dense and routed-expert batched weight reuse, then refactor common
   collective dependencies to overlap independent compute with reduction.
   Measure exposed communication, buffer lifetimes and splitting overhead.

Use universal policy where possible, with orthogonal topology, shared
mathematical operation, precision/codec and device implementation boundaries.
An optimized CUDA operation can be shared across families without placing
CUDA assumptions in universal scheduling. The October Mac Studios require
the same policy with Metal backend operations, not another model scheduler.
