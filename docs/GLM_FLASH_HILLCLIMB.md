# GLM 5.3 Flash hill-climbing log

Operator scope, 2026-09-08: concentrate on GLM Flash until it meets and exceeds
the hardware-normalized performance target. Defer other model integrations
and optimization lanes. Use their existing code as potential donors, but
prove the GLM result first. This log becomes the migration handbook for other
drivers after the path is qualified; unfinished ideas are not recipes.
Keep the implementation as simple as the problem permits. Extend existing
common primitives, keep required ownership transitions explicit, and avoid
new frameworks or alternate paths that are not needed for GLM qualification.

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
| Layered KV/index backing payload, PR #862 / main `d687fc8` | Common gather/scatter with a hardware copy callback; GLM supplies native geometry. Actual GLM host hook tests pass both regions across three layers and five pages; substituting the previous contiguous copy fails the payload check. CUDA CI, merged-main B3 memcheck and resident/lazy/concurrent parity pass. | Trace native layer/page indexing against backing payload layout. Total allocation size is insufficient; distinguish each page and layer in tests. Include index state in payload sizing. |
| Numerical gate integrity, PR #863 / main `52ce0e5` | Three probe failure results were discarded; projection readback could skip a comparison. Checks now affect exit status. Common metrics reject nonfinite inputs and accumulate squared errors directly; regression tests reject the previous metric implementation. Corrected GPU component validator passes on merged main. | Test the acceptance test with bad values. Error-norm cancellation and ignored return codes can turn an optimization regression green. Component repeatability is not numerical correctness. |
| Cache admission wiring, draft PR #861 | GLM builds persistent cache lanes for submitted frames and routes prepare/commit/abort through a common algorithm with a validation callback and caller-owned scratch. Host tests preserve B3 lane identities and transaction generations, propagate driver failure, validate all inputs before dispatch and keep release separate. Full reset/restoration and GPU serving acceptance remain incomplete. | Common policy builds and validates the transaction; the model supplies geometry and hooks. Keep submitted lane storage alive through device completion. |
| KDA oracle recurrence, PR #864 / main `06be88e` | The C oracle decayed state, then applied decay again in its prediction. A shared scalar reference has hand-calculated two-token nonzero-state and rectangular-state tests; injecting the old second decay fails. CUDA CI and merged-main synthetic GPU validation pass; KDA+dense+HC relative L2 is 0.00376, cosine 0.9999930. | Validate recurrent state directly with nonzero initial state. Small random end-to-end fixtures can underweight reference errors. Keep reference math independent of production kernels. |
| Shared page pinning and transaction mapping, draft PR #861 | Common cache operations now own prepare, commit, claim, finish and abort. GLM uses them under its cache mutex, initializes device mappings to invalid entries, and uploads actual physical mappings only when they change. Host tests exercise the actual GLM admission/claim/upload/completion hooks, plus common stale-owner, partial-failure and eviction tests. GPU serving qualification remains pending. | Physical residency is separate from logical prefix identity. Keep logical tables immutable and physical pages pinned until GPU completion. Serialize cache metadata access; after GPU completion unpin before deduplication can free a writable page. This policy belongs in common code. |

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

After #862, all three jobs passed on clean main `d687fc8`:
`glm-cache-build-d687fc8` (attempt `a2f9cdc14eef41aa91b2b70c4f8e4454`),
`glm-cache-memcheck-d687fc8` (`c158b1860ba24b9bbca1ccf4b39ed654`, zero errors),
and `glm-cache-compare-d687fc8` (`c75db10414b74f498cbf4e528a76d89a`, parity).
The queue confirmed all cgroups stopped and Spark0 released. This proves the
component checks survived the cache change; full prefix restoration remains
unqualified.

The existing synthetic validator also passed on `5500665`, job
`glm-synthetic-oracle-5500665`, attempt `8a86853a13754fd6983359322e3d5fb9`.
Its KDA+dense+mHC numerical check measured relative L2 0.00468 and cosine
0.9999891. KDA and DSA attention repeatability passed. Source inspection proves
this is TP1/B1, despite a maximum-sequence environment setting of eight; this
checkout does not contain the multirow tiers claimed by the old handoff.
DSA reference comparison and routed MLP execution remain absent. PR #863 fixes
discarded probe failures and labels the component coverage explicitly.

The corrected validator passed on clean main `52ce0e5`, build job
`glm-numerical-build-52ce0e5` (attempt `6e673b1919204a818eb5d43575b75fae`) and
run `glm-numerical-oracle-52ce0e5` (`57d766739f974f4abd5f343dcbc8daef`). Both
finished with exit 0 and released Spark0. KDA+dense+mHC metrics remain 0.00468
relative L2 and 0.9999891 cosine. All three formerly ignored numerical probes
also pass with their results now counted. This remains synthetic TP1/B1.
Subsequent audit found the C oracle's separate double-decay error, corrected in
PR #864. Treat the earlier PASS as a scoped execution receipt, not proof of
the reference's correctness. The actual GLM adapter host harness now also
checks B3 admission and verifies that frame-owned cache lanes survive mutation
of the caller's lane array; this does not exercise full cache restoration.

The corrected KDA reference passed on clean main `06be88e`, build job
`glm-kda-build-06be88e` (attempt `7f9b084156a1411ca319329aa68b2acb`) and
run `glm-kda-oracle-06be88e` (`b00d1a70aa624f539d96dcb13d65749a`). Both finished
with exit 0, all participant cgroups stopped and Spark0 released. The run's
configuration SHA-256 is
`06a3495bf92558f341eac53c8a38abb68022bd637f679dbe9b4bc3ed6daab623`.
KDA+dense+HC relative L2 is 0.00376 and cosine 0.9999930; the three intermediate
probes also pass. This remains synthetic TP1/B1, with DSA determinism only.
There is no new qualified distributed throughput result.

The draft's common pinned-transaction tests pass through `build/test_kv_cache`.
They use real arena/page-cache code, including insufficient output capacity
after lane binding: failure releases the new writable page and its binding
while preserving the reusable prefix and other owners' pins. The added owner
records retain exact transaction identity and lane content. Prepare holds
pinned pages; commit authorizes execution; claim prevents abort until GPU
completion. Failed B3 preparation unwinds prior lanes. Stale transactions and
duplicate slots are rejected. GPU failure discards the affected sequence,
including an existing writable page: metadata rollback cannot undo writes.
If one lane fails completion, all sequences in that batch are invalidated so
the scheduler cannot continue with mixed token positions.

GLM's real host harness now checks the admission decisions and dispatch
cookies, claims a committed batch, uploads non-identity physical mappings,
then proves an unchanged mapping performs no additional copy. Its injected
execution error releases both lanes and invalidates the mapping shadow.
The common API has three entry points (admit, claim, finish) in the existing
page-cache implementation; allocation remains at startup. GLM adds its mutex,
CUDA upload and continuity bookkeeping. No other model driver was changed.
This is host execution with copy stubs. CUDA compilation passed for draft
`a525c5a` (run `34281042843`); merged-main serving tests are still required.
The draft still lacks complete KDA/convolution snapshots, restoration and reset.

The GLM adapter also retained stack-local frame/context/buffer descriptors and
borrowed row arrays after submit returned. Its existing pending slots now own
these until completion. Only submitted rows are copied; the per-submission
clear of the entire multi-megabyte pending slot is removed. The real adapter
host test submits B3, returns from the caller and overwrites its arrays, then
checks the saved frame and completes it. Per-completion debug printing is
removed from the production path.
Pending-slot ownership uses an atomic claim/release so the submit thread and
CUDA completion thread do not race on a plain flag; rejection counters are
atomic as well. The host harness also checks eight concurrent claimants for one
slot: exactly one owns it, and the slot is available again after release.

Zero-token release now goes through cache admission and emits a zero-token
completion without launching decode. The module releases common cache ownership
and clears continuity bindings. The shared frame-to-admission helper preserves
zero rather than silently converting it to one: positive DSV4 and Qwen callers
were checked, and the DSV4 runner regression passes. Host tests exercise both
the actual GLM adapter release and the module's release of completed lanes.
This does not yet prove GPU recurrent-state reset or slot reuse.

The direct probe now prepares, commits and submits real cache lane transactions,
then releases all lanes and requires a drained, empty snapshot. Its fake-driver
test verifies order and identity, plus rollback on prepare/commit/admit/submit
failure. It verifies host protocol only. The reusable lesson is to keep async
descriptors in existing owner storage and make benchmarks use the same ownership
protocol as serving, without adding a parallel cache implementation.

PR #865 removes full-head KDA allocation on each TP rank and the repeated Q/K/V
window factor, using the shared kernel's existing stride parameter. At TP16,
the calculated recurrent-plus-window allocation per resident sequence falls
from 155.125 MiB to 8.8984375 MiB across 34 KDA layers. See
`docs/GLM_KDA_STATE_LAYOUT.md` for formulas and the actual allocator host test.
The draft includes this merged layout for checkpoint integration.

Clean main `d62e93e` passed GPU build (`82ef7d87dae8435c80a1d2a34639a107`),
B3 memcheck (`4d3b9fe729b34aa5aa07d1e60b81bc20`, zero errors), resident/lazy
comparison (`f81b2729301842a0ab48a356af25d74c`) and the synthetic numerical
validator (`1dbc3289146147ba8d5640e31c4cafe0`). All queue jobs stopped and
released Spark0. All 44 tokens across five resident/lazy receipts exactly match
the pre-change `06be88e` baseline. KDA+dense+HC relative L2 remains 0.00376,
cosine 0.9999930. This is component qualification: token checks use TP16 rank0
without collectives, and the numerical oracle uses synthetic TP1/B1. No new
distributed throughput result is claimed. Local artifacts are
`/private/tmp/ds4_glm_rank_baseline_06be88e` and
`/private/tmp/ds4_glm_rank_result_d62e93e`.

## Current next steps, not completed work

The draft adds `SparkKvPageStoreReadback` to the existing bounded page-store
worker. It restores one saved generation into caller-owned storage, without
constructing an arena prefetch plan or changing KV residency. The caller keeps
the destination alive and polls the same request through completion; generic
arena progress cannot consume that completion. Invalidation cannot recycle an
in-flight record. Existing storage, copy callbacks and backing-slot accounting
are reused. Real worker/file tests cover byte equality, stale generations,
destination ownership, error propagation and unchanged arena residency.
GLM checkpoint capture, publication and restore still need to be connected to
this primitive. No prefix-hit correctness claim follows from the store test.

1. Complete GLM integration with the shared cache: qualify dynamic mappings on
   the GPU and implement full KV/index/KDA/convolution/continuity restoration.
   Prove prefix-hit execution matches uninterrupted computation. No fixed
   identity page mapping or KV-only snapshot may masquerade as this result.
   PR #862 fixes the identified layer-major/backing-page mismatch and missing
   index payload; its merged-main component checks pass. The payload tests do
   not establish full prefix restoration.
   See that PR's `docs/GLM_KV_PAYLOAD.md` for the regression and reusable lesson.
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
