# GLM 5.3 Flash hill-climbing log

The independent reference reads only the requested BF16 embedding row through
the shared checkpoint reader. Token lookup requires 8192 payload bytes, not the
entire 1268776960-byte embedding matrix. Row offsets, matrix extents, bounds and
short reads are checked. This removes an observed cold Ceph read stall before
the first layer; full-model projection weights are still required by the reference.

The checkpoint reference now accepts `--layers 45` to compute the first token's
logits through the full text model. It adds first-position DSA, routed/shared
experts, the final mean of HC streams, and the output head. DSA uses the single
visible self token at position zero, so its value projection determines the
attention result; this does not test index selection or attention over history.
Routing retains FP32 checkpoint router weights, applies correction bias only to
selection, and normalizes the selected sigmoid scores before expert reduction.
The reference records per-layer arrays and progress. It remains independent of
driver captures and does not model activation FP8 quantization or recurrent
decode. Compare earlier saved prefix arrays before using its final prediction.

2026-09-09 routed-expert TP slicing: shard each up and gate projection over
the same intermediate rows, then concatenate the local halves. Use that same
range for down-projection columns. The previous packer sliced the concatenated
stack, giving a rank only up or only gate rows; the local SwiGLU kernel then
multiplied unrelated channels. Dense/shared packing already used paired slices.
Payload and scale producers now share one slice iterator for routed experts.
Old affected TP packs must be regenerated and verified against the corrected
checkpoint plan before deployment; their unchanged sizes cannot prove validity.

The new numerical pack test consumes actual produced BF16 payloads and compares
summed sharded SwiGLU/down projections with the unsharded computation for TP1,
TP4 and TP16, two experts and three inputs. It fails on the old TP4 packer for
all 384 checked outputs and passes after the fix. The existing payload test
incorrectly expected single-projection rank slabs; that expectation is removed.
This host test establishes sharding semantics, not CUDA or full-model acceptance.
Before correction, independent first-position error rose from 1.84% relative
L2 at layer 3 input to 75.4% at layer 4 input. Preserve the comparison at
`/private/tmp/ds4_glm_layer4_19bbc6e/comparison.json` for the corrected-pack rerun.

The checkpoint layer reference accepts `--layers 3` to compute the entire
initial KDA/dense prefix at position zero, including every residual transition.
Each intermediate array is named `layerN_stage`; `next_attention_norm` is the
following layer's input. Unsupported layer types fail before tensor loading.
The one-layer run on reference revision efb24be, compared with merged TP16
driver 19bbc6e, measured relative L2 0.0129513 and cosine 0.9999163 at layer 1.
This is diagnostic evidence, not acceptance: activation FP8 quantization is
not modeled, and full-model output still repeats. Receipts are preserved at
`/private/tmp/ds4_glm_layer1_19bbc6e/comparison.json` on the coordinator.

The checkpoint layer reference (`tools/glm5_next_checkpoint_layer_reference.py`)
computes layer zero at position zero from a token and checkpoint weights only.
It covers embedding, full-width KDA, dense MLP, both hyperconnection sites and
the next layer's normalized input. It reuses the existing checkpoint decoder
and BF16 conversion helpers. It does not consume driver intermediates, perform
TP reductions, or qualify a complete sequence. FP8 checkpoint weights are
expanded to BF16 and matrix products accumulate in FP32; this diagnostic does
not emulate activation quantization in an upstream FP8 runtime. Compare its
outputs with the first captured step before interpreting subsequent recurrence.

2026-09-09 KDA reference audit: remove the extra BF16 Q/K normalization
between convolution and the shared delta-rule kernel. The recurrence already
normalizes Q/K in FP32 with epsilon 1e-6. Apply the missing query scale
1/sqrt(key dimension) in that shared kernel. Fuse output RMS normalization,
FP32 gain and sigmoid gating before the single BF16 output store. This removes
three launches per GLM KDA layer and restores the reference operation order.
The shared delta-rule scale also affects its other model callers; this change
does not qualify those drivers.

Reference source is Transformers `modeling_glm5_next.py`, retrieved 2026-09-09,
SHA256 `5a885692edc74056f370d70af10ba746c0b12d59d245fb782c8ec8964059ea6e`.
Its recurrent and chunk implementations normalize Q/K once and scale Q;
its gated RMS function keeps normalization and gating in FP32. The Python
oracle now follows those equations. PR #871's intermediate BF16 rounding
matched the old driver but was incorrect relative to the reference and is
reversed. A stage-local oracle must not adopt a driver's arithmetic merely
to eliminate its mismatch.

Host verification executes the production recurrence against an independent
scaled reference over six steps (relative error 0.001921 at BF16 output),
with exact committed-state replay, chunk/decode equivalence and no state
mutation for uncommitted work. Substituting the old kernel fails with relative
error 1.003. A separate small-value fixture checks fused gated normalization
against a double-precision formula, including epsilon-sensitive values.
GPU component and full-model results remain pending merged-main deployment.

2026-09-09 MLP formula audit: the checkpoint's `swiglu_limit=10.0` was
generated into the model header but unused by dense, routed and shared MLP
activation launches. All three used an unclamped SiLU multiplication. The
[Transformers GLM reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py)
limits gate to at most 10 and up to [-10,10] before activation. A shared
`LmClampedUpGateKernel` now implements this formula for the existing up|gate
pack layout. All three launches pass the model constant; the synthetic dense
reference also applies the limits. Host and CUDA fixtures exercise the same
production kernel with signs, boundary values, negative gates, multiple rows
and output guards. GPU and full-model results must be recorded after deployment.

The preceding DSA layer-three capture on `dc48e0c` checked eight single-row
steps. Attention latent and value outputs matched bitwise at every step;
output-projection differences were at most one BF16 ULP. FP32 hyperconnection
comparisons still failed the oracle's 90% exact-bit requirement, with most
relative differences around 1e-6. This is stage-local evidence using captured
upstream inputs, not a full-model reference result. Queue attempt:
`c66a0c5afdcf4564b0510d1a9d5e2a64`; receipts:
`/private/tmp/ds4_glm_dsa_dc48e0c`.

2026-09-09 embedding diagnosis: rank-zero captures on main `dd57197`
contained zero layer-zero inputs for tokens outside its vocabulary shard.
The embedding kernel intentionally emits zeros on non-owning ranks, but the
required HC-wide embedding reduction returned success without enqueueing
whenever `mtp_active` was zero. That field was never assigned. Removing the
gate restores the collective before attention. The host regression executes
the production submission function at B1 and B3, checks deferred completion,
missing collective initialization and enqueue failure. The original function
fails this regression with exit 2. Distributed numerical qualification follows
merged-main deployment; this finding alone does not establish correct output.

The same capture's recurrent-state difference against FP64 recomputation was
approximately 1.1e-7 relative L2 at pass four. The first two captured inputs
were all zero, so those exact comparisons provide little numerical coverage.
PR #871 corrected missing BF16 rounding in the Python output-gating oracle
and prevented normalization mismatches from silently omitting a comparison.
Its merged-main replay removed the gating mismatch without changing tolerances.
For future drivers, verify replicated input availability before judging
downstream kernels; a locally correct computation on zero input is insufficient.

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
| Mandatory serving contract, draft PR #861 | ABI 21 removes seven opt-out bits, makes callbacks/cache geometry mandatory and removes zero-cache scheduling paths. Common host tests pass; GLM GPU cache/reset qualification remains incomplete. | Required means fail explicitly when missing. Callback presence is only structural validation; stubs and flags cannot establish behavior. |
| Accurate ABI probe, draft PR #861 | Replaced copied, incorrect structs/flag values with the public header and common loader. | Diagnostic tools must consume the same contract as production, or they can report misleading capability results. |
| Layered KV/index backing payload, PR #862 / main `d687fc8` | Common gather/scatter with a hardware copy callback; GLM supplies native geometry. Actual GLM host hook tests pass both regions across three layers and five pages; substituting the previous contiguous copy fails the payload check. CUDA CI, merged-main B3 memcheck and resident/lazy/concurrent parity pass. | Trace native layer/page indexing against backing payload layout. Total allocation size is insufficient; distinguish each page and layer in tests. Include index state in payload sizing. |
| Numerical gate integrity, PR #863 / main `52ce0e5` | Three probe failure results were discarded; projection readback could skip a comparison. Checks now affect exit status. Common metrics reject nonfinite inputs and accumulate squared errors directly; regression tests reject the previous metric implementation. Corrected GPU component validator passes on merged main. | Test the acceptance test with bad values. Error-norm cancellation and ignored return codes can turn an optimization regression green. Component repeatability is not numerical correctness. |
| Cache admission wiring, draft PR #861 | GLM builds persistent cache lanes for submitted frames and routes prepare/commit/abort through a common algorithm with a validation callback and caller-owned scratch. Host tests preserve B3 lane identities and transaction generations, propagate driver failure, validate all inputs before dispatch and keep release separate. GPU reset/restoration and serving acceptance remain incomplete. | Common policy builds and validates the transaction; the model supplies geometry and hooks. Keep submitted lane storage alive through device completion. |
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
The draft now has host-tested snapshot, restore and reset paths; merged-main
GPU numerical and lifecycle qualification remain required.

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
GLM capture uses the existing write path and restore uses this
primitive. No prefix-hit correctness claim follows from the store test.

GLM completion now hands off from the CUDA callback to a dedicated instance of
the existing bounded CUDA-context worker (`SparkWeightdWorker`). Cache finish,
publication and lane release execute there; the callback performs no CUDA work.
One queued job per occupied pipeline slot fits the existing queue capacity,
checked at compile time. Startup creates the worker and teardown waits for it.
The real module host harness proves handoff retains ownership until execution
and that a failed handoff does not free in-flight lanes. This prepares the host
context needed for checkpoint processing; checkpoint capture now runs there.
Other MTP callback paths remain outside this non-speculative qualification.

Checkpoint publication must wait for the recurrent state and all three
convolution windows to be saved. Restore must verify the same logical-page
generation and finish before execution. Snapshot eviction must follow common
prefix ownership, and bounded backing-store exhaustion needs coordinated prefix
eviction rather than stale records or unbounded pinned-memory allocation.

The common page cache now accepts a state-record store at startup and reclaims
it with the matching KV record. `SparkKvPageStoreInvalidatePair` checks both
stores under a fixed lock order before changing either. A busy read or a stale
generation leaves both intact. `SparkKvPageCacheEvictUnused` reuses the existing
LRU policy for backing-capacity pressure. Page-cache ABI is 4 because the cache
now retains the attached store. GLM attaches that store at startup; GPU prefix reuse remains unqualified.

This work exposed an existing reclamation error: release dropped the logical
reference before finding that the page remained pinned. The new regression
exits 61 against the old implementation and passes with the fix. Reclamation
now checks ownership before mutation, and LRU skips pinned pages rather than
blocking an otherwise available victim. Real worker/file tests verify paired
invalidation, pinned protection, exact restored bytes and eventual reclamation.

Common prefix publication now requires a completed matching-generation record
whenever a state store is attached. The check runs before publishing or
deduplicating the mutable page, under the page-store lock, without scheduling
a transfer. A missing or stale record leaves the prefix unpublished and the
sequence position unchanged. The real backing-store regression first rejects
missing and wrong-generation records, then saves the correct record and proves
successful publication and paired eviction. GLM now saves the recurrent payload before invoking common completion.

GLM initialization failure previously released its ledger and module object
without destroying the cache worker or freeing cache host allocations. Normal
shutdown and initialization failure now share one cache cleanup helper, which
joins the store worker before freeing staging memory. The host harness queues
a real backing write and verifies teardown closes the descriptor and removes
the worker; partial-allocation fixtures use the same production helper.
Checkpoint backing capacity now covers one KV/index record and one recurrent
record per logical page. The two stores share the configured upper bound;
startup reports the required bytes when an explicit budget is too small.
A zero budget derives this capacity from geometry, replacing the one-KV-page
default. The deployment generators use that derived capacity instead of a
fixed 8 GiB budget. This is a backing-file limit, not a GPU allocation; record
payloads consume disk space as they are written. Reserving record capacity for
all logical pages avoids waiting to evict referenced prefix ancestors.

Successful GLM cache initialization creates and attaches the recurrent store.
Two checkpoint-sized pinned host buffers provide gather/scatter storage and
worker staging; they do not scale with logical page count. The store uses host
copies, with CUDA gather/scatter performed by the model hook on its owning
execution context. Shared cleanup joins the worker before freeing the buffers.
TP1/4/16 host tests verify checkpoint sizes, combined budget arithmetic, exact
budget acceptance and rejection one byte below it. These are host checks;
merged-main GPU initialization and capture/restore remain unqualified.

The GLM recurrent copy hook now describes four existing pools to
`SparkKvPageStoreCopyLayered`: KDA state followed by Q, K and V convolution
windows, each in layer order. A resident sequence slot selects one slice from
every layer. All pool geometry and the complete host buffer are checked before
copying. The host harness round-trips three slots across three layers, verifies
packed ordering and untouched neighboring slots, and rejects missing windows
or malformed buffers without copying. This hook is connected to capture before
checkpoint publication and restore before dispatch. GPU prefix-hit execution
remains unqualified.
Keep the integration narrow: reuse the existing store, worker and layered-copy
algorithm; add only the model layout and the ownership transitions required by
capture and restore.

GLM completion now captures KDA and all convolution windows for each publishing
lane while its transaction still pins the mutable page. The CUDA-context worker
gathers into bounded host storage, saves the matching logical-page generation,
and only then calls common transaction finish to unpin and publish. Copy or
write failure follows the existing whole-batch discard path. An inability to
establish transfer quiescence retains lane/slot ownership rather than reusing
memory still owned by the store.

`SparkKvPageStoreWaitForTransfers` uses the existing worker condition variable;
it does not consume completion results or spin. Submission broadcasts that
condition so a completion waiter cannot steal the worker wakeup. The caller
then consumes the original operation result. Host tests use real KV and state
stores attached before admission, claim a publishing transaction, run actual
GLM finish, and compare the saved bytes. Missing-window failure publishes
nothing and releases the failed sequence. These prove host capture/publication
ordering, not GPU numerical prefix reuse or performance.

Newly bound prefix lanes now seed continuity from their committed cache
transaction, under the cache mutex. This updates only the local validation
view; the frame still must claim its complete transaction identity before any
restore. Dispatch restores the terminal prefix generation while its pages are
pinned, then starts GPU work. Existing continuous lanes retain their current
recurrent state. Missing checkpoint records fail explicitly.

The real-store host fixture now captures in slot 0, releases that sequence,
admits its prefix for another sequence in slot 1, and restores the exact bytes
there despite stale slot-1 continuity metadata. Slot 0 remains unchanged. A
stale dispatch cookie is rejected and a deliberately removed checkpoint causes
restore failure. These checks exercise capture, publication, admission,
continuity and restore together; they do not establish numerical equality of
GPU continuations, full serving reset, or distributed performance.

The common transaction cache now provides quiescent reset. It rejects executing
lanes before changing anything, aborts prepared/committed reservations with the
existing rollback helper, releases sequences, and walks unreferenced prefix
chains to reclaim paired records. It does not repeatedly scan LRU for each
entry. If an external pin or store transfer prevents cleanup, admission must
remain stopped while the caller retries; completed cleanup is preserved.
Tests cover every owner phase, a pinned branched chain and retry, repeat reset,
and real paired backing-record reclamation. GLM now calls this primitive from its reset control path; distributed reset
and GPU lifecycle acceptance remain unqualified.
CUDA compilation of capture/restore head `a55efe559e32bd76ed6defc10367e17a2faff187`
passed in run `34289359047`; this is compilation, not GPU execution evidence.

GLM now implements the mandatory serving reset callback through a zero-row
reset control request in the existing admission interface. This is a control
operation, not an option to disable functionality. The common request validator
rejects reset requests carrying rows or cache lanes. The adapter stops admission,
waits for its pending frames and driver snapshot to be idle, then dispatches
reset; failure leaves admission stopped and success advances its generation.
An atomic owner serializes reset callbacks, and pending reservation rechecks
quiescence after claiming a slot. Submissions from earlier reset generations
are rejected.

The module claims all execution slots and lanes before resetting the shared
cache, clears KDA/convolution pools on its execution stream, and clears mapping
shadows and continuity. It preserves loaded weights and cumulative execution
counters. Successful reset generations cannot repeat; a failed cache cleanup
can retry the same generation. A failed CUDA stream drain retains claims and
reports pending instead of making potentially active memory reusable. Recovery
from that device-failure state still requires lifecycle qualification.

Host tests cover active-driver and adapter reservations, malformed reset shape,
copy-stub zeroing, failed reset/retry, stale generations, reset serialization,
and injected CUDA-drain ownership retention. Common cache, serving-admission
and mandatory-interface tests pass. This does not prove GPU reset, rank-wide
collective restart, continuous traffic or throughput.

The generated driver wrapper must preserve control-operation semantics. It now
allows zero-row reset and does not require free decode capacity for an accepted
cache control or release. A module without admission cannot acknowledge reset.
The generator version changes so cached drivers rebuild. A regression compiles
the actual emitted wrapper and checks these cases; the full driver compiler
test also passes. CUDA compilation of reset head
`43a121bfa34bfa3560010aca7cd759ed50b44972` passed in run `34290859245`.

The existing rank-local driver probe accepts an optional `prefix` argument. It
publishes a 64-token checkpoint, compares four continuation tokens after reuse
in different slots, resets, requires the old prefix to be missing, and compares
fresh execution against its initial output. Host fixtures cover resident B1/B3
and lazy B3 and reject injected continuation/reset mismatches. This probe still
needs merged-main GPU execution; token parity alone does not establish full
numerical or distributed correctness. Extend existing probes and shared control
paths before adding infrastructure; each abstraction needs a concrete caller.

GLM completion now snapshots the completion record, releases finished lane and
slot claims with the existing common helpers, then notifies the caller. The old
callback-first helper intentionally retains claims through a callback; that
contract caused immediate reuse after GLM completion to race cleanup. The GLM
regression reclaims the same resources inside the callback and overwrites the
old slot record, proving the returned completion remains stable. Output copies
and cache completion still precede release; uncertain cleanup retains claims.
Validation also no longer dereferences an invalid submission to print debug
fields after the common validator has rejected it. Both host regressions pass.

The queue-owned comparison runner accepts `--prefix` to run this same probe
for resident B1/B3 and lazy B1/two concurrent B3 consumers. It verifies all 76
steps per row, including replay and reset, before comparing resident/lazy
tokens. Missing, truncated and reordered receipts fail host tests. The mode is
recorded in `RESULT.json`; it remains rank-local, with collectives disabled.
The broader `make -j4 test` at `c6fe4fa` failed in the older DSV4 adapter test:
its load-success assertion conflicts with mandatory rejection of its missing
reset callback. This is an unresolved non-GLM acceptance failure. No stub or
exception was added; the full host suite is not passing.

Merged-main `3278d6c` passed the rank-local GPU prefix/reset comparison on
Spark0: resident B1/B3, lazy B1 and two concurrent lazy B3 consumers produced
836 matching token records. B3 prefix/reset also passed CUDA memcheck with zero
errors. All three queue attempts exited 0 and released ownership. The exact
source, pack/driver hashes, commands and scope are recorded in
[the PR #861 receipt](https://github.com/sparkpipe/sparkpipe/pull/861#issuecomment-5593709181).
Artifacts: `/private/tmp/ds4_glm_prefix_3278d6c/`. This does not qualify full-model
numerical accuracy, distributed serving, continuous traffic or throughput.

The next full TP16 run built successfully on all 16 nodes but failed before
inference: generated runtime limits still contained zero KV page capacities.
The mandatory validator correctly rejected them. Both GLM generators now
derive capacities from resident slots and context length. The host regression
loads the complete deployment through the C validator and checks its capacity
against the firmware's page size; previously it loaded only the stage config.
TP4xPP4 still needs adapter geometry and boundary integration beyond this fix.

Persistent API testing on merged `0f2b0a3` exposed a slot-release contract
mismatch. GLM still advertised position-zero reuse, so the common batch engine
completed requests without sending RELEASE. Live inspection found sequence
100001 still owning slot 0 after its 64 output tokens. The next concurrent
requests repeatedly returned BUSY when that slot was reassigned. One of three
requests completed; two timed out at 180 seconds. The API stayed alive and its
health endpoint returned OK despite the stalled work. This is not successful
continuous-batching qualification.

The common contract now requires explicit release for every bound sequence.
ABI 22 removes the driver's slot-reuse policy field and all three options;
the scheduler always queues release and residentd never accepts a new owner
over an existing binding, including at position zero. Old ABI adapters fail
loading. Testing a release callback directly does not prove the scheduler
will invoke it; regression coverage must exercise ownership transitions.
The driver contract is firmware: implement efficient inference through fixed
required operations. Common code owns scheduling, batching, cache ownership,
release, cancellation and lifecycle. Model hooks describe math and state;
backend hooks implement hardware operations. Do not add capability switches
to accommodate an incomplete implementation. Missing required behavior blocks
acceptance. Topology and hardware differences belong in explicit configuration
and narrow hooks, not exceptions to ownership or correctness.
Use one persistent engine per deployed code/configuration version for all
benchmark phases; request completion must preserve weights and connections
while releasing sequence ownership. Replace the engine on deployment changes
or unrecoverable failure. HTTP completion latency includes prefill and is not
a decode-throughput metric. The 0f2b0a3 outputs remain incoherent, so matching
tokens between runs establish consistency only, not numerical correctness.

Merged `f1f7825` was rebuilt on all 16 Sparks (queue attempt
`1f5a1455a3d544da93a67ede58b44d8e`, all exits zero and reservations released).
The versioned resident/API services then passed B1, three concurrent requests,
and B1 again: five requests, 64 matching tokens each, one unchanged API process,
and all ranks still active afterward. HTTP elapsed times were 8.98 seconds,
approximately 12.71 seconds for each concurrent request, and 8.76 seconds.
These include prefill and are not decode throughput. Output remains incoherent.
Receipts: `/private/tmp/ds4_glm_persistent_f1f7825/` on the controller.
Persistent services still need integration with queue memory accounting.

The API queue-lifetime test previously searched source strings and compiled a
binary without running its claimed queue scenario. Its replacement executes
the real HTTP connection/enqueue/completion path over socket pairs, supplies
controlled completion events, finishes B before A, enqueues C, and verifies
the A/C replies and empty queue. It does not run an inference engine. A separate
negative control removed tail recovery and was rejected by the test deadline.

Head-selection inspection found another numerical defect: B1 passed the rank
vocabulary offset to the common certified head, whose final reduction already
adds it, and then the GLM TP maxloc pack added the offset again. Batched full
head candidates were local and received only the latter offset. Keep B1
candidates local as well. `make test-glm-head-offset NVCC=/usr/local/cuda/bin/nvcc`
executes the production head path on synthetic BF16 inputs with a known local
winner at B1/B3 for every TP16 rank. GPU qualification of this change is pending;
this defect alone does not explain the existing rank-zero-range output stream.

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

## 2026-09-09: corrected expert packing and unresolved numerical drift

PR #877 (`e38ebb2f6df03079a4839fd0423cf18dfb3b521e`) corrected TP expert
packing: every rank now receives matching slices of up and gate projections
and the corresponding down-projection columns. All 16 destination packs were
checkpoint-verified and their hashes checked before deployment. The earlier
layout test had repeated the incorrect packing rule; its replacement also
executes sharded SwiGLU/down-projection mathematics against an unsharded result.

For token 9880 at position zero, the layer-4 attention-normalized input now has
relative L2 error 0.01211963128298521 against the independent checkpoint prefix
reference, cosine 0.9999265670776367 and maximum absolute difference 0.01171875.
The former relative L2 error was approximately 0.7542. This is improvement,
not numerical acceptance. The receipt explicitly records `qualified: false`.
The reference source is `d3fbf5e4fdb758421c06bee0ef227f6ae77556f4`.

Do not attribute this discrepancy to FP8 activation quantization. The deployed
GLM expert calls use the default `SPARK_ACTIVATION_CODEC_NONE` in
`runtime/gemm.cuh`: FP8 expert weights with BF16 activations. The spine calls
`Glm5NextLaunchBf16Linear`. A previous conversational explanation suggesting
activation quantization was unsupported by those actual call sites.

Two rounding-boundary differences require separate measurement:

- `tools/glm5_next_checkpoint_layer_reference.py:rms` preserves the upstream
  BF16 boundary between normalization and gain multiplication. The production
  `LmFusedResidualRmsNormKernel` retains FP32 until after gain multiplication.
- The reference rounds SiLU to BF16 before multiplying by up. Production
  `LmClampedUpGateKernel` retains FP32 through the multiplication and rounds
  the product once. Both paths apply the model's required clamping.

An isolated RMS calculation used the captured 4096-element collapsed input,
the first 256 captured gain values, and the corresponding 256 GPU output
values. With the model epsilon of 1e-5, the single-round calculation matched
all 256 GPU values exactly. The upstream two-round calculation differed on
77 values: relative L2 0.0030511226505041122 and maximum absolute difference
0.001953125. This proves a local rounding difference; it does not establish
the cause or acceptable size of the complete prefix discrepancy. The gain
capture covers only 256 elements, and this observation covers one position.

Keep the upstream semantic reference independent. A production-arithmetic
diagnostic may separately model fusion and TP reduction order, but must never
replace that reference or turn unexplained drift into an automatic pass.
Next isolate each operation using identical captured inputs, then propagate
the measured differences through layers and recurrent decode positions.
Report both implementation error and departure from upstream arithmetic,
including routing decisions, logits and generated-token agreement. Define
acceptance from that evidence rather than choosing a tolerance that passes
the present output.

The corrected deployment produced the expected answer 70 for the math fixture,
but continued into a fabricated user turn because required model EOS defaults
are missing from common serving configuration. Functional acceptance remains
failed. No accepted warm-decode throughput follows from these requests.

Controller artifacts:

- `/private/tmp/ds4_glm_paired_deployment_receipt.json`
- `/private/tmp/ds4_glm_layer4_e38ebb2/comparison.json`
- `/private/tmp/ds4_glm_layer4_e38ebb2/residentd.log`
- `/private/tmp/ds4_glm_layer4_e38ebb2/rms_rounding_analysis.json`
- `/private/tmp/ds4_glm_chat_checks/math-receipt.json`

## Deterministic direct TP reduction

The GLM direct-all-to-all callback started with each rank's local contribution
and folded all other ranks in ascending order, rounding to BF16 after every
addition. Different local ranks therefore used different arithmetic orders.
For contributions 256, 1 and -256 this can produce either 0 or 1, violating
the requirement that an all-reduce return the same result on every rank.

The shared `inference/kernels/tp_reduce.cuh` kernel reads contributions in
ascending global rank order, accumulates in FP32, and rounds once to BF16.
The local contribution occupies its global rank position even when it aliases
the destination. The GLM callback launches this kernel once instead of issuing
one pair-add launch per peer. Existing tree reductions are outside this change.

The host execution test exercises the actual kernel with cancellation, every
local rank at TP1/TP4/TP16, row counts 1/3/17, and a width with a thread-loop
tail. Restoring the previous local-first BF16 arithmetic makes that test fail.
GPU execution and the full-model discrepancy must still be measured on a
merged-main deployment; this fix alone is not numerical qualification.

## TP16 tree subgroup fold ordering

The common tree stage exchanging four subgroup partials previously folded
peers as their arrival nonces appeared, starting from each root's own subgroup.
This allowed both network arrival order and local root identity to change BF16
addition order. The stage now waits for all three peer nonces, places all four
partials at their global root indices, and invokes the shared all-contribution
callback once before packing the next stage. TP16 creation requires this
callback; missing support fails explicitly. Earlier pairwise tree stages and
U64 maximum operations retain their existing behavior.

`test_tp_tree_fold` executes the production readiness/fold helper for all six
arrival permutations on each of the four roots. It verifies no fold before
all inputs arrive, one fold only, local-pointer placement and equal sums under
cancellation. This is host orchestration coverage, not GPU or full transport
qualification. The broader `test_tp_device_collective` currently fails at its
`TestCreate` initialization assertion on unchanged `df24e05` as well; that
baseline failure must not be reported as a passing suite.

Logical-batch selection remains outstanding: the current payload threshold
keeps ordinary hidden reductions direct through eight execution rows. The B3
measurement of 40.6141 aggregate tok/s on `df24e05` therefore does not qualify
the requested B2+ tree policy.

## Required model EOS metadata

Resident deployment ABI 3 carries `eos_token_ids`. Common batch-engine startup
rejects missing, empty, excessive or duplicate model EOS metadata, and rejects
IDs outside a declared tokenizer vocabulary. The transport deployment reader
can still inspect deployments without generation metadata; such a deployment
cannot connect a generation engine. There is no no-EOS serving mode.

The engine copies the model EOS list and appends caller stop tokens. Caller
stops are additive: an empty list or a different token cannot disable model
termination. EOS applies to generated tokens, not prompt tokens. The existing
common completion/release path handles EOS just as it handles output-budget
completion. The API uses the same model list for text decoding and no longer
reads `SPARK_EOS_TOKEN_IDS`.

`tools/glm5_next_gen_deployment.py` obtains the three GLM EOS tokens from
`model_contracts/glm53_flash_authoritative.json`. Other deployment producers
must likewise populate authoritative model EOS metadata before their outputs
can serve. Old binaries must be rebuilt for the deployment ABI change.

Host qualification executes the real common pipeline with fixture ranks. It
submits a prompt containing model EOS, uses a different additional caller stop,
observes generated model EOS, and verifies completion and transaction drain.
Startup tests reject missing metadata even when a caller stop token is supplied.
Parser tests load valid metadata and reject empty lists, repeated IDs, repeated
JSON keys and oversized lists. These tests do not establish GLM GPU completion;
that requires the merged-main deployment and repeated chat/lifecycle requests.

## Persistent-engine decode measurements

Batch-engine ABI 6 events expose coordinator monotonic nanoseconds and the
request's reused prompt-prefix token count. The count comes from common prefix
lookup, not an inference from faster elapsed time. The API records token event
timestamps in the request and writes one JSON `request_measurements` record
after responding. It performs no per-token logging. CLI batch events expose
the same metadata.

Use `python3 tools/glm5_next_bench_wrap.py --api-log CAPTURE` on records from
the intended measurement requests in one API process. Preserve the source,
configuration, prompt and output receipts separately. The parser rejects
failed/incomplete requests, invalid timestamps and mixed process identities.
It reuses the existing all-sequences decode window: after the latest first
token, through the earliest final token. This excludes remaining prefill and
completion tails. Per-request timings begin at common engine acceptance;
they do not include earlier HTTP handling. Generated-token counts include EOS.

A prefix-hit count is cache-path evidence, not proof that every restored KV,
index, recurrent and convolution value is correct. Compare cached output with
the corresponding uncached numerical reference. Likewise, a calculated rate
does not confer numerical or SOTA performance qualification. Short or empty
overlap windows are unsuitable for performance acceptance.

### TP16 measurements on bc679ff

All 16 Sparks rebuilt clean merged main
`bc679ffa7374d4bed158930a89392346248aac1b` in queue attempt
`c09daf0eb745420fa78bde187ceff6bd`. Artifact and corrected expert-pack hashes
were checked before deployment. One persistent API process served all 21
requests. Hardware: 16 NVIDIA GB10 nodes, driver 580.159.03; non-speculative
GLM Flash, FP8 expert weights and BF16 spine/activations. These measurements
use the same 69-token math prompt, 151 generated tokens including EOS, and
the current 512-position context / 8-active-request configuration.

| Concurrent requests | Aggregate decode tok/s | Relative to warm B1 |
|---|---:|---:|
| 1 | 13.8191 | 1.00x |
| 3 | 37.7776 | 2.73x |
| 5 | 59.7353 | 4.32x |
| 8 | 87.2346 | 6.31x |

The cold request reported zero reused prompt tokens and 5.1986 seconds from
engine acceptance to first token. The repeated request reported 64 reused
tokens and 0.3683 seconds. Its decode-only window covered 150 intervals in
10.8545 seconds. All concurrent requests reported 64-token prefix hits and
matched the cold output token-for-token. Single requests after each concurrent
group also matched, with the API PID unchanged.

This is a shared-prompt, high-reuse workload, not a diverse serving-capacity
benchmark. The overlap windows contain 445 tokens / 11.7795 seconds at B3,
736 / 12.3210 at B5, and 1172 / 13.4350 at B8. B8 reaches about 79 percent of
ideal scaling from this B1 baseline. B8 is the configured occupancy limit;
these results do not locate the compute crossover near 100 requests.

Timing evidence is valid for this fixture, but full numerical, functional and
hardware-normalized SOTA acceptance remain incomplete. The previously observed
1.21 percent layer-4 relative L2 discrepancy is still unexplained in aggregate.
Neither matching repeated output nor correct answer 70 resolves it. TP4 and
TP4xPP4 comparisons, diverse prompts, larger occupancy and profiling remain.

The B8 run exposed interleaving between ordinary status logging and an atomic
measurement record. One complete JSON record followed a plain-text prefix on
the same line. A JSON decoder recovered it, and all 21 consecutive unique
request IDs were verified before calculating B8. The line-oriented reader
alone had found only 20 records. Lock the complete ordinary log message as
well; the concurrent logging test rejects the old unlocked implementation and
accepts the fix. Do not silently accept a benchmark missing expected requests.

Controller receipts are under `/private/tmp/ds4_glm_measure_chat/`, including
hardware metadata, cold/warm and occupancy request receipts, complete logs,
`all-measurements.json`, and `cold-metrics.json`, `warm-metrics.json`,
`b3-metrics.json`, `b5-metrics.json`, `b8-metrics.json`.
Deployment provenance is `/private/tmp/ds4_glm_measure_deployment_receipt.json`.
