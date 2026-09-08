# GLM lazy-driver integration

Current integration audit: draft #842 at 69fa227. These are remaining implementation requirements, not a production acceptance claim.

## Existing call path

In glm5_next_resident_decode_stage, SparkGlm5NextPackLoadEntry loads whole tensor entries through SparkStageModuleLoadDeviceRegion and SparkGlm5NextPackAssign stores permanent payload/scale pointers. SparkGlm5NextRunLayerMlp binds those pointers and calls Glm5NextLayerMoe. That function submits normalization, router GEMM, top-k, LmRouteBuild, both expert GEMMs, and shared expert work without returning to the runtime between routing and expert reads. The TP chain enters this through SparkGlm5NextLaunchCudaLayerMlp before REDUCE_MLP.

Replacing attach alone cannot make this lazy. The route-to-expert submission boundary and all failure/cancellation exits must participate in lease ownership.

## Required integration sequence

1. At initialization, load the strict v2 manifest and verify pack identity. Allocate compact spine storage within the explicit consumer budget; use SparkWeightdSpineLoad and SparkWeightdManifestSpineSlice for non-expert entries. Preserve expert source offsets instead of loading their full payloads. Create SparkWeightdMap for the lazy attachment and keep the attachment/client alive until map destruction succeeds.
2. Split Glm5NextLayerMoe into route and expert phases while preserving the resident path numerical order. Route phase includes LmRouteBuild. Copy its 289 group offsets to preallocated pinned host storage and record a readiness event. Validate offsets are monotonic and terminate at rows times top-k; acquire the layer/expert keys whose adjacent offsets differ. This transfers bounded per-expert metadata rather than all token routes.
3. Poll readiness from the runtime worker, then call SparkWeightdMapAcquire. Do not call CUDA mapping APIs inside a CUDA host callback. The selected set is deduplicated and covers every weight and scale range. On acquisition failure, submit no expert kernels and propagate the exact error.
4. BeginUse exposes consumer-local sparse VA. Bind expert entry offsets against that VA; never use the daemon device_handle as a consumer pointer. Submit the expert phase only after successful acquisition/import. RecordCompletion after all streams that read the expert lease have joined the recording stream.
5. Retain the identifier until MapRelease succeeds. BUSY is pending GPU work, not permission to evict. A nonzero identifier returned with an acquisition error still requires cleanup. Cancellation must drain already submitted work before recording completion/releasing; an IPC failure cannot prove GPU completion. Retain retry state for cleanup failure.
6. Prewarm/pin the measured performance working set explicitly and separately from on-demand debug loading. Measure host metadata transfer, acquisition misses, full-pack startup hashing, and GPU overlap. Do not silently switch to eager loading when any lazy operation fails.

## Required acceptance evidence

Compare resident and lazy paths numerically for arbitrary rows including 1, 3, 17, 97, and 100 once execution capacity supports them. Exercise at least two consumers with overlapping/disjoint routed sets under a pool smaller than full expert storage; demonstrate actual consumer-local CUDA maps and bounded residency. Cover queue cancellation, partial acquisition, missing/corrupt manifests, checksum failures, pending completion, and disconnected consumers. GPU/deployment qualification must use merged-main zero-drift queue builds. Current host/stub tests do not establish these gates.

## Route/expert boundary implementation

The draft now splits Glm5NextLayerMoe into Route and Experts helpers and exposes SparkGlm5NextLaunchCudaLayerMlpRoute/Experts. The resident wrapper invokes both in order; dense layers complete in Route and Experts is a no-op for those layers. Kernel submission bodies were preserved during extraction. The lazy runtime state machine is not connected yet: it must retain the same slot/stream between calls and satisfy all lease gates above. CUDA CI now includes the glm5_next FP8 translation unit; compilation and numerical parity must be checked on the resulting revision before qualification.

## Shared build dependency list

Use runtime/weightd_sources.mk (SPARKPIPE_WEIGHTD_SOURCES) when a module links the shared weightd client. GLM Flash and DSV4 now consume that list, as do the runtime/model-common libraries. Do not retain a private two-file weightd/attach list: the daemon/client now depends on manifest and lease code, and mapping/spine integration uses the same shared set. GLM Flash host-source syntax checks and the host working-set test passed after this change; CUDA linking and GPU execution remain separate gates.

The explicit split Route entry point now queues the 289 group offsets to per-slot pinned host storage, allocated with existing startup staging. Its caller must establish stream-event completion before inspecting those offsets. Dense layers skip the readback. The combined resident entry point does not call this readback API, so this adds no metadata transfer to resident decode. Host syntax checks pass; real readback/event and CUDA compilation gates remain pending on this revision.

Each slot now creates a disable-timing route readiness event at startup and destroys it with slot host resources. The split Route entry records it after readback (or dense work), and PollCudaLayerMlpRoute rejects an unrecorded event and otherwise returns CUDA completion status. Calls remain serialized under slot ownership; do not begin another route before consuming the current result. Runtime polling/acquisition and cancellation integration remain outstanding.

## Host worker boundary

SparkWeightdWorker now provides a shared serialized worker bound to the creating CUDA context, with 64 queued tasks plus one executing task and no submission allocation. It refuses destruction while work is queued/executing and joins when idle. Each task owns its deadline, callback/context lifetime, and lease cleanup. Use this worker to wait for route readiness and perform blocking acquisition off the collective progress thread. The GLM chain is not yet connected to it. Host FIFO/admission/teardown tests pass with CUDA stubs under ASan/UBSan; the CUDA context binding signature was verified against the installed Spark toolkit.

CUDA CI run 34243438798 passed on c08eda0, including the GLM Flash FP8 compilation gate and the route/expert split plus readback readiness API. This is compile evidence only, predates this worker, and does not establish numerical or serving correctness.

## Composed startup API

SparkWeightdLazyPackCreate now owns the strict manifest, compact spine allocation, lazy attachment/client, expert map, and shared worker as one startup result. It takes explicit socket/identity/expert-pool/spine-budget inputs, validates the full pack before publishing slices, and has no eager fallback. Spine allocation charges up to 255 alignment bytes against the explicit budget. LazyPackSlice rejects expert ranges. Destroy requires caller-drained spine GPU readers, refuses outstanding worker/map ownership, and retains partial cleanup state on failure. A nonnull failed Create result is cleanup-only.

The composed host test uses its own daemon instance, checks missing manifests, insufficient budget, CUDA allocation failure, incorrect SHA, exact non-expert bytes, expert-slice rejection, and lease-protected teardown. It passes under ASan/UBSan with CUDA stubs; GLM host syntax checks also pass. This startup API is not yet called by GLM production initialization. Mutable pack/manifest publication, real GPU qualification, and full chain dispatch/cancellation remain open integration gates.

## Destruction ordering

GLM already calls SparkStageModuleWaitForSlots before stream synchronization and resource release. Keep the slot occupied for the entire acquisition task, kernel use, and lease cleanup so this existing quiescence gate covers host work as well. Join the idle weightd worker before destroying maps/spine state. Do not add a second family-specific shutdown queue. GLM destruction now reports a failed stream drain and retains its state instead of freeing resources after an ignored CUDA synchronization error. This protects teardown but is not a completed lazy-chain integration.

## Wave-local expert addresses

Layer weights now retain their validated pack payload/scale offsets independently of resident pointers. A lazy wave binds these offsets against its consumer-local lease base and identifies the local layer covered by that lease. An absent base or mismatched layer yields null expert addresses; it never selects resident pointers. Routing no longer requires expert weight addresses, while the expert phase checks all four payload/scale pointers before submission. This allows routing to produce the working set before acquisition without mutating shared layer weights. The runtime still must acquire the complete routed set, bind only after BeginUse, retain the lease through completion, and clear the wave binding before advancing layers. Startup and chain dispatch are not connected yet.

## Configured GLM startup

Configured weightd startup now calls the shared LazyPackCreateChecked path. SPARK_WEIGHTD_EXPERT_POOL_BYTES and SPARK_WEIGHTD_SPINE_BUDGET_BYTES are required positive byte budgets, alongside the socket and exact pack SHA. The module supplies its model identity, pack revision, and TP degree. It validates every FP8 expert payload and F32 scale range against the already validated directory and rejects extra ranges before spine allocation. Non-expert entries bind compact spine slices; expert entries retain only offsets. Missing or malformed manifests and all lazy errors abort startup with no eager retry. Unconfigured resident loading remains the explicit comparison path. Lazy MTP is unsupported.

The shared startup checker is tested to reject model geometry before GPU allocation: an injected allocation failure remains pending after checker rejection. GLM host syntax passes. This draft still lacks the chain acquisition dispatch, so configured lazy routed execution cannot yet complete; it must not be merged or deployed for serving until that dispatch and cleanup are integrated. Failed startup cleanup retains CUDA resources until process exit when destruction itself fails; recoverable ownership reporting remains unfinished.

## Initial chain acquisition dispatch

The TP chain now dispatches lazy routed layers through the shared worker. It waits for the route event, builds the expert key set from group offsets, acquires and begins the lease, binds the current layer consumer VA, submits expert work, records completion, drains that stream, releases the lease, and resumes MLP reduction. Dense layers use the existing path. Each slot stays claimed throughout the worker task. This first path serializes the layer completion wait on the worker and is a correctness/debugging baseline, not the performance overlap implementation.

Partial acquisition identifiers are released without GPU use. Partial expert launches still record/drain/release. Failed release or CUDA drain retains the chain, slot and pins with a diagnostic until process exit; retry/recovery and cancellation qualification remain unfinished. Other preexisting submission error paths still need the same drain audit. The host test test_glm5_next_lazy_dispatch exercises real route-key extraction with CUDA/map stubs and checks acquisition/binding/launch/completion/release order, partial acquisition, failed launch and retained release ownership. It does not test kernels, actual IPC mappings, collective threads or full serving. This supersedes the earlier note that chain acquisition was entirely disconnected; no GPU qualification is claimed.
