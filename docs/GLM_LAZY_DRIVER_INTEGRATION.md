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
