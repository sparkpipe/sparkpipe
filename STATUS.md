# SparkPipe status

## Implemented

- One low-level model-description JSON input contract.
- One-shot compilation of every model stage from one JSON parse.
- Exact stage-target and module-ID resolution.
- Content-addressed validated firmware link-unit library.
- Relocatable-object and normal-static-archive module artifacts.
- Explicit rejection of thin archives because they are not self-contained.
- Validator execution only for a new artifact contract; identical publication reuses the passing record.
- Ordered validator contract arguments included in the validation identity, so a changed tolerance or latency ceiling causes one new validation.
- Stored-byte and link-unit-kind verification before and after validation.
- Read-only library artifacts and separate read-only package copies.
- Direct collection of only selected link units.
- Generated C with explicit operation calls and no hot-path operation loop.
- Direct archive linking, including extraction of private helper members required by the firmware entry point.
- One model-driver shared object per stage with one stable scheduler ABI.
- Hidden module and archive-helper symbols; only `SparkModelDriverGetInterface` is public.
- One package receipt embedding the source JSON and complete resolved artifact map.
- Fail-closed package transaction that removes stale or partially rebuilt firmware.
- Separate offline compiler and serving-runtime archives.
- Driver loader with one-time ABI and target binding.
- Heterogeneous-target orchestrator with pre-resolved numeric routes, firmware-identical replicas, driver admission decisions, opaque dispatch-slot handoff, per-program inflight accounting, runtime snapshots, and completion retirement.
- Tests proving compilation, loading, routing, and execution do not rerun validators.
- First exact CUDA firmware source under `modules/glm52_resident_sparse_mla/`.
- Fixed GLM 5.2 BF16 resident sparse-MLA boundary with adjacent-pair query/key RoPE, current-token placement into the final paged KV layout, sparse cached attention, and stream-ordered completion.
- Correctness-first GLM 5.2 resident decode-stage CUDA firmware source under `modules/glm52_resident_decode_stage/`.
- Decode-stage CUDA covers attention RMSNorm, BF16 Q/KV projection kernels, native DSA sparse-token selection, RoPE plus final-layout KV write, sparse MLA, attention output projection, residual, final RMSNorm, restricted-vocabulary logits, MXFP4 E2M1/E8M0 MTP draft logits, verify/commit/rollback counters, phase clock markers, and stream-ordered completion.
- Driver ABI v3 adds optional direct module admission and runtime snapshot symbols, program scheduling profiles, opaque dispatch-slot validity, dispatch generations/cookies, residency tokens, CUDA graph replay counters, stale-admission counters, and no-host-staging/device-memcpy counters without exposing LLM internals to SparkPipe.
- Model-specific node binding for caller-owned streams, resident device buffers, paged KV storage, RoPE tables, CUDA graph slots, and capacities; the sparse-MLA and decode-stage drivers choose opaque dispatch slots through admission functions and report zero memcpy/host-staging counters plus private queue pressure.
- Cold-build module workflow with dependency files and self-contained archive creation.
- Target-hardware sparse-MLA validator source covering nonzero first-block offset, block-table remapping, fused KV write, RoPE, full 64-head output comparison, completion semantics, and a mandatory maximum submission-to-completion wall-clock threshold.
- Target-hardware decode-stage validator source covering full-shape resident device buffer allocation, backend submit, stream-ordered completion, phase markers, MTP counters, latency ceiling, and post-package generated-driver/orchestrator submit.
- End-to-end host control-path test proving one-time publication, validation reuse, complete JSON-to-package compilation, hidden archive symbols, direct driver execution, slot ownership, asynchronous completion, and quiescent destruction.
- Remaining iteration-073 CUDA implementation work preserved outside the active build as candidate source.

The active documentation authorities are:

```text
README.md                              use and build flow
SPEC.md                                production architecture contract
STATUS.md                              current implementation boundary
docs/LLM_DEVICE_DRIVER_INTERFACE.md    scheduler/driver handoff contract
docs/CUDA_FIRMWARE_SOTA_METHODS.md     CUDA firmware handoff methods
docs/HANDOFF_RELEASE_079.md            release-specific handoff summary
```

## Removed

- overlapping execution-plan types;
- generic graph and adapter execution;
- readiness bundles and inference gates;
- live blocker registries;
- runtime qualification-file scanning;
- production fallback paths;
- universal kernel registries;
- monolithic linking of every implementation;
- per-iteration architecture and remaining-items paperwork.

## Current boundary

- The sparse-MLA and decode-stage CUDA firmware archives have been compiled with CUDA 13.0 for `sm_121` on a GB10 Spark node, admitted by hardware validators, published into the module library, compiled into generated drivers, and loaded from packaged `model_driver.so` outputs.
- The decode-stage package target additionally runs a generated-driver/orchestrator submit smoke test after package compilation. This proves driver load, route resolution, admission, CUDA backend submit, stream-ordered completion, runtime snapshot counters, and zero host-staging/device-memcpy accounting through the LLM driver boundary.
- The sparse-MLA module covers resident sparse MLA plus fused RoPE/current-token KV placement.
- The decode-stage module fills the first CUDA gaps around projections, native DSA selection, restricted logits, and MTP draft/verify, but it is still correctness-first code. The current validator is a hardware execution and control-path gate, not a full GLM logits equivalence proof.
- MoE expert execution, resident transport handoff, and tensor-core/persistent-kernel optimization remain outside the new decode-stage archive. CUDA graph state and replay hooks are present, but target execution and graph-update debugging remain to be done on hardware.
- Publication is intentionally impossible without a user-supplied maximum full-stage latency and a target-hardware pass; a slow implementation must be optimized, not accepted because it is numerically correct.
- The orchestrator currently manages local driver instances. Remote node agents and wire transport are unfinished.
- The JSON selects exact prebuilt modules. A checkpoint plus deployment-profile importer that emits this low-level language is not implemented.
- Inter-stage deployment topology and fixed buffer contracts are not yet compiled automatically.
- Package signing and immutable deployment activation are not implemented.
- The Mac development environment has neither `nvcc` nor compatible CUDA hardware. CUDA claims must come from the Spark hardware validators.

## Next engineering sequence

1. Replace zeroed smoke tensors with deterministic nonzero GLM tensor fixtures and CPU/reference checks for KV writes, cached attention, restricted logits, and MTP verify/commit.
2. Run multi-token cached attention with context length greater than one and verify KV read/write checksums across positions and block-table remapping.
3. If the latency ceiling fails with real tensor fixtures, replace the correctness-first projection, DSA, attention, logits, and MTP kernels with measured tiled, tensor-core, persistent, or graph-captured implementations; do not publish the slow artifact.
4. Add resident MoE expert execution, graph capture, and transport handoff inside one or a few model-specific GLM firmware archives.
5. Publish only exact archives that pass numerical and model-stage performance qualification, then compile the GLM model JSON into direct-call drivers.
6. Measure end-to-end stage latency and throughput before deciding whether another module split or fusion is profitable.
7. Add the remote node agent and fixed submission/completion wire path without changing the driver ABI.
