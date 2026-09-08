# GLM lazy-driver integration

Draft #842 implements shared lazy loading and an initial GLM Flash dispatch path. It has not established real GPU numerical correctness or serving performance.

## Startup and shared code

Use `SparkWeightdLazyPackCreateChecked` from `spark_weightd_lazy_pack.h`. The composed object owns its manifest, compact spine, attachment/client, consumer map and CUDA-context worker. Link `runtime/weightd_sources.mk`; do not maintain private weightd source lists. GLM Flash and DSV4 module builds already use the shared list.

Configured GLM requires `SPARK_WEIGHTD_SOCKET`, `SPARK_WEIGHTD_PACK_SHA256`, `SPARK_WEIGHTD_EXPERT_POOL_BYTES`, and `SPARK_WEIGHTD_SPINE_BUDGET_BYTES`. Budgets are positive byte counts. `SPARK_WEIGHTD_ATTACH=0` conflicts with a configured socket and fails. An unconfigured resident path remains available for numerical comparison; failure of configured lazy loading never selects it. GLM supplies its module identity, pack revision and TP degree. Lazy MTP and lazy codecs other than FP8 are currently unsupported.

Startup validates the pack header, directory geometry, inventory and ranges. Its manifest checker then verifies every routed payload and scale range, including all 288 experts, against that directory and rejects extra ranges before spine GPU allocation. The consumer hashes the full pack while copying only the non-expert complement into a compact, 256-byte-aligned spine. This still reads the full pack from disk at startup. Non-expert tensor pointers come from `LazyPackSlice`; expert entries retain pack offsets instead of permanent device pointers.

The daemon returns SHA-256 of its loaded manifest in canonical group order, including layer, expert, kind, version, offset, length and range checksum. The consumer compares that fingerprint with its parsed manifest before creating mappings or the worker. Both sides must be rebuilt together for the changed draft IPC frame. This prevents manifest disagreement at attachment; pack publication and pathname/file identity handling still need further review.

## Route, acquire, execute and release

The GLM TP chain uses the existing slot and stream throughout each layer:

1. The split Route entry runs normalization, router GEMM, top-k and route grouping. It copies 289 group offsets into preallocated pinned slot storage and records the route readiness event. Dense layers use the existing combined path.
2. The shared worker waits for readiness outside CUDA and collective callbacks. `SparkWeightdRouteKeys` validates the entire prefix, including the final `rows * top_k` count, and emits the active layer/expert keys.
3. `MapAcquire` pins and imports all payload/scale chunks for that set. A nonzero identifier returned on failure still requires cleanup. `MapBeginUse` exposes the consumer-local sparse address space only after acquisition succeeds.
4. The wave binds validated pack offsets against that address space and names the local layer covered by the lease. Shared layer pointers are never mutated. Missing or wrong-layer bindings yield null expert addresses; expert submission rejects them. Never use daemon `device_handle` as a consumer pointer.
5. After expert submission, `MapRecordCompletion` marks the end of reads. The initial implementation drains the stream, releases the lease, clears ownership, and resumes MLP reduction. Recording clears the wave address immediately; retry state prevents a second completion recording. IO/BUSY cleanup is retried once.

The worker serializes mapping operations on the creating CUDA context, has bounded nonblocking admission, and refuses destruction while busy. The initial per-layer completion wait is a debugging baseline. It does not implement compute/communication overlap or the measured performance working-set policy.

## Failures and destruction

Keep the request slot claimed until acquisition, GPU work and lease cleanup have finished. Partial expert launches still need completion recording and draining. A persistent cleanup or CUDA drain failure retains the chain, slot and pins with a diagnostic. Lazy teardown submits a recovery pass on the same worker before waiting for slots: failed cleanup republishes ownership; successful cleanup completes the failed request and releases its claims. Atomic transfer prevents duplicate recovery, and publication happens after the retaining thread stops using the chain. Permanent errors still retain resources, and disconnected-client recovery remains unfinished. Socket disconnection alone does not release daemon pins because it does not prove GPU completion.

GLM destruction waits for slots and reports failed stream drains before releasing resources. The idle worker must be joined before destroying maps and spine. `LazyPackDestroy` preserves partial cleanup state on failure and forbids new slices once teardown starts. A nonnull failed Create result is cleanup-only. GLM startup currently retains resources until process exit if cleanup itself fails; recoverable ownership reporting and other preexisting submission-error drain paths still need work.

## Validation and remaining qualification

Host tests cover strict manifests, atomic production fixtures, bounded multi-range acquisition, rollback, scoped FD imports, overlapping leases, map completion, worker FIFO/admission, and composed startup with CUDA stubs. A startup race test changes the manifest after consumer parsing and confirms HASH_MISMATCH before publication. `test_glm5_next_lazy_dispatch` checks route-key selection, acquire/bind/launch/completion/release ordering, partial acquisition, failed launches, release retries, retained-chain ownership transfer, and GLM range geometry for 288 experts. These tests do not execute expert kernels.

CUDA compile run [34247297418](https://github.com/sparkpipe/sparkpipe/actions/runs/34247297418) passed at `ecd0cd5`, including the GLM Flash FP8 translation unit. Later revisions need their own CI evidence. Host syntax and package checks are separate from CUDA execution.

Remaining gates:

- Generate and verify manifests for the corrected real packs.
- Compare resident and lazy results for arbitrary row counts, including 1, 3, 17, 97 and 100 once capacity supports them.
- Run at least two real GPU consumers with overlapping and disjoint sets under a pool smaller than full expert storage; prove consumer-local mappings and bounded residency.
- Qualify cancellation, partial failures, persistent cleanup recovery and disconnected/orphaned consumers.
- Verify pack publication identity and startup cleanup ownership.
- Prewarm/pin measured performance sets, remove unnecessary serialization, and measure metadata transfer, misses, startup hashing, weight reuse and overlap.

GPU/deployment qualification uses reviewed, merged-main queue builds with recorded SHAs and receipts. Passing component tests does not satisfy full serving qualification.
