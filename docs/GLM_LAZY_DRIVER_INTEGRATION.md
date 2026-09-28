# GLM expert residency and lazy loading

This is the authoritative description of how GLM 5.3 Flash (`glm5_next`)
gets its routed experts. Lazy attach through weightd is mandatory; the module
has no direct pack loader. Production pins every expert at attach, which
departs from invariants I28-I30. The first section describes that.

## Serving residency today

The fleet serves with `G5_PIN_EXPERTS=1` in the fleet-agent drop-in
`20-serving.conf`, as recorded in the 2026-09-28 COMPSEC receipt
(`qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/REPORT.md`).
The agent passes it to residentd as `SPARK_GLM5_NEXT_PIN_EXPERTS=1`
(`tools/fleet_node_agent.sh`).

After a successful lazy attach, `SparkGlm5NextPinAllExperts` acquires and
begins use of every routed expert the stage owns, and holds the leases for
the life of the process. At TP16 that is 42 routed layers × 288 experts =
12096 keys (arithmetic; layers 3-44). Leases hold at most
`SPARK_WEIGHTD_LEASE_GROUPS_MAX` = 512 keys each, so the pin takes 24 leases
(arithmetic: 12096 / 512 rounded up). A pin failure prints
`EXPERT-PIN-FAILED` and fails startup. The mode arrived as a bring-up aid in
`28fe1f7` (2026-09-18) and has been required for graphs since `78c2c21`
(2026-09-22).

The fast paths depend on it:

- **Whole-chain graphs.** `SparkGlm5NextGraphClaimExperts` refuses unless
  every expected expert is pinned. It prints
  `GLM whole-chain graph requires N leased experts; held M` and returns
  `UNSUPPORTED`, and the chain then runs eager.
- **Linear eager chains.** `SparkGlm5NextLinearEligible` requires
  `SparkGlm5NextExpertsPinned`.
- **Without pinning**, every chain runs the per-layer state machine below.
  That machine synchronizes the stream at every collective round and at every
  routed layer.

Memory cost: one expert's rank-local slice is about 1.573 MB at TP16 (see
"Batch roofline" in [GLM5_NEXT_ROOFLINE.md](GLM5_NEXT_ROOFLINE.md)). The pinned
inventory is therefore about 19.0 GB per rank (arithmetic: 12096 × 1.573 MB).
That fits the agent's default pool, `G5_EXPERT_POOL_BYTES` = 34359738368 bytes
(32 GiB), which the agent passes as `SPARK_WEIGHTD_EXPERT_POOL_BYTES`.

This is a deviation from the contract. I29 ("Expert residency is bounded")
expects a working set loaded within a budget, with unowned data reclaimed.
Pinning makes the full per-rank inventory the minimum pool size, and nothing
is ever reclaimed. A model or a shared multi-developer node whose per-rank
expert inventory does not fit gets no graphs or linear chains at all. For GLM
5.3 Flash at TP16 the inventory fits, so the fleet runs this way while the
lead dev redesigns it (2026-09-28). TECHDEBT should carry the deviation.

## Plan to restore bounded residency

Commits `8adebc6`, `360c0ee` and `1a674be` (2026-09-16/17) describe a
"relocation/linker" approach: capture a graph once, record which kernel
arguments are expert pointers, and patch them on load. It was never
implemented. `1a674be` deleted the route-union sweep in its favour, and
`78c2c21` then required full pinning instead.

Addresses are not the obstacle. The weightd map reserves one consumer-local
virtual range at create time, and its base is constant for the life of the
map (`SparkWeightdMapBase`, `include/sparkpipe/spark_weightd_map.h`). Every
expert pointer is that base plus a pack offset (`SparkGlm5NextBindLayer` in
the module's `.cu`), so a captured pointer stays valid while its chunk is
committed. What is missing is keeping a bounded working set committed while a
replay runs, and recovering when a replay routes to an expert that is not
committed. The detection side exists: the cover bitmap and miss ring
(`SparkGlm5NextGraphCoverEnsure`, `cuda/layer.cuh`) flag such an expert, and
the module turns a flagged replay into `GRAPH-EXPERT-MISS` and `BUSY`.

The exit criterion for the redesign follows from I29. Graph decode must run
with an expert pool smaller than the full inventory, and it must produce the
same tokens as the pinned build under the gates in
[GLM_PERFORMANCE_GATES.md](GLM_PERFORMANCE_GATES.md).

## Startup and shared code

Use `SparkWeightdLazyPackCreateChecked` from `spark_weightd_lazy_pack.h`. The
composed object owns its manifest, compact spine, attachment and client,
consumer map, and CUDA-context worker. Link `runtime/weightd_sources.mk`; do
not keep private weightd source lists. The GLM Flash and DSV4 module builds
already use the shared list.

The lazy settings are:

| Variable | Rule |
| --- | --- |
| `SPARK_WEIGHTD_SOCKET` | Required. Without it `SparkWeightdAttachRequested` returns `BUSY`, which the module reports as `UNSUPPORTED` (`c67be23`). |
| `SPARK_WEIGHTD_PACK_SHA256` | Required, the pack's 64-character SHA-256 digest. |
| `SPARK_WEIGHTD_EXPERT_POOL_BYTES` | Required, a positive byte count. |
| `SPARK_WEIGHTD_SPINE_BUDGET_BYTES` | Optional. Defaults to 8 GiB. |
| `SPARK_WEIGHTD_ATTACH` | `0` with a socket fails with `INVALID_ARGUMENT`. `1` without a socket fails the same way. |

GLM supplies its module identity, pack revision and TP degree. Lazy MTP fails
with `UNSUPPORTED`. The manifest check (`spark_module_manifest_check_fp8.h`)
accepts only FP8 expert payloads. Attach retries once per second for up to
600 attempts. It logs `LAZY-ATTACH-RETRY` while the attach fails, and
`LAZY-ATTACH-MESH-PENDING` while a TP rank's mesh send buffer is not yet
published.

Startup validates the pack header, directory geometry, inventory and ranges.
The manifest checker then verifies every routed payload and scale range,
including all 288 experts, against that directory, and it rejects extra
ranges before any spine GPU allocation. The consumer hashes the full pack
while copying only the non-expert complement into a compact, 256-byte-aligned
spine. Startup therefore still reads the full pack from disk. Non-expert
tensor pointers come from `LazyPackSlice`; expert entries keep pack offsets
instead of permanent device pointers.

The daemon returns the SHA-256 of its loaded manifest in canonical group
order, covering layer, expert, kind, version, offset, length and range
checksum. The consumer compares that fingerprint with its parsed manifest
before creating mappings or the worker. Daemon and consumer must be rebuilt
together when the IPC frame changes. This prevents manifest disagreement at
attach time. Pack publication and pathname and file identity handling still
need review.

## Route, acquire, execute and release

When experts are not pinned, the GLM TP chain uses the existing slot and
stream throughout each layer:

1. The split Route entry runs normalization, the router GEMM, top-k and route
   grouping. It copies 289 group offsets into preallocated pinned slot storage
   and records the route readiness event. Dense layers use the existing
   combined path.
2. The shared worker waits for readiness outside CUDA and collective
   callbacks. `SparkWeightdRouteKeys` validates the entire prefix, including
   the final `rows * top_k` count, and emits the active layer and expert keys.
3. `MapAcquire` pins and imports all payload and scale chunks for that set. A
   nonzero identifier returned on failure still needs cleanup. `MapBeginUse`
   exposes the consumer-local sparse address space only after acquisition
   succeeds.
4. The wave binds validated pack offsets against that address space and names
   the local layer the lease covers. Shared layer pointers are never mutated.
   A missing or wrong-layer binding yields null expert addresses, which expert
   submission rejects. Never use the daemon's `device_handle` as a consumer
   pointer.
5. After expert submission, `MapRecordCompletion` marks the end of reads. The
   chain drains the stream, releases the lease, clears ownership and resumes
   the MLP reduction. Recording clears the wave address immediately, and retry
   state prevents a second completion recording. IO and BUSY cleanup is
   retried once.

The worker serializes mapping operations on the creating CUDA context, admits
work with a bounded nonblocking queue, and refuses destruction while busy. The
per-layer completion wait gives no compute/communication overlap; this is why
the pinned linear and graph paths exist.

## Failures and destruction

Keep the request slot claimed until acquisition, GPU work and lease cleanup
have all finished. Partial expert launches still need completion recording
and draining. A persistent cleanup or CUDA drain failure keeps the chain, slot
and pins, with a diagnostic. `SparkGlm5NextLazyRetryRetained` retries retained
cleanup on the same worker. Failed cleanup republishes ownership, and
successful cleanup completes the failed request and releases its claims.
Atomic transfer prevents duplicate recovery, and publication happens after
the retaining thread stops using the chain. Permanent errors still keep their
resources, and recovery from disconnected clients is unfinished. A socket
disconnect alone does not release daemon pins, because it does not prove that
the GPU has finished.

GLM destruction waits for slots, then waits with a monotonic timeout for the
worker to go idle before releasing resources. This covers the gap between the
final slot release and the worker task's return. The idle wait requires
submissions to have stopped and rejects calls from the worker itself. Failed
stream drains keep their resources. The idle worker must be joined before the
maps and the spine are destroyed. `LazyPackDestroy` keeps partial cleanup
state on failure and forbids new slices once teardown starts. A non-null
result from a failed Create is for cleanup only. If cleanup itself fails, GLM
startup keeps its resources until the process exits.

## Validation

Host tests cover strict manifests, atomic production fixtures, bounded
multi-range acquisition, rollback, scoped FD imports, overlapping leases, map
completion, worker FIFO and admission, and composed startup with CUDA stubs. A
startup race test changes the manifest after the consumer has parsed it and
confirms `HASH_MISMATCH` before publication. `tests/test_glm5_next_lazy_dispatch.c`
checks route-key selection, the acquire/bind/launch/completion/release order,
partial acquisition, failed launches, release retries, retained-chain
ownership transfer, and GLM range geometry for 288 experts. None of these
tests executes expert kernels.

Open qualification items:

- Bounded residency under a pool smaller than full expert storage, with at
  least two real GPU consumers holding overlapping and disjoint sets (I29;
  see the plan above).
- Cancellation, partial failures, persistent cleanup recovery, and
  disconnected or orphaned consumers.
- Pack publication identity and startup cleanup ownership.

## Two-process mapping probe

Build `build/sparkpipe_weightd` and `build/weightd_lazy_consumer`, then run
`python3 tools/weightd_lazy_pair.py --daemon build/sparkpipe_weightd --probe build/weightd_lazy_consumer`.
The controller owns a fresh temporary fixture and daemon. It starts two
separate consumers with expert sets {0,1} and {1,2}, waits for both to hold
their leases, and then lets them complete and release. The pool holds three
of the fixture's four chunks. While both holders wait at the barrier, a
separate pressure consumer must get `CAPACITY_EXCEEDED` for the fourth chunk.
After they release, another consumer reads experts {2,3}, which requires
evicting unpinned storage within the same pool. Both consumers read payload
and scale bytes through consumer-local maps and compare the contents. Child
processes have bounded waits and are terminated on failure. The fixture
writer is for this test only, not for publishing production models.

The process, FD and mapping orchestration passes locally with CUDA stubs. A
real-CUDA merged-main build must rerun it before anyone claims GPU proof. The
probe tests mapped reads and lease coexistence, not expert kernels or full
model inference.
