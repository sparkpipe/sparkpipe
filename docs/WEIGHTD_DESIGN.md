# spark_weightd — the weight-residency daemon (operator design, 2026-08-30)

THE PROBLEM: every process start re-reads and re-uploads every pack
byte per rank (fopen + sequential per-tensor reads → device memory).
With the operator's constraint — NO budget for 2× RAM during
transitions — and the dev-cycle pain (code redeploy paying a full
reload), the fix is a residency daemon.

## The design (as specified)

OWNERSHIP: one spark_weightd per node owns the weight arenas. VMM API
(cuMemCreate + CU_MEM_HANDLE_TYPE_POSIX_FD), loads the stagepack
once, verifies content hash + geometry fingerprint once, exports
shareable handles. Consumers: cuMemImportFromShareableHandle +
cuMemMap at startup.

IDENTITY-KEYED ATTACH: (model, revision, topology, pack SHA-256,
geometry fingerprint, ABI version). Code bump w/o pack change
attaches in ms; pack change misses → daemon-side reload. The daemon
is the module library's runtime twin — content-addressing throughout.

READ-ONLY EXPORT: VMM access flags map consumers read-only (the
marketplace tenant-scribble protection for free).

## Production ownership

On every Spark the fleet agent (`tools/fleet_node_agent.sh`, systemd user unit
`fleet-agent`) owns one weightd. `ensure_weightd` starts
`~/sparkdata/weightd/sparkpipe_weightd --socket /tmp/spark_weightd.sock
--mesh-rank R --mesh-rank-mask 0xffff --mesh-interface … --mesh-sgid-index …`.
It passes no `--device-bytes-max`, so the budget is
`SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT` (110 GiB) unless the unit environment
sets `SPARK_WEIGHTD_DEVICE_BYTES_MAX`. It passes no `--mesh-dir`, so records live
in `/tmp/weightd-mesh`. The agent copies its own `mesh-<rank>.rec` to the hub
(`release/qpn/<host>/mesh/`) and pulls the 15 peers' records into the same
directory. The production GLM residentds attach to this socket. weightd and the
residentds run in the agent's cgroup, so restarting the agent restarts them.

These rules follow from that code:

- `ensure_weightd` returns early while any other `*/sparkpipe_weightd`
  executable runs ("unknown owner … refusing automatic startup"). The main loop
  then skips root sync, rendezvous, engine recovery and the API until that
  process exits. A private daemon started by `tools/inference_smoke.py`,
  `tools/weightd_execute_receipt.py`, `tools/multi_dev_orchestrate.py` or a
  `WEIGHTD_MODE=private` family job therefore freezes fleet management on that
  Spark for the whole run. Do not run them on a serving Spark.
- A second weightd-line daemon given mesh flags must use its own `--mesh-dir`
  (or `SPARK_WEIGHTD_MESH_DIR`). In the default directory it overwrites
  `mesh-<rank>.rec` and `.ready`, and the agent publishes the overwritten record
  to all 15 peers (`node/weightd_mesh.c`, 81ddd83). `--mesh-rank`,
  `--mesh-rank-mask`, `--mesh-interface` and `--mesh-sgid-index` must be given
  together or not at all (`node/weightd.c`).
- Daemon death invalidates every consumer (crash semantics below), so a restart
  takes GLM serving down. The agent replaces its weightd only after a new
  binary is announced through the hub's `core/WEIGHTSD_BIN` and every residentd
  on the node has drained (FLEET_RELEASE_RUNBOOK.md §2.4). Never restart it by
  hand.
- The per-node execute-receipt rig (`tools/weightd_execute_receipt.py` with
  `build/weightd_execute_probe`) starts its own daemon, so the first rule
  applies to it. Its history is in
  [archive/WEIGHTD_EXECUTE_STABILITY.md](archive/WEIGHTD_EXECUTE_STABILITY.md).

## Concurrent mesh lane reservations

Weightd IPC ABI 7 extended the existing `LANE_ACQUIRE` request with an exact
`requested_lane`. Each lane owns two mesh bands. There are 16 lanes
(`SPARK_WEIGHTD_MESH_MAX_LANES`, `include/sparkpipe/spark_weightd.h`; raised from
8 in 4f0e339). A coordinator must assign a unique lane from 0 through 15 to each
concurrent job and set `SPARK_WEIGHTD_LANE` to that same value on every
participating rank. Local first-free allocation alone is insufficient: reversed
job startup order on two hosts can otherwise assign one job different bands.
The lane table, including the lane 0 reservation for production GLM, is in
[MULTIDEV_QUICKSTART.md](MULTIDEV_QUICKSTART.md#lanes).

4f0e339 doubled the mesh region (`SPARK_WEIGHTD_MESH_BANDS` is twice the lane
count) but left `SPARK_WEIGHTD_IPC_ABI_VERSION` at 8. A client maps the region
only when its size equals the client's compile-time
`SPARK_WEIGHTD_MESH_REGION_BYTES` (`SparkWeightdMapMeshFd` in
`runtime/spark_weightd.c`), so a main-built client fails with `SCHEMA_ERROR`
against an 8-lane daemon such as the shared-serving-20260922 build. Bump the
ABI whenever the mesh layout changes.

An occupied explicit lane returns `NO_LANE`; it never redirects to another
lane. A second acquire on an owning connection returns `DUPLICATE`, preserving
its original reservation. The client validates that the returned lane matches
its request. GLM logs mode, requested lane, resolved lane, capacity and rank.
The multi-job runner checks every rank's log. An absent environment variable
retains the existing automatic single-job mode and logs it explicitly; an
empty or malformed value fails initialization.

The shared device collective now acquires its reservation through its existing
weightd connection. All single-collective model families inherit this behavior;
logical collective identifiers no longer select mesh bands by their low bits.
GLM lends its existing reservation to its main and HC collectives, with separate
band bindings. Binding rejects an unreserved owner, another daemon generation
or a duplicate band. Common teardown releases bindings only after stream and
registration cleanup; borrowed reservations remain with their caller.

Normal GLM teardown keeps its reservation until collective drain and cleanup
succeed. Closing one idle owner releases only its lane. An unexpected
disconnect of an owner with mesh activity quarantines only that lane
(`orphan_lanes`, `SparkWeightdServerCloseConnection`, since 14d8005) and
releases its activity count; other lanes keep working. The quarantined lane is
reused only by an acquire that carries a topology and passes
`SparkWeightdMeshLaneConfigure`'s quiescence checks in `node/weightd_mesh.c`: no
lane activity, no pending raw mesh RPC, and no pending or failed transfer,
unconsumed doorbell or open wait request on the lane's bands. An acquire
without a topology gets `BUSY`, and an activity request on a quarantined lane
gets `IO_ERROR`. Lane reservation does not partition the shared expert-memory
budget.

`test_weightd_mesh_mock` runs two actual IPC servers with reversed 2-, 3- and
4-job startup order across 24 seeded lane permutations, plus capacity,
occupied/duplicate rejection, neighbor retention and reuse. CUDA and verbs are
host mocks; concurrent real-model inference needs a separate fleet receipt.

## The perf notes (preserved verbatim from the analysis)

- Consumers' kernels read the same physical DRAM pages — zero copies
  after mapping; no IPC-per-access; the pointer IS the weight.
- GB10 unified memory: no PCIe boundary — not the discrete-GPU
  zero-copy trap. Essentially free.
- 2 MB VMM pages (a 25-100 GB arena must not drown the TLB in 4 KB).
- CUDA graphs re-captured after attach (process-local; imported
  addresses stable for process lifetime = the existing prewarm path).
- Cold load unchanged (~20s/100GB NVMe); WARM CODE REDEPLOY < 1s
  attach — the dev-cycle win.
- MODEL UPDATE: background load while old serves, then re-attach —
  needs TRANSIENT 2× the shard footprint. OPERATOR CONSTRAINT: NO
  2× BUDGET → the background-load variant is DEFERRED: updates go
  through stop-attach-start (the fleet is dark-briefly, per the
  registrar's cold wave — seconds). Revisit only with explicit
  budget.
- CRASH SEMANTICS: daemon death invalidates all consumers (detect +
  refuse, fail-closed — never chase stale pointers); consumer death
  drops a refcount.

## Loader fixes regardless (ride the same lane)

L1 parallelize the per-tensor sequential fread+upload (single-thread
today; the daemon's cold path pays it too).
L2 parallel hash (SHA-256 of 25-100 GB single-threaded is a double-
digit-seconds tax).
L3 zero-copy cold: if the pack is already exact runtime layout,
mmap the pack AS the arena backing (no copy-then-fill) — VERIFY the
layout claim per family before adopting.

## Not debug-only (the rationale)

Exact identity + fail-closed = production-safe by construction: an
attach-by-hash consumer loses nothing vs loading the bytes itself;
the determinism receipts stay valid (same bytes). The risk is
lifecycle (refcounts, orphaned arenas, version skew) — gated by the
existing promotion/qualification chain. THE OPERATIONAL WIN: the
per-node multi-topology layout becomes cheap — sixteen topologies'
packs daemon-managed; switching stops being a reload.

## Implementation order (deliberately incremental)

W1 loader fixes L1+L2 (pure win, no daemon needed, measure first).
W2 the daemon core: arena alloc + identity table + export; ONE
  family (dsv4 — its loader is the reference), consumer attach
  path, the crash semantics, 2 MB pages, graph re-capture.
W3 fleet integration: the registrar's GO gains weightd-healthy;
  the wave tools attach instead of load; the qualification gates
  re-run on attached-arena serving (determinism must be identical).
W4 multi-family + the multi-topology operational win.

## Shared mesh topology profiles

Weightd IPC ABI 8 carries the logical-to-physical rank map with the existing
lane reservation. The coordinator assigns one lane in 0–15 and one ordered
`SPARK_TP_MESH_RANKS` list to every rank of a job before launching it. For
example, TP4 on physical hosts 4–7 uses `4,5,6,7`; logical rank 2 must run on
physical host 6. The list must have exactly the collective degree, contain
unique physical ranks in 0–15, and match the daemon's physical rank and
configured participant mask. Omitting the list selects explicit identity;
an invalid explicit list fails. Startup logs the resolved map and physical
peer mask. TP groups with different mappings require separate lane profiles.

Each daemon fixes a lane's topology on its first configured reservation.
Changing its root, membership, order or degree returns `UNSUPPORTED`, even
while idle. Changing profiles requires draining all dependent residents and
starting a fresh daemon. A healthy same-profile restart preserves source
tags and request watermarks and must wait for that lane's activity,
doorbells, hardware gates and transfer completions to drain. Outstanding raw
mesh RPC writes conservatively block all lane reconfiguration. A null
reservation topology is allocation-only and cannot begin GPU mesh activity.

GPU slot indices, peer tails and doorbells remain logical. The existing
transport selects physical QPs from the reserved map, and common host
control broadcasts use physical masks. Chain publication sends the exact
packed map and degree in the existing BASE cell before its ordered key;
peers reject a different topology before adopting that key. Borrowed main
and HC clients must match their owner's full topology. None of these shared
host checks qualifies an individual model's math or GPU serving path; those
still require its numerical and inference gates.

## Expert residency

ATTACH CONTRACT (CONFIGURED LAZY ARENA): when SPARK_WEIGHTD_ATTACH_LAZY
is set the pack MUST attach through KIND_ATTACH_LAZY (VMM reserve, no
read; per-expert demand acquisition and reclaim) and requires a valid
per-pack .experts manifest. A lazy failure is terminal for that attach:
the caller's configured lazy load must never silently degrade to a
whole-pack resident arena. Modules that do not implement the
acquisition protocol must leave the env unset.

### Manifest format (version 2)

`PACK.experts` starts with a 16-byte header (magic, version, range count,
zero; four little-endian u32) followed by 48-byte range records (layer, expert,
kind, zero, offset, bytes, 16-byte ck128 digest), as documented in
`include/sparkpipe/spark_weightd_manifest.h`. The parser groups ranges by
(layer, expert), checks framing, bounds, overlap and unique kinds per expert,
and never infers version 1. Range kinds are producer-defined; producer and
consumer must agree on them, including separate scale ranges. Limits:
`SPARK_WEIGHTD_RANGE_COUNT_MAX` (131072) ranges per manifest,
`SPARK_WEIGHTD_RANGES_PER_EXPERT_MAX` (16) per expert, and
`SPARK_WEIGHTD_EXPERT_BYTES_MAX` (64 MiB) per range
(`runtime/spark_weightd_manifest.c`). The GLM 5.3 Flash TP16 rank pack has
(45 − 3) × 288 = 12,096 routed experts and 12,096 × 4 = 48,384 ranges, two
weights and two scales per expert (arithmetic from
`model-families/glm5_next/include/sparkpipe/spark_glm5_next_model.h`).
`SPARK_WEIGHTD_EXPERT_COUNT_MAX` (40960) is not a manifest cap; only
`tools/weightd_warm.c` uses it, to bound its per-layer EXPERTS argument.

The spine is the sorted exact complement of the expert ranges. It includes
headers and padding and has no digest of its own, so the loader validates the
pack identity before publishing a pointer. Compact spine offsets keep each
source offset's alignment modulo 256. `SparkWeightdSpineLoad` streams the pack
once through a 64 KiB buffer, hashing all of it while copying only spine
bytes. Do not map complement spans at their pack VA: small padding gaps would
pin nearly every expert chunk.

### Pool sizing

Every lazy attach declares `expert_pool_bytes`. The daemon rejects 0,
`UINT64_MAX` and values above its device budget. A pool larger than the pack
becomes one allocation for the whole arena whose chunks are never evicted
(`SparkWeightdPremapPool`, logged as "pool single-alloc"). A pool no larger
than the pack stays per-chunk lazy: acquisitions load chunks on demand and
evict the least recently used unpinned groups to stay inside the pool
(`SparkWeightdAcquireBudget`). A later attach to an existing arena must
declare the same pool or gets `INVALID_ARGUMENT`. The old 8 GiB default was
deleted in 13c1113. Consumers set `SPARK_WEIGHTD_EXPERT_POOL_BYTES` explicitly,
and the GLM module refuses to start without it. The fleet agent passes
34359738368 (32 GiB), which is larger than the GLM rank pack, so production
pools the whole pack.

### Leases and consumer mapping

- ACQUIRE takes at most 512 (layer, expert) keys
  (`SPARK_WEIGHTD_LEASE_GROUPS_MAX`); larger sets need several leases. Each
  arena has 256 leases (`SPARK_WEIGHTD_LEASE_COUNT_MAX`, raised from 64 in
  159a000). Acquisition deduplicates keys, pins the whole set before planning
  the union of physical chunks, counts other owners' pins against capacity,
  evicts only unpinned groups, and verifies every range's ck128 digest before
  copying it. An allocation, read or digest failure releases the new lease and
  frees newly allocated chunks.
- RELEASE checks the owner and the lease identifier. On
  SparkWeightdClientRelease the caller must have established GPU completion and
  unmapped first. Each connection gets an owner identifier that is never
  reused, and explicit detach is `BUSY` while the owner holds leases. When a
  connection closes, the daemon releases that owner's leases in every arena and
  drops its attach references (7e388eb): a dead process cannot signal
  completion, and its GPU work ended with it.
- SparkWeightdManifestIdentity is the canonical identity of a successfully
  loaded, grouped manifest. EXPORT_LEASE returns the sorted physical chunk
  union of one lease, at most 64 descriptors per response
  (`SPARK_WEIGHTD_EXPORT_BATCH_MAX`); batch_offset indexes that union and
  chunk_count stays the arena's total virtual chunk count. Whole-arena EXPORT
  rejects lazy arenas, and the legacy single-range ENSURE returns
  `UNSUPPORTED`.
- The consumer helper `runtime/spark_weightd_map.c` reserves consumer-local VA
  and imports the leased chunks read-only. BeginUse marks a lease in flight;
  Release refuses it until RecordCompletion has recorded an event and that
  event has completed. Overlapping leases share local chunks, and the last
  local owner unmaps before the daemon RELEASE is sent. MapAcquire runs under
  one monotonic deadline; expiry returns `BUSY` and keeps the identifier for
  explicit cleanup. Calls are serialized on the creating CUDA context, and
  callers join every using stream before recording completion.

### Working-set recording

After every successful ACQUIRE the daemon adds the new keys to `PACK.wset`,
eight bytes per key (u32 layer, u32 expert), written to a temporary file and
renamed. At arena creation it loads an existing `PACK.wset`; a key missing from
the manifest or an empty file fails the attach with `SCHEMA_ERROR`. The file is
a trace, not a preload: `build/weightd_warm SOCKET PACK SHA256 REVISION
TOPOLOGY --wset FILE` replays a named set in 512-key acquire/release batches
and prints `WSET-WARM keys=N elapsed_ms=T` (`tools/weightd_warm.c`).

### GLM graph residency today

GLM 5.3 Flash serves through CUDA graphs whose kernel arguments bake expert
pointers at capture. Since 78c2c21 the whole-chain graph runs only when the
resident holds leases on every routed expert of its layers
(`SparkGlm5NextGraphClaimExperts`): 12,096 keys in ⌈12,096 / 512⌉ = 24 leases
(arithmetic). Otherwise it logs "GLM whole-chain graph requires N leased
experts". `SPARK_GLM5_NEXT_PIN_EXPERTS=1` takes those leases at attach. The
fleet runs graph mode with pinning (`G5_GRAPH_PATH=1`, `G5_PIN_EXPERTS=1` in the
agent's `20-serving.conf` drop-in), so the whole pack is resident and pinned on
every rank: the daemon held 20,874 MiB of device memory on spark6 on 2026-09-28
(nvidia-smi), the same figure measured in the PR #1082 campaign
([archive/PARALLEL_RESIDENT_QUALIFICATION.md](archive/PARALLEL_RESIDENT_QUALIFICATION.md)).
Per-wave lazy acquisition runs only on the eager path
(`SPARK_GLM5_NEXT_GRAPH_PATH=0`). This conflicts with the bounded lazy residency
of invariants I28–I30 (`sparkpipe_invariants.md` §6); the redesign is open.

One arena's 256 leases allow at most ⌊256 / 24⌋ = 10 fully pinned GLM residents
per Spark sharing that arena, production included (arithmetic).

### Open: relocatable graphs

8adebc6 proposed capturing each graph once, recording which kernel arguments
are expert pointers, and patching them when experts load, as a dynamic linker
relocates symbols. 360c0ee and 1a674be deleted the route-sweep and union-lease
machinery on that premise, but the patching was never implemented, and
78c2c21 made full pinning the requirement instead. Until graphs are
relocatable, graph serving needs the whole pack resident, and a graph-mode GLM
job cannot run from a partial working set.

### Host tests

`tests/test_weightd_manifest.py` (12,096 groups of four ranges, lookup,
malformed files), `build/test_weightd_lease` (pin accounting, duplicate keys,
stale or wrong-owner release, capacity), `build/test_weightd_working_set`
(real daemon/client IPC with CUDA stubs: shared chunks, pool pressure,
rollback, consumer import, completion-gated release, 65 chunks across two
export responses), `build/test_weightd_expert` (v2 leases, corruption and
drift rejection, ENSURE unsupported) and `build/test_weightd_fd_frames`
(64-descriptor frames and cleanup). These are host and CUDA-stub tests; they
do not qualify GPU residency. Family-specific consumer notes live with the
family, for example
[the glm52 module README](../modules/glm52_resident_decode_stage/README.md).
