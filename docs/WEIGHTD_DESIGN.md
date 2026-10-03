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
directory. The production GLM residentds attach to this socket. The agent starts
weightd as its own transient user unit, `sparkpipe-weightd`, and each root as its
own unit, so restarting the agent does not restart them. Restart weightd with
`systemctl --user restart sparkpipe-weightd` only when no residentd runs.

These rules follow from that code:

- `ensure_weightd` returns early while any other `*/sparkpipe_weightd`
  executable runs ("unknown owner … refusing automatic startup"). The main loop
  then skips root sync, engine recovery and the API until that process exits.
  The mesh record exchange runs in its own loop and continues. A private daemon started by `tools/inference_smoke.py`,
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
  on the node has drained (FLEET_RELEASE_RUNBOOK.md §5.2). Never restart it by
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
lane activity, no pending raw mesh RPC, nothing in flight, no open wait request
and no doorbell that could still ship on the lane's bands. A refusal is `BUSY`
and logs `WD-LANE-BUSY` with the reason once per change. On success the lane's
cells are reset to a fresh daemon's state (`WD-LANE-RESET`, with the count of
failed cells and stale doorbells): doorbell entries, shipped words, wait
entries, transfer state and their bookkeeping are zeroed and the capabilities
re-advertised. The band's `BASE` and `CANCEL` words and the slot payloads and
tails are kept, so rank 0's epoch stays monotonic and a faster peer's first
round is not lost. A failed transfer fences only its own cell while the lane is
configured; it no longer refuses the next acquire. An acquire without a
topology gets `BUSY`, and an activity request on a quarantined lane gets
`IO_ERROR`. Lane reservation does not partition the shared expert-memory
budget.

When a peer re-wires (a new record or a QP that left RTS), weightd first drains
the completion queue, then clears that peer's pending bits in every cell, marks
those cells failed with the peer named, zeroes the peer's send and RPC counters
and advances the peer's wire epoch (`WD-PEER-RESET`). RPC work requests carry
the epoch in their id, so a completion from before the reset never decrements
the new counters, and transfer completions are already dropped by their
generation and bit. A shipped wait on a cell that lost a round to the reset
returns `SPARK_WEIGHTD_MESH_WAIT_ERROR_PEER_RESET` with the peer's physical
rank in the low bits.

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

## Mesh substrate and rendezvous

Each weightd owns one all-to-all RoCE mesh (`node/weightd_mesh.c`). It opens
the device named by `--mesh-interface`; the fleet agent passes `rocep1s0f1`
with `--mesh-sgid-index 3` (`tools/fleet_node_agent.sh`). It creates one RC
send QP and one RC receive QP per peer, 30 QPs for 15 peers, and moves payloads
and tails with plain `IBV_WR_RDMA_WRITE` into the peer's registered receive
region. There are no posted receives and no immediate data. Every topology is a
routing choice over these QPs: lane profiles (next section) map logical ranks
to physical peers. The algorithm is chosen by `tp_device_collective.c`, not
here. With `SPARK_TP_WAIT_MODE=hardware`, the fleet's mode, every collective
runs as chunked direct rounds (`SparkTpLaunchMeshHardware`), with
reduce-scatter plus all-gather over slice routes for large sums; the tree
runs only in spin mode. That the wait mode picks the algorithm is a known
deviation from I36 ([TECHDEBT.md](../TECHDEBT.md#mesh-collectives)).

The region layout is in `include/sparkpipe/spark_weightd.h`. It has 16 lanes
(`SPARK_WEIGHTD_MESH_MAX_LANES`, 0-15) with two bands each (GLM uses one for its
main and one for its HC collective), 16 ranks per band and two slots per rank.
A slot holds 8 rows of 32 KiB plus a 64-byte trailer, 262,208 bytes, and its
last 8 bytes are the tail tag the receiver matches. The slot for band `b`, peer
`p` and ring `r = (tag - 1) mod 2` is `b*32 + p*2 + r`. The slot area is
32 bands x 32 slots x 262,208 bytes = 268,500,992 bytes (arithmetic, about
256 MiB). A 128 KiB area follows it with the doorbells, shipped cells and one
128-byte hardware wait request per rank and band.

Rendezvous goes through files. weightd writes its record `mesh-<rank hex>.rec`
(per-peer send and receive QPNs, rkey, receive address, GID, rank mask, boot
time) into `--mesh-dir` (env `SPARK_WEIGHTD_MESH_DIR`, default
`/tmp/weightd-mesh`) and reads peer records from the same directory. weightd
has no network client for records. On the fleet, the agent's background
`mesh_exchange_loop` publishes the node's own record to the hub's
`release/qpn/<host>/mesh/` (written to a temporary name, then renamed) and
fetches each peer's record from the hub's release HTTP server every second with
a conditional GET. Until every peer in its rank mask is wired, weightd retries on every poll. After that it
rechecks records once per second, rewires a peer whose record boot time
changed, and re-transitions a QP that left RTS (`WD-QP-REPAIR`). When every peer
is wired it writes `<mesh-dir>/.ready`, and the agent starts a multi-rank
residentd only after that file exists. `WD-MESH-STATS` logs the wiring, rewire
and repair counters every 10 seconds, but only while the mesh is ready; an
unready mesh is silent there. Use `sparkpipe_mesh_status` (next section) to see
an unready mesh.

## Mesh readiness surface

**Startup order.** `node/weightd.c` creates the server without a socket
(`SparkWeightdServerCreateUnbound`: CUDA context and worker), then runs mesh
init, then binds the socket (`SparkWeightdServerListen`), then starts the mesh
thread and prints `spark_weightd ready`. Mesh init waits up to 120 s for the
switch GID and up to another 120 s for the pair GID, so for up to 240 s after a
start the socket is missing (a clean predecessor removed it) or refuses
connections (a crashed predecessor left the file). Clients see that as "absent",
not as a hung HELLO. A CUDA or worker failure still happens before the mesh
record is published, so a broken daemon never makes peers rewire to it. The
socket is removed only by the instance that bound it.

**`.ready` v1.** The file is one line:
`weightd-ready v1 pid=<pid> boot_ns=<boot_ns> rank=<rank> rank_mask=0x<mask>`.
`boot_ns` is the mesh boot identity, the same value the record carries and peers
wire against. weightd writes `.ready.tmp` and renames it over `.ready` when the
mesh becomes ready, removes `.ready` while the mesh is unready, rewrites it
within a second if it is missing or was written by another process, and removes
its own file on a clean stop (`WD-MESH-STOP rank=.. ready_marker_removed=1`).
All marker I/O runs on the main thread outside the mesh lock and without fsync.
A SIGKILL or crash leaves the file behind until the next start removes it, so a
reader must check the pid (against the unit's MainPID) or ask the daemon.

**MESH_STATUS (kinds 41/42).** An additive kind under ABI 9; an ABI 8
connection, or a request before HELLO, is closed. It is answered inline on the
server thread, so a probe never waits behind a cold attach. The request carries
`layout` (1) and a reserved word; layout 0 or a non-zero reserved word gets
`status = INVALID_ARGUMENT` and the connection stays usable. A request for a
newer layout is answered in the daemon's layout. The 4096-byte result
(`SparkWeightdIpcMeshStatusResult`) holds the daemon generation (HELLO's),
pid, rank, rank mask, `mesh_state` (disabled, wiring, ready), the mesh boot
identity, a change counter `mesh_generation`, `ready_since`, the twelve
`WD-MESH-STATS` counters, one entry per physical rank and one per lane:

- peer: state (absent, self, no_record, record_rejected, record_invalid,
  wire_failed, wired), the record status behind it, the record and wired boot
  identities, queue-pair RTS bits, send and RPC work in flight, the time of the
  last good and bad completion and the run of completion errors since the last
  good one;
- lane: topology (`packed_ranks` nibble i = physical rank of logical rank i),
  configured / owned / quarantined flags, activity, failed and pending cells,
  and `configure_status` with `busy_reason` and `busy_index`. These come from
  the same check `SparkWeightdMeshLaneConfigure` uses (`LaneCheckLocked`), so
  the report cannot drift from the real gate. Owned and quarantined are server
  facts published when the worker is idle; a cold attach can keep a dead
  owner's lane owned until it finishes.

`mesh_generation` counts peer state changes, readiness transitions, a wired
peer's new boot identity, failed cells and lane configuration. It does not
count activity toggles, which happen once per chain. It is a change counter, not
an identity: to see whether a peer restarted, compare its `wired_boot_ns`.
`WIRED` means the local queue pairs were transitioned to the peer's record; it
is not proof the peer is alive. A powered-off peer whose stale record is still
in the mesh directory is wired. The only liveness evidence is traffic:
`cq_err_since_ok` and `last_err > last_ok`.

Layout rules: bytes [0,80) keep their meaning forever; an additive layout may
only give meaning to `reserved0` and `reserved_tail` and add enum values
(readers print unknown values as `unknown(N)` and treat them as not ready), and
keeps `layout_compat = 1`; any other change raises `layout_compat`; a different
size needs a new kind. `status` carries only OK, INVALID_ARGUMENT or
UNSUPPORTED: readiness is in `mesh_state`. A runtime linked without the mesh
answers UNSUPPORTED; a daemon started without mesh flags answers OK with
`mesh_state = disabled`.

**Client.** `SparkWeightdClientConnect` now bounds the connect by the client
timeout (10 s by default): the socket is non-blocking during connect, a full
listen backlog (Linux `EAGAIN`) is retried until the deadline and then reported
as BUSY. `SparkWeightdClientConnectWithin` takes the caller's timeout.
`SparkWeightdMeshStatusQuery` opens, asks and closes, and classifies the
outcome: answered, absent (`ENOENT`, `ECONNREFUSED`, or the daemon restarted
between two connects: `WD-STATUS-RESTARTED`), unresponsive (connect, HELLO or
MESH_STATUS timed out), unserved (a pre-readiness daemon closed the connection
on kind 41 and a reconnect found the same daemon generation:
`WD-STATUS-UNSERVED`), incompatible (`layout_compat` above the client's) or
fault.

**Tool.** `sparkpipe_mesh_status --socket PATH --timeout SECONDS
[--wait-lane-peers MASK [--lane N]]`. Without `--wait-lane-peers` it prints one
JSON line. With it, it polls every 250 ms until `status` is OK, the mesh is
ready and every rank in MASK is wired; with `--lane`, the lane must also be
neither owned nor quarantined and its configure gate open. A daemon restart
during the wait prints `MESH-STATUS-DAEMON-RESTART` and the wait continues.
Exit codes: 0 met (or answered, in status mode); 1 fault; 2 usage (including a
MASK outside the daemon's rank mask); 3 timed out with the daemon answering
(stderr starts with `MESH-STATUS-TIMEOUT state=.. missing=0x..` followed by one
`MESH-STATUS-PEER` line per missing rank and, with `--lane`, a
`MESH-STATUS-LANE` line); 4 mixed version (unserved or incompatible, no retry);
5 the lane is configured for another rank set; 6 daemon absent at the deadline;
7 mesh disabled or not built; 8 daemon unresponsive at the deadline. The tool
requires `ready` because the data path still gates on the global mesh state.
`tools/fleet_release/weightd.sh publish` publishes it to `core/bin` with
weightd. Nodes sync `core/bin` every loop but
install a new weightd only after the announce, so the tool may run ahead of the
daemon; against an older daemon it exits 4. The fleet agent does not use it
yet: switching the agent's gate from `.ready` to the tool is a second release,
after every node runs this weightd.

## ABI 9: per-peer routes, served ABI range, row cap

ABI 9 is one bump that adds capacity without moving anything an ABI 8 engine
depends on.

**Served range.** The daemon accepts connections that speak ABI 8 or ABI 9
(`SPARK_WEIGHTD_IPC_ABI_VERSION_SERVED_MIN` to `SPARK_WEIGHTD_IPC_ABI_VERSION`).
- A connection's first frame fixes its version. A later frame with another
  version closes the connection.
- Every reply carries the connection's version, so an ABI 8 client validates
  it exactly as before.
- Kinds added in ABI 9 (`MESH_STAGING_MAP`, 39/40) close an ABI 8 connection.
- ABI 7 and ABI 10 frames close the connection.
- An ABI 9 client against an ABI 8 daemon (921360ca) fails at connect: the old
  daemon closes the socket, and the client reports `status=4`. It never falls
  back.

**Arena identity.** `SparkWeightdIdentityPrepare` rewrites any served client
ABI to the daemon ABI. An ABI 9 engine therefore attaches, or read-only
shares, the arena an ABI 8 engine loaded, and the reverse also holds. An
engine upgrade reuses the warm arena instead of loading a second copy. Leases,
manifests, receipts and the pack bytes are the same under both ABIs.

**Unchanged from ABI 8:**
- the mesh region (`SPARK_WEIGHTD_MESH_REGION_BYTES`, 268,632,064 bytes);
- the slot geometry, doorbells, shipped cells and wait requests;
- the rendezvous record;
- the FULL, SCATTER and GATHER routes.

Daemons of both ABIs therefore wire to each other during a per-node roll.

**PEER route (mode 3).**
- Staging:
  - Each daemon owns a staging area of `SPARK_WEIGHTD_MESH_STAGING_BYTES` =
    32 bands x 16 peers x 256 KiB = 128 MiB.
  - It is a second memfd, registered as a second MR with local access only.
  - The slot for band `b` and logical peer `p` is at
    `SPARK_WEIGHTD_MESH_STAGING_OFFSET(b,p)`.
- Doorbell:
  - The route's `slice_bytes` is the per-peer length: 8-byte aligned, at most
    256 KiB.
  - The doorbell's `bytes` must equal it (`SparkWeightdMeshRouteFits`).
- Posting: for each peer in the mask, weightd writes `staging[b][p]` to offset
  0 of this rank's receive slot at the peer. The tail tag still ships from the
  sender's own slot.
- Capacity: each peer receives a full 256 KiB per round, where the SCATTER
  route gives it a 16 KiB slice at TP16.
- One parity is enough. A driver publishes the next round on a band only after
  that band's shipped cell reports the previous round, so weightd has finished
  reading the staging slot before the driver rewrites it.
- Discovery:
  - The lane's wait cells advertise `SPARK_WEIGHTD_MESH_CAPABILITY_PEER_ROUTES`
    (bit 1, next to SLICE_ROUTES).
  - `SparkWeightdClientMeshStagingMap` returns the staging fd. The client
    checks the capability and the geometry, and maps exactly the staging
    bytes.
  - A driver that needs PEER routes must refuse to start when the capability
    is absent. It must not fall back to SCATTER.

**Row cap.** `SPARK_WEIGHTD_MESH_MAX_BATCH_ROWS` is 1,024 (was 128). The cap
is compile-time in the engine, so it applies only to engines built against
ABI 9. A deployment still chooses its own `execution_row_capacity`.

**Drivers on PEER routes.** When the lane advertises PEER_ROUTES, the
hardware-wait device collective maps the staging export once, through the
lane owner's client. Two operations then use PEER:

- **kv_shard all-to-all.** Each peer receives up to 131,072 BF16 values per
  round. The rank's own share goes straight to its own slot.
- **Large BF16 sums: reduce-scatter + all-gather.**
  - One chunk carries degree x 131,072 values: 2,097,152 at TP16, against
    131,096 for the slot chunk.
  - Reduce-scatter: every owner's slice leaves from its staging slot, and the
    owner sums the degree contributions from offset 0 of each peer's slot. The
    per-element order (peers 0..degree-1, fp32 accumulation, the same BF16
    truncation) is the same as the slot path, so the result bits are the same.
  - All-gather: a FULL route of the owner's reduced slice from offset 0 of its
    own slot.
- **Failure.** If the lane advertises PEER_ROUTES but the staging export fails,
  preparation stops. It does not fall back.
- **Direct rounds.** Single-row direct rounds (below 49,152 values or below
  degree 4) keep their slot path.

**Round counts (TP16, arithmetic).**

kv_shard partial all-to-all, per DSA layer, one direction. Payload is 4,112
bf16 values per row per peer.

| rows | SCATTER (16 KiB per peer per round) | PEER (256 KiB per peer per round) |
|---:|---:|---:|
| 1 | 1 | 1 |
| 8 | 5 | 1 |
| 128 | 65 | 5 |
| 129 | 65 | 5 |
| 512 | 257 | 17 |
| 1,024 | 514 | 33 |

Hidden all-reduce at 4,096 wide (hardware wait mode, reduce-scatter +
all-gather, 2 rounds per chunk):

| rows | values | slot chunks (131,096) | rounds | PEER chunks (2,097,152) | rounds |
|---:|---:|---:|---:|---:|---:|
| 16 | 65,536 | 1 | 2 | 1 | 2 |
| 128 | 524,288 | 4 | 8 | 1 | 2 |
| 256 | 1,048,576 | 8 | 16 | 1 | 2 |
| 1,024 | 4,194,304 | 32 | 64 | 2 | 4 |

**Memory.** +128 MiB of pinned host memory per node, for the staging area.

**Host tests:**
- `test_weightd_mesh_mock` checks PEER routes at 1, 8, 128, 129, 512 and
  1,024 rows: the source slot per logical peer, the staging lkey, the landing
  offset, tails, shipped release and partial masks.
- `test_weightd` checks the served range, the version echo, closing on mixed
  versions, and identity canonicalization.
- `weightd_peer_route_probe` drives PEER and SCATTER exchanges between two
  real daemons and checks every byte that lands.

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
`SPARK_WEIGHTD_RANGE_COUNT_MAX` (262144) ranges per manifest,
`SPARK_WEIGHTD_RANGES_PER_EXPERT_MAX` (16) per expert, and
`SPARK_WEIGHTD_EXPERT_BYTES_MAX` (64 MiB) per range
(`runtime/spark_weightd_manifest.c`).

#### Range count bound

The range bound is derived from the host memory the manifest tables cost,
not picked to fit one model. Each range costs at most 120 bytes of host
tables in the daemon: the parsed range (48), its group slot (16, groups are
allocated at the range count), its spine span (24, the spine is allocated at
range count + 1) and at most 32 bytes of per-group daemon state (the expert
entry, 24, plus the lease pin, 4;
`SPARK_WEIGHTD_MANIFEST_GROUP_STATE_BYTES_MAX`).
`SPARK_WEIGHTD_MANIFEST_TABLE_BYTES_MAX` gives one manifest 32 MiB of host
tables. `SPARK_WEIGHTD_RANGE_COUNT_MAX` is the largest power of two that fits:
262144 × 120 B = 30 MiB, while 524288 would need 60 MiB. With at most
`SPARK_WEIGHTD_ARENA_COUNT_MAX` (16) arenas, the daemon's worst case is
16 × 32 MiB = 512 MiB (`SPARK_WEIGHTD_MANIFEST_DAEMON_BYTES_MAX`), 2.5% of
the 20 GiB per-node MemAvailable floor. Static assertions in
`include/sparkpipe/spark_weightd_manifest.h` and `runtime/spark_weightd.c`
fail the build if a struct grows or a constant is raised without the budget.
The on-disk manifest at the bound is 16 + 262144 × 48 B = 12 MiB.

A manifest above the bound is refused before any table is allocated. The
loader prints the path, the range count, the bound and the budget to stderr
and returns `SPARK_STATUS_CAPACITY_EXCEEDED`; nothing is clamped or truncated.
The bound covers the known large ranks: Kimi K3 TP16 (164,864 ranges, 63% of
the bound) and DSV4-Pro TP16 (147,456 ranges, 56%), both of which the old
131072 cap refused. `tests/test_weightd_manifest.py` loads manifests of both
shapes and one of exactly 262144 ranges, and checks that 262145 is refused
with the message. The GLM 5.3 Flash TP16 rank pack has
(45 − 3) × 288 = 12,096 routed experts and 12,096 × 4 = 48,384 ranges, two
weights and two scales per expert (arithmetic from
`model-families/glm5_next/include/sparkpipe/spark_glm5_next_model.h`).
`SPARK_WEIGHTD_EXPERT_COUNT_MAX` (40960) is not a manifest cap; only
`tools/weightd_warm.c` uses it, to bound its per-layer EXPERTS argument.

The spine is the sorted exact complement of the expert ranges. It includes
headers and padding and has no digest of its own, so the loader validates the
pack identity before publishing a pointer. Compact spine offsets keep each
source offset's alignment modulo 256. `SparkWeightdSpineLoad`
(`runtime/spark_weightd_spine.c`) copies only the spine spans when the pack
has a valid verify-once receipt (below). Without one it streams the whole
pack through a 1 MiB buffer, SHA-256 hashing all of it while copying only
spine bytes, and records the receipt. With a per-chunk (non-pooled) map, do
not map complement spans at their pack VA: small padding gaps would pin
nearly every expert chunk. A pooled map already covers the whole pack, so
there the spine is read in place (below).

### Verify once

A pack is hashed once, not once per load. After a full verification succeeds,
the verifier writes `<pack>.verified` next to the real pack (symlinks are
resolved). If the pack directory is not writable, the receipt goes to
`$XDG_STATE_HOME/sparkpipe/verified/<dev>-<inode>.verified`, or
`~/.local/state/sparkpipe/verified/` when `XDG_STATE_HOME` is unset
(`runtime/spark_weightd_receipt.c`).

The receipt is a short text file: device, inode, size, mtime_ns, ctime_ns,
the verified SHA-256 and/or ck128, the verifier (role, host and pid), the
wall-clock verification time, and a SHA-256 seal over those lines. It is
written to a temporary file, fsynced, and renamed over the old one, so
readers see either the old or the new receipt, never a partial one.

A receipt is trusted only if all of the following hold:

- it parses exactly: re-rendering the parsed fields gives the same bytes and
  the seal matches;
- it is a regular file owned by the loading user or root, and not group- or
  world-writable;
- every stat field matches `fstat` of the open pack descriptor;
- its digest equals the digest the loader expects: the identity SHA-256, or
  the `<pack>.ck128` sidecar value in ck128 mode.

If any check fails, the loader logs the reason (`absent`, `stale`,
`unreadable or malformed`, `untrusted owner or mode`, `digest differs`) and
runs the full verification. A bad receipt never counts as a pass, and a hash
mismatch fails the attach with `HASH_MISMATCH`.

Any change to the pack changes its ctime and so invalidates the receipt: a
restamp (even to the same mtime), chmod, in-place write, copy (new inode), or
repack-and-rename. File timestamps are coarse. A write in the same timestamp
tick as the verification would therefore keep the old stat, so a receipt is
written only if the pack's mtime and ctime are at least
`SPARK_WEIGHTD_RECEIPT_SETTLE_NS` (1 s) older than the wall clock. A
just-written pack is verified on every load until it has settled; the loader
logs `receipt not written status=busy`.

Trust boundary. The seal is an unkeyed checksum: it catches a torn or
hand-edited receipt, not a forged one. Anyone who can create files as the
loading user or root can mint a receipt, and could equally rewrite the pack,
so receipts add no trust beyond that uid. A receipt vouches for the file's
metadata, not its bytes. Changes that bypass the filesystem's ctime are not
seen: media corruption after verification, raw block-device or debugfs
writes, or root moving the clock and restamping. Run
`weightd_receipt verify <pack>` to re-hash a pack when its bytes are in
doubt. A writer that already holds a shared writable mapping is covered by
the `O_DIRECT` verification pass: it writes dirty pages back and
write-protects them, so the writer's next store updates ctime. The receipt
also binds `st_dev`. On a node where device numbers can change across a
reboot (several NVMe drives probed in varying order, or device-mapper), the
first load after the reboot re-verifies once.

Every cold path reads the receipt:

- The eager attach (`SparkWeightdServerAttachCold`) streams the pack into the
  arena through `SparkWeightdPackStream`: a reader thread doing `O_DIRECT`
  reads (buffered where the filesystem refuses `O_DIRECT`, such as tmpfs)
  into three 32 MiB aligned buffers, overlapped with the device copy. With a
  valid receipt nothing is hashed. Without one, the same pass also hashes.
- The lazy attach (`SparkWeightdServerAttachLazy`) checks the receipt, or
  runs a hash-only pass once, before it preloads the spine. The preload
  reads the same descriptor, and the stat must be unchanged after it, so
  the spine that consumers read in the arena comes from verified bytes.
- With the pool mapped, the client keeps no spine copy of its own.
  `SparkWeightdLazyPackSlice` returns `map base + file offset`, a pointer
  into the arena that every process of the same pack shares read-only.
  The mapping is private to no process, so a second instance of a model on
  the node adds no spine bytes (6.91 GiB per GLM-5.3 Full TP16 rank). Only
  when the daemon stages no pool handle does the client allocate the
  compacted spine and fill it through `SparkWeightdSpineLoad`.
- The client spine load (`SparkWeightdSpineLoad`) records a receipt with
  both SHA-256 and ck128.

The per-boot `/tmp/spark-weightd-spine` receipts are retired.

Each verification logs `weightd pack-verify path=... mode=receipt|sha256|ck128
bytes= seconds= gbps= io=direct|buffered read_wait_s= work_s=`. `work_s` is
the time spent hashing and copying, and `read_wait_s` is the time spent
waiting on the disk.

`build/weightd_receipt` (`tools/weightd_receipt.c`) operates on receipts
outside the daemon:

- `verify <pack> [sha256]` runs a full SHA-256 and ck128 verification and
  writes the receipt. Without a digest argument it reads `<pack>.sha256`.
- `check` reports receipt state.
- `show` prints the fields.
- `read` and `hash` measure raw read and hash throughput.
- `adopt <pack> [dir]` converts a legacy `/tmp/spark-weightd-spine` receipt
  without rehashing, but only if it is owned by the user, its size, mtime,
  ctime and inode match the pack, its file name binds the recorded digest,
  and it agrees with `<pack>.sha256` when that sidecar exists.

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

### Pack-scoped reclaim

`weightd_warm SOCKET --reclaim` (IPC `RECLAIM`) frees every cold arena on the
node: refcount 0 and no leases, whichever lane loaded it. On 2026-09-28 the
glmfull and K3 teardowns each freed the qwen lane's staged lane-3 arenas on
spark0/1/2/5 this way (lanes/glmfull-w7.md, lanes/k3-w7.md).

`RECLAIM_PACK` (IPC kinds 35/36) takes one lowercase pack SHA-256 and frees
only the cold arenas whose identity carries it. Every identity with that SHA
matches (model, revision and topology are not compared), because the bytes are
the lane's own pack. Matching arenas that are still attached or leased are
counted as `busy_arena_count` and left alone; the cold-arena rule does not
change. Both reclaims log one line with their scope and counts
(`weightd reclaim scope=<sha>|all-cold freed_arenas=… busy=…`). A malformed SHA
returns `INVALID_ARGUMENT` and frees nothing.

`weightd_warm SOCKET --reclaim-pack PACK|SHA256 [...]` resolves each PACK
through its `PACK.sha256` sidecar before it connects and never hashes the pack.
It exits 0 when everything matching was cold, 3 when a matching arena is still
busy, 2 on an unresolvable argument and 1 on a daemon error. A weightd older
than this change closes the connection on the unknown kind. The tool then
reports that and exits 1; it never falls back to the node-global reclaim.

A lane stops its units first and then runs `--reclaim-pack` over its own
`packs/*.pack`. A lane that shares another lane's arena (below) must not name
that pack. A cold shared arena would be freed, and the owner's next start
would pay a full cold load.

### Read-only arena sharing

Arenas are keyed by the whole identity (model, revision, topology, geometry,
pack SHA-256, arena bytes), so two lazy attaches with equal identities, equal
manifests and equal `expert_pool_bytes` already map one arena. What a
development root of the production model lacked:

1. Nothing stopped a dev attach whose identity differed by one field (for
   example the `-ws` revision) from loading a private copy of the rank pack.
   The copy is the whole pack in pooled mode, 20.9 GiB per GLM Flash rank
   (above).
2. The pooled pool is imported read-write into every consumer
   (`WD-MAP-POOL-BULK`), so a dev kernel writing out of bounds could corrupt
   the weights production is serving from.
3. A pool mismatch failed with a bare `INVALID_ARGUMENT` (lanes/jitkv.md).

Opt-in per lane: `SPARK_WEIGHTD_SHARE=readonly` in the residentd environment.
Unset or empty keeps today's private attach. Any other value is refused
(`INVALID_ARGUMENT` and a stderr line); nothing guesses.

- `SparkWeightdLazyPackCreate` sends `ATTACH_LAZY_SHARED` (kinds 37/38, same
  body as `ATTACH_LAZY`). weightd attaches only to a resident arena with an
  identical identity, the same manifest ranges and the same pool. It never
  creates an arena for a shared attach: without a match it returns `NOT_FOUND`
  and logs `weightd shared attach refused: no resident arena for model=…
  revision=… pack_sha256=…`. A pool mismatch returns `INVALID_ARGUMENT` and
  logs both sizes, for private and shared attaches alike.
- The consumer maps the pooled pool `CU_MEM_ACCESS_FLAGS_PROT_READ`
  (`SparkWeightdMapCreateAccess`, logged `access=read-only`). Per-chunk
  (non-pooled) imports were already read-only for every consumer. Spine
  slices point into the pool and are only read.
- `SparkWeightdAttachPack`, the whole-arena resident path, refuses the variable
  with `UNSUPPORTED` / reason `share_unsupported` instead of ignoring it.
- The ABI version stays 8. Old clients never send kinds 35-38. A lane that
  sets the variable against an old weightd fails at attach, because the old
  daemon closes the connection on the unknown kind.

What a sharer must match: the owner's `model_revision`, topology and pack (a
symlink to the production rank pack with the same `.sha256` and `.experts` is
enough, since the identity carries the SHA and not the path), and the owner's
`SPARK_WEIGHTD_EXPERT_POOL_BYTES` (34359738368 in production).

Memory: the sharer adds only its residentd (KV, workspaces, CUDA context; no
spine copy on a pooled arena). A GLM
Flash dev root is about 6-14 GiB per node (lanes/glmproofs-w6.md), not another
20.9 GiB arena.

What sharing does not isolate:

- Leases. Each fully pinned GLM resident takes 24 of the arena's 256 leases,
  so at most 10 pinned residents fit per arena, production included (above).
- Eviction. A per-chunk (non-pooled) sharer's misses can evict the owner's
  unpinned groups. Pooled arenas never evict, which is the production case.
- GPU time. The perf-window rule is unchanged; sharing saves memory, not SMs.
- Lifecycle. A sharer's reference keeps the arena. A production restart with
  the same identity attaches warm to it, which is fine. A production release
  that changes the pack needs the old arena gone first, so every sharer stops
  before such a release, or the NO-2x budget refuses the new arena.

Evidence: `build/test_weightd_working_set` `check_shared_attach` (CUDA stub).
It checks refusal before the owner exists, the pool-mismatch refusal, one
generation and refcount 2 once shared, a read-only consumer mapping that
refuses a write probe while the owner's mapping accepts it, the lazy pack
path with the variable, refusal of a bad value, and pack-scoped reclaim
afterwards. On sparkf GB10 (2026-09-29), a private weightd on its own socket
and latch port with a 128 MiB pooled pack showed: shared attach refused
before the owner; owner mapping `cuMemGetAccess` = 3 (read-write); shared
mapping = 1 (read-only); all 8 leased experts read back byte-exact through
both mappings; `--reclaim-pack` freed exactly that arena. A deliberate write
through the read-only mapping was not run on a node serving production,
because it would raise a GPU fault on that node.

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
