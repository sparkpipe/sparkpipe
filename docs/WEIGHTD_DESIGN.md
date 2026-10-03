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

### Record directory

`sparkpipe_weightd` and `sparkpipe_weightsd` are two binaries built from the
same sources (`node/weightd.c`, `node/weightd_mesh.c`). Both can run on one
host, each with its own socket and latch port (`SPARK_WEIGHTD_LATCH_PORT`),
and each deployment also needs its own record directory. What a shared
directory breaks on the fleet is under "Production ownership".

- `--mesh-dir` takes precedence over `SPARK_WEIGHTD_MESH_DIR`. An empty
  environment value is ignored; an empty `--mesh-dir ''` is a usage error
  (exit 2). `SPARK_WEIGHTD_MESH_DEFAULT_DIR` applies only when neither is
  given. `SparkWeightdMeshInit` sets the directory, and nothing changes it
  afterwards.
- A daemon started without the mesh flags runs no mesh and writes no
  records.
- weightd creates the directory (mode 0755, not its parents) when it is
  missing. It writes its record to `mesh-<rank hex>.rec.tmp`, fsyncs it and
  renames it into place, so a reader never sees a partial record.

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
`tools/publish_core.sh weightd` and `tools/fleet_release/weightd.sh publish`
publish it to `core/bin` with weightd. Nodes sync `core/bin` every loop but
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

### Manifest producers

Each model family writes `<pack>.experts` with its own tool. Without a valid
one weightd logs `weightd lazy attach rejected: <pack>.experts status=...`
and fails the attach, private or shared. The DSV4 Pro and Qwen 3.8 27B tools
set the range kind to `tensor kind * 2 + plane`, plane 0 for the payload and
1 for the scales.

`tools/dsv4_pro_experts_manifest.c`, usage
`dsv4_pro_experts_manifest <pack> [<out>]` (default `<pack>.experts`):

- In a DSV4 Pro TP rank pack every expert is itself tensor-parallel: each
  expert's W1 and W3 hold its `EXPERT_WIDTH / TP` rows and W2 holds its
  column shard (`tools/dsv4_tp16_stagepack.py`). One directory entry per
  tensor kind and layer holds all experts back to back, so the tool cuts each
  FP4 expert entry (tensor kinds 19-21, range kinds 38-43) into equal
  per-expert spans per plane: `rows * columns / 2` payload bytes and
  `rows * columns / 32` scale bytes, divided by the header's expert count
  (384 for DSV4 Pro).
- It includes the three draft (MTP) layers at layer markers `0xFFFFFFFB` to
  `0xFFFFFFFD`, which the TP packer replicates full-width on every rank. Any
  other layer marker of 64 or above fails.
- It requires expert entries on exactly the backbone layers of the stage,
  3 x (stage layer count) backbone expert entries in total, exactly nine
  draft entries and at least one range.
- It truncates the output on open and deletes it on a later failure, so a
  run that fails after opening the output leaves no manifest at that path,
  even one that existed before. It does not
  load the result with `SparkWeightdManifestLoad`.

`tools/qwen38_27b_experts_manifest.c`, usage
`qwen38_27b_experts_manifest <pack.q38sp|pack.qwen38_27bsp>`, output
`<pack>.experts`:

- The 27B is dense, so its expert tier is the per-layer FFN. Each pack layer
  becomes one group `(layer, expert 0)` holding the payload and scale planes
  of gate, up and down (tensor kinds 5-7, range kinds 10-15), the range-kind
  rule of `tools/qwen38max_experts_manifest.c`.
- The global layer and the MTP layer are skipped, so the MTP layer's FFN
  stays in the spine with every non-FFN tensor.
- Every pack layer must carry all three FFN tensors.
- It writes `<pack>.experts.partial.XXXXXX`, fsyncs it, loads it with
  `SparkWeightdManifestLoad`, and publishes it with `link()`. `link()` fails
  when `<pack>.experts` exists, so the tool never replaces a manifest
  (error -27): remove the old file to regenerate. On any failure an existing
  output is left untouched.

`tools/hy4_experts_manifest.c`, usage `hy4_experts_manifest <pack>`, writes
`<pack>.experts`, replacing any file there, in the version-1 layout: a
16-byte header and 40-byte records (reserved, chunk ordinal, offset, bytes,
ck128) covering the whole pack in 64 MiB chunks. `SparkWeightdManifestLoad`
accepts only version 2, so weightd rejects a lazy attach that finds this
file. The version-2 producer for hy4 is `tools/hy4_experts_manifest_v2.c`,
which reads a safetensors rank pack.

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

## Client API contracts

What each weightd header guarantees and what its caller must guarantee. The
lease lifecycle itself is under "Leases and consumer mapping".

### `spark_weightd_attach.h`

`SparkWeightdAttachRequested` reads `SPARK_WEIGHTD_SOCKET` (an empty value
counts as unset) and `SPARK_WEIGHTD_ATTACH`:

| `SPARK_WEIGHTD_SOCKET` | `SPARK_WEIGHTD_ATTACH` | result |
|---|---|---|
| any | set, not `0` or `1` (empty included) | `INVALID_ARGUMENT` |
| set | unset or `1` | `OK` |
| set | `0` | `INVALID_ARGUMENT`: a configured socket cannot be switched off |
| unset | `1` | `INVALID_ARGUMENT` |
| unset | unset or `0` | `BUSY` |

The last row is a known violation, not a loading mode. `BUSY` here reports a
missing configuration with no operation outstanding (I17), and callers that
read it as "attach not requested" copy the whole stage pack to the device
themselves, an opt-out from mandatory attach (I03,
[sparkpipe_invariants.md](../sparkpipe_invariants.md)). The violation is
recorded in two open entries under
[TECHDEBT.md](../TECHDEBT.md#model-residency-and-storage): the
`SparkWeightdAttachRequested` entry and the entry for a deployment that omits
the `weightd` member.

`SparkWeightdAttachPack`:

- Failure reasons before the attach request, in check order: `attach_config`
  (`INVALID_ARGUMENT` from the table above), `env_off` (`BUSY`, the same
  violation, for `SPARK_WEIGHTD_ATTACH=0` without a socket), `no_socket`
  (`INVALID_ARGUMENT`), `share_unsupported` (`UNSUPPORTED`, for any non-empty
  `SPARK_WEIGHTD_SHARE`), `no_identity` and `identity` (`INVALID_ARGUMENT`;
  the second when `SparkWeightdIdentityPrepare` refuses the identity), then
  `no_daemon` when the connect fails.
- `SPARK_WEIGHTD_IDENTITY_MODEL` and `SPARK_WEIGHTD_IDENTITY_REVISION`
  override the slice's model and revision. The slice's `pack_sha256` takes
  precedence over `SPARK_WEIGHTD_PACK_SHA256`.
- With `SPARK_WEIGHTD_ATTACH_LAZY` set it sends a lazy attach with
  `expert_pool_bytes` = pack bytes + 2 MiB and maps nothing. Any failure on
  that path returns `INVALID_ARGUMENT`; the reason is `no_daemon` for a failed
  connect, otherwise the name of the failed exchange status or of the
  daemon's result status. `SparkWeightdAttachImportMap` uses whole-arena
  `EXPORT`, which refuses lazy arenas ("Leases and consumer mapping"), so it
  cannot map such an outcome.

`SparkWeightdAttachMappedPack` attaches, requires
`arena_bytes == slice->pack_bytes` (`ABI_MISMATCH`, reason `arena_mismatch`),
then calls `SparkWeightdAttachImportMap`. On success the outcome owns an open
client and a mapped range at `map_base`. Every failure after the connect
calls `SparkWeightdAttachRelease`, which closes the client and zeroes the
outcome, so the caller has nothing to clean up. Handles imported before the
virtual range is reserved are not released on failure:
`SparkWeightdAttachRelease` and `SparkWeightdAttachMapUndo` release handles
only when `map_base` is set. `SparkWeightdAttachImportMap` maps the whole
arena with `CU_MEM_ACCESS_FLAGS_PROT_READWRITE`; read-only consumer mappings
exist only on the lazy path (per-chunk imports and
`SPARK_WEIGHTD_SHARE=readonly`).

### `spark_weightd_lazy_pack.h`

`SparkWeightdLazyPackCreate` and `SparkWeightdLazyPackCreateChecked` are
startup calls. They connect, attach lazily (or shared, see "Read-only arena
sharing"), parse `<pack>.experts`, create the consumer map, place the spine
and start a worker thread. A missing or invalid manifest fails the call like
any other failure.

Every input is explicit:

- `expert_pool_bytes` of 0 is `INVALID_ARGUMENT` before any connect.
- The identity must pass `SparkWeightdIdentityPrepare`.
- A missing pack is `NOT_FOUND`. The pack must be a regular file whose size
  equals `identity.arena_bytes`, otherwise `IO_ERROR`. A relative `pack_path`
  is resolved with `realpath` before it is sent to the daemon.
- `spine_budget` bounds the spine. `spine_allocation_bytes + 255` above it
  is `CAPACITY_EXCEEDED`, checked even when the spine will be read from the
  pool.

Failure ownership. On failure the call destroys the partial pack. If that
cleanup itself does not return `OK`, `*out` is the non-null partial pack and
the status is the original error. That pack is not ready, so
`SparkWeightdLazyPackSlice` refuses it. Its only valid use is
`SparkWeightdLazyPackDestroy`, repeated until it returns `OK`.

The `SparkWeightdManifestCheck` callback of
`SparkWeightdLazyPackCreateChecked` runs after the daemon attach and the
manifest parse, and before the consumer map and any spine allocation on the
GPU. A model can therefore reject a manifest shape it cannot serve before
the consumer spends device memory. The daemon attach, including the daemon's
pack verification for a new arena, has already happened at that point. After
the check passes, the manifest is parsed again.

When the attach loaded the arena from the pack (`loaded_from_pack`), the
client compares `SparkWeightdManifestIdentity` of its own parse with the
daemon's `manifest_sha256` and fails with `HASH_MISMATCH`
(`LAZY-MANIFEST-MISMATCH`). On a warm attach the daemon compares the ranges
with the resident arena's instead and returns `HASH_MISMATCH` when they
differ.

`SparkWeightdLazyPackDestroy`:

- The caller first stops submitting work and drains every GPU reader of the
  spine.
- It runs on the creating CUDA context: the map refuses another context, and
  a private spine is freed with `cudaFree`.
- It clears `ready` before anything else, so `SparkWeightdLazyPackSlice`
  hands out no new pointer once destruction has started.
- Order: worker, map, private spine, mesh mapping, client, manifest. If the
  worker or the map returns `BUSY` or an error, or `cudaFree` fails
  (`IO_ERROR`), it returns that status and keeps that object and everything
  after it. Call `SparkWeightdLazyPackDestroy` again.

### `spark_weightd_lease.h`

The daemon's per-arena lease table.

- `SparkWeightdLeaseTableCreate` allocates the table and one pin counter per
  manifest group, and keeps a pointer to the manifest, which must outlive the
  table. `SparkWeightdLeaseTableDestroy` returns `BUSY` while any lease is
  held.
- The table has no lock. The daemon serializes its calls: requests run one
  at a time on the daemon's worker (`SparkWeightdServerDispatchWork`), and
  `SparkWeightdServerCloseConnection`, which releases an attached owner's
  leases on the server thread, runs for such a connection only while no
  dispatch is in flight.
- `SparkWeightdLeaseFind` and `SparkWeightdLeaseRelease` match owner and
  identifier, and owner 0 is invalid. The daemon assigns the owner at `HELLO`
  and refuses `HELLO` rather than wrap the owner counter. Lease identifiers
  start at 1 and only increase; the table returns `CAPACITY_EXCEEDED` rather
  than wrap.
- `SparkWeightdLeaseAcquire` returns `BUSY` when all
  `SPARK_WEIGHTD_LEASE_COUNT_MAX` leases are in use, `NOT_FOUND` for a key
  missing from the manifest (logged as `weightd_parity lease_miss` with the
  layer's expert range), and `CAPACITY_EXCEEDED` when a pin counter would
  overflow. A lane that is neither below `SPARK_WEIGHTD_MESH_MAX_LANES` nor
  `SPARK_WEIGHTD_LANE_NONE` is `INVALID_ARGUMENT`.

`SparkWeightdLeaseRelease` only drops pins. It cannot see whether GPU work
still reads the chunks, so that guarantee comes from its caller:

- IPC `RELEASE`: the caller contract, and how `SparkWeightdMapRelease` gates
  it on the completion event, are under "Leases and consumer mapping". A
  pooled map keeps its chunks mapped (below).
- `ACQUIRE` rollback: a lease whose load or working-set record failed is
  released before its identifier reaches the client.
- `SparkWeightdLeaseReleaseOwner`: called only from
  `SparkWeightdServerCloseConnection`, for an owner that attached. Nothing
  checks that the consumer process actually exited.
- `SparkWeightdLeaseReleaseForLane`: called only from the `EVICT` handler.
  A connection on a lower lane number releases every lease of a
  higher-numbered target lane in every arena (an equal or higher requester
  lane gets `EVICT_DENIED`). Nothing on this path waits for, or checks, GPU
  completion by the target lane's consumers.

`SparkWeightdRouteKeys` turns host routing offsets into lease keys:

- `offsets` has `expert_count + 1` entries, the exclusive prefix sum of rows
  per expert, copied to the host after routing finished. `offsets[0]` must be
  0 and `offsets[expert_count]` must equal `packed_rows`.
- The whole array is checked first: start and end values, monotonic order
  and the `packed_rows` bound (`SCHEMA_ERROR`), then the key count against
  `capacity` and `SPARK_WEIGHTD_LEASE_GROUPS_MAX` (`CAPACITY_EXCEEDED`). No
  key is written before both checks pass, and every failure leaves `*count`
  at 0.
- It emits one `(layer, expert)` key per expert with at least one row, in
  expert order, and allocates nothing.

### `spark_weightd_manifest.h`

The wire format, validation and limits are under "Manifest format
(version 2)" and the table sizes under "Range count bound". The API adds:

- `SparkWeightdManifestLoad` allocates the range, group and spine tables, so
  it is a startup call. On success the caller owns them until
  `SparkWeightdManifestDestroy`. On failure the function frees them itself
  and leaves the struct zeroed.
- Status: `NOT_FOUND` for a missing file. `PARSE_ERROR` for a non-regular
  file, a bad magic or version, a non-zero reserved word, a zero-length,
  oversized, out-of-pack or overlapping range, a repeated kind in one group,
  a short file, or bytes after the last record. `CAPACITY_EXCEEDED` for zero
  ranges or more than `SPARK_WEIGHTD_RANGES_PER_EXPERT_MAX` ranges in one
  group, besides the range-count bound.
- Groups are sorted by (layer, expert) and the ranges of a group by kind.
  `SparkWeightdManifestFind` is a binary search over the groups.
- `SparkWeightdManifestSpineSlice` translates a pack slice to its compact
  spine offset with a binary search over the spans, O(log N). The slice must
  lie inside one spine span, that is, touch no expert range; otherwise
  `NOT_FOUND`. `spine_allocation_bytes` includes the alignment padding
  between spans, so it can exceed `spine_bytes`. For a private spine,
  `SparkWeightdLazyPackCreate` allocates it plus 255 bytes and rounds the
  base up to 256, so a slice keeps its pack alignment in device memory.

### `spark_weightd_map.h`

The consumer side of a lazy arena.

Context and ownership:

- `SparkWeightdMapCreate` records the current device and CUDA context.
  `SparkWeightdMapDestroy`, `SparkWeightdMapAcquire`,
  `SparkWeightdMapBeginUse`, `SparkWeightdMapRecordCompletion` and
  `SparkWeightdMapRelease` return `TARGET_MISMATCH` on any other context.
- `SparkWeightdMapAcquire` and `SparkWeightdMapRelease` hold an internal
  mutex around their client exchanges. `SparkWeightdMapBeginUse`,
  `SparkWeightdMapRecordCompletion` and `SparkWeightdMapDestroy` take no
  lock; the caller serializes them with the other calls.
- The map borrows the client for every `ACQUIRE`, `EXPORT_LEASE` and
  `RELEASE` and never closes it. The client, whose connection holds the lazy
  attach, must outlive the map. From the attach result the map copies only
  the generation and the chunk geometry.
- `SparkWeightdMapCreate` does all host allocation, event creation and
  address reservation: per-chunk host tables, 64 lease slots each with a CUDA event
  (`cudaEventDisableTiming`), a virtual range of the chunk span plus one
  chunk for the epoch page, and, when the daemon exported them, the
  read-only epoch page and the pooled mapping. A map holds at most 64 leases
  at once, fewer than the daemon's `SPARK_WEIGHTD_LEASE_COUNT_MAX` per arena.
  If `SparkWeightdMapCreate` fails and its cleanup also fails, `*out` holds
  the map; pass it to `SparkWeightdMapDestroy`.

`SparkWeightdMapBase` returns a base that does not change after
`SparkWeightdMapCreate`;
`SparkWeightdMapBeginUse` returns the same base. In a pooled map every pack
offset is mapped. In a per-chunk map, `base + offset` is valid only while a
lease of this map holds the covering chunk.

`SparkWeightdMapAcquire`:

- It returns only an identifier. The address comes from
  `SparkWeightdMapBeginUse`, which accepts only a lease that is acquired and
  not yet begun.
- The deadline is `now + timeout` at entry; a timeout of 0 means
  `SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS`. Each `EXPORT_LEASE` batch gets
  the remaining time, each chunk import checks it, and it is checked once
  more after the last chunk. The `ACQUIRE` exchange itself is given the full
  timeout, the wait for the internal mutex is unbounded, and a CUDA call in
  progress is not interrupted, so a call can overrun its deadline.
- If the import fails after the daemon granted the lease, the slot becomes
  retiring. With time left, the call releases it (local unmap, then
  `RELEASE`) and returns identifier 0. If no time is left or that release
  fails, the identifier stays non-zero: the caller must pass it to
  `SparkWeightdMapRelease` and must not use it on the GPU
  (`SparkWeightdMapBeginUse` refuses it).
- While any slot is retiring, or all 64 slots are in use, it returns `BUSY`.
- A map whose teardown has started returns its failure status from every
  `SparkWeightdMapAcquire` and `SparkWeightdMapBeginUse`;
  `SparkWeightdMapAcquire` logs `MAP-STICKY-FAILURE` once per process.

`SparkWeightdMapRecordCompletion` requires a begun lease and records the
slot's event on `stream`. No work may use the lease after this call.

`SparkWeightdMapRelease`:

- An acquired lease that was never begun is released at once; this is the
  cancel path.
- A recorded lease queries its event. Not ready returns `BUSY`, an error
  returns `IO_ERROR`. `BUSY` keeps the local mappings and the daemon pins.
- A pooled map never unmaps chunks on release. A daemon `NOT_FOUND` counts
  as released. A local unmap failure (`MAP-RELEASE-SLOT-ERROR`) or a daemon
  error leaves the slot retiring, which blocks new acquires; call
  `SparkWeightdMapRelease` again.

`SparkWeightdMapDestroy` returns `BUSY` while any slot is not empty, and the
map stays usable. Otherwise it marks the map failed and frees everything. If
a CUDA call fails it returns `IO_ERROR`, and from then on the map accepts
only another `SparkWeightdMapDestroy`.

### `spark_weightd_worker.h`

One thread and a FIFO ring of `SPARK_WEIGHTD_WORK_QUEUE_CAPACITY` (64) jobs.
The daemon runs every IPC request on one, except `HELLO`, `MESH_WRITE`,
`MESH_BROADCAST`, `MESH_ACTIVITY` and `MESH_STATUS`, which the server thread
answers inline. The lazy pack starts one (`SparkWeightdLazyPack.worker`) on
which drivers run their lazy acquisition jobs.

- `SparkWeightdWorkerCreate` requires a current CUDA context
  (`TARGET_MISMATCH` otherwise), makes it current on the new thread and
  returns only after the thread has started. The context must outlive the
  worker.
- Jobs run one at a time, in submission order, on that thread, not inside a
  CUDA host callback or a collective callback, so a job may call CUDA and
  block. The worker has no deadline, timeout or cancellation of its own.
  Each job bounds its own operations, signals its own completion and cleans
  up its own leases.
- `SparkWeightdWorkerSubmit` takes the mutex and copies
  `(function, context)` into the ring. It neither allocates nor waits for
  space: a full ring or a stopping worker returns `BUSY`. After `OK` the
  caller keeps `context` valid until the function returns.
  `SparkWeightdWorkerSubmit` and `SparkWeightdWorkerDestroy` must not race,
  because `SparkWeightdWorkerDestroy` frees the worker.
- `SparkWeightdWorkerWaitIdle` polls about once per millisecond on
  `CLOCK_MONOTONIC` until the ring is empty and no job runs (`OK`) or the
  timeout passes (`BUSY`). Idle is only an observation: a producer can
  submit right after it. Stop every producer before relying on it for
  teardown. On the worker thread it returns `INVALID_ARGUMENT`.
- `SparkWeightdWorkerDestroy` returns `BUSY` and keeps the worker while a
  job is queued or running; it never cancels one. When idle it stops and
  joins the thread. On the worker thread it returns `INVALID_ARGUMENT`.

## Daemon internals

How `runtime/spark_weightd.c` lays out arena memory, frees arenas, hands
chunk handles to consumers as file descriptors, and runs a connection from
accept to close.

### Arena memory

An arena is a CUDA VMM (`cuMem*`) allocation, not a `cudaMalloc` block. The
daemon reserves the virtual span once (`cuMemAddressReserve`) and maps
physical chunks into it (`cuMemCreate`, `cuMemMap`). The base address stays
fixed for the arena's life, so chunks can be created, freed or exported
without moving the base or copying data. Every physical allocation requests
`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`, so any chunk can be exported as
an fd.

Chunk size. `cuMemCreate` sizes must be multiples of the allocation
granularity. The daemon picks:

- Eager arena (`ATTACH`, `SparkWeightdVmmAllocate`): the larger of
  `SPARK_WEIGHTD_VMM_CHUNK_BYTES` (64 MiB) and the driver's recommended
  granularity, rounded up to a multiple of that granularity. All chunks are
  created and mapped at attach, and the whole span gets read-write access in
  one call.
- Lazy arena (`ATTACH_LAZY`, `SparkWeightdVmmReserve`): the larger of 2 MiB
  and the driver's minimum granularity. `SparkWeightdVmmReserve` reserves
  the span and creates only the epoch page (below). Chunks are created on demand
  (`SparkWeightdArenaChunkEnsure`) and freed when no present expert range
  covers them.
- Pooled lazy arena (see Pool sizing): the single `cuMemCreate` covers the
  whole span rounded up to the minimum granularity and is mapped in one
  piece. Chunk addressing keeps the lazy chunk size.

No chunk is smaller than 2 MiB, the VMM page size from the perf notes above.

A lazy arena reserves one chunk more than its span. The extra chunk, right
after the span, is the epoch page. `SparkWeightdVmmReserve` maps it with
read-only device access. After every eviction `SparkWeightdEvictGroup`
increments `epoch` and writes it to the page with `cudaMemcpy`, ignoring the
result.
`EPOCH_EXPORT` exports the page's handle as one fd to a connection attached to
the arena.

`SparkWeightdArena` fields:

| Field | Meaning |
|---|---|
| `identity` | The identity after `SparkWeightdIdentityPrepare`. `SparkWeightdServerFindArena` finds arenas by comparing it. |
| `device_base` | The reserved virtual base. Fixed for the arena's life; attach replies return it as `device_handle`. |
| `virtual_bytes` | `chunk_count * chunk_bytes`, the span the chunks cover. A lazy arena's reservation is one chunk larger (the epoch page), but `SparkWeightdVmmRelease` passes `virtual_bytes` to `cuMemAddressFree`. |
| `chunk_bytes` | The chunk size chosen above. It is at least the driver granularity and can be larger, so it is not the granularity value. |
| `chunk_count` | The arena bytes rounded up to whole chunks. |
| `chunk_handles` | One physical handle slot per chunk. Eager: every slot is set. Per-chunk lazy: null until the chunk is created, null again after it is freed. Pooled: every slot holds `pool_export_handle`, which `SparkWeightdFreeChunk` and `SparkWeightdVmmRelease` skip so the pool is released once. |
| `chunk_refs` | Per chunk, the number of present expert ranges that touch it. `SparkWeightdCommitLease` increments it, `SparkWeightdEvictGroup` decrements it and frees the chunk at zero. Only lazy arenas use it. An eager arena's slot holds null: `SparkWeightdServerAttachCold` copies neither this array nor the staging buffer that `SparkWeightdVmmAllocate` allocated into the slot, and nothing frees them afterwards. |
| `generation` | Taken from `next_arena_generation`; unique per arena this daemon created. Export, lease, epoch, acquire, release and detach requests name an arena by generation. An attach reference stores both generation and slot, and a lookup or refcount drop through it requires both to match. |
| `refcount` | The number of attach references held by connections. |
| `lazy` | Set for arenas created by `ATTACH_LAZY`. |
| `expert_pool_bytes` | The pool the lazy attach declared. |
| `preload_chunk_bytes` | Bytes committed when the lazy attach finished: the whole span for a pooled arena, zero otherwise. `SparkWeightdAcquireBudget` does not count them against `expert_pool_bytes`. |
| `pool_committed_bytes` | Bytes a lazy arena has committed: `chunk_bytes` per created chunk, or the span for a pooled arena (not its rounded-up allocation). This is the lazy arena's charge in `resident_bytes`. An eager arena is charged `identity.arena_bytes`, not its chunk-rounded span. |
| `expert_present_bytes` | The sum of the bytes of present expert ranges. |
| `expert_count` | The number of (layer, expert) groups in the manifest. |
| `pack_stat`, `pack_path` | A lazy arena's pack stat and path at attach. A lease load fails with `HASH_MISMATCH` if the pack's device, inode, size or mtime changed. `pack_path` also names `PACK.wset`. |
| `experts` | One entry per group: layer, expert, `present`, and `last_use_ns`, which orders LRU eviction. |
| `manifest` | The parsed `PACK.experts` manifest. |
| `failure_status` | Sticky. Set when unmapping or releasing a chunk fails, or when a lease cannot be released after a failed working-set write. Once set, `ACQUIRE` returns it and `EXPORT_LEASE` returns `INVALID_ARGUMENT`. |
| `leases` | The arena's lease table. |
| `needed_chunks` | Scratch flags per chunk: the chunks that the pinned groups (`SparkWeightdAcquireBudget`) or one lease (`SparkWeightdServerExportLease`) cover. |
| `created_chunks` | Flags per chunk for chunks the current acquisition created. A failed acquisition frees them again. |
| `epoch_device`, `epoch_handle`, `epoch` | The epoch page's address, its physical handle, and the eviction counter written to it. |
| `pool_export_handle` | The pooled arena's single allocation, or null. |
| `staging` | A 4 MiB host buffer (`SPARK_WEIGHTD_ARENA_STAGING_BYTES`) that `SparkWeightdPreloadSpine` reads through. Only a pooled arena preloads its spine in the daemon. |
| `direct` | The pack reader for lease loads (`SparkWeightdDirectCreate`, 8 MiB blocks, 4 readers), created on the first load. |
| `recorded_keys`, `recorded_count`, `recorded_capacity` | The working set mirrored to `PACK.wset`. |

### Reclaim and the NO-2x rule

`SparkWeightdServerReclaimCold` runs before a new arena is created: in
`SparkWeightdServerAttachCold` with the arena bytes, and in
`SparkWeightdServerAttachLazy` with the declared `expert_pool_bytes`. While
the resident bytes exceed `device_bytes_max`, or the needed bytes do not fit
in what remains, it frees the cold arena with the lowest `generation` (the
oldest created, not the least recently used). An arena is cold when its
`refcount` is 0 and it holds no leases. When no cold arena is left it
returns without a status, and the caller applies its own checks.

- After reclaim, both attaches refuse a full arena table. The eager attach
  also requires `resident_bytes + arena_bytes` to fit under
  `device_bytes_max`. The lazy attach is charged later, as
  `SparkWeightdPremapPool` or `SparkWeightdArenaChunkEnsure` commits memory,
  and each of those fails with `CAPACITY_EXCEEDED` when the budget is
  exceeded.
- Reclaim is driven by bytes only. When the new arena's bytes fit, reclaim
  frees nothing, so with `SPARK_WEIGHTD_ARENA_COUNT_MAX` (16) arenas resident
  a new arena fails with `CAPACITY_EXCEEDED` even when some of them are cold.
  `RECLAIM` or `RECLAIM_PACK` frees those.
- An arena with an attach reference or a lease is never reclaimed. This is
  how the daemon holds the NO-2x constraint from the perf notes: it never
  frees a serving process's arena to make room. If a model update's new
  arena does not fit next to the live one, an eager or pooled attach fails
  with `CAPACITY_EXCEEDED`; a per-chunk lazy arena fails later, when a chunk
  cannot be committed. Stopping the old consumer drops its references, and
  the next attach reclaims the now cold arena.

`SparkWeightdServerFreeArenaSlot` frees one arena:

- It releases the VMM memory (`SparkWeightdVmmRelease`), subtracts the
  arena's `resident_bytes` charge, and frees its manifest, expert, lease and
  chunk tables.
- The arena table has no order, because lookups compare identities. The
  freed slot is closed by moving the last arena into it. The moved arena's
  lease table is re-pointed at the moved manifest, and every connection's
  attach reference with the moved generation is rewritten to the new slot.
  `SparkWeightdServerDetachRelease` decrements a refcount only when both slot
  and generation match. Without the rewrite, the moved arena would keep a
  nonzero refcount forever and could never be reclaimed.
- `SparkWeightdVmmRelease` ignores every CUDA result, because nothing can
  recover at that point. A non-pooled arena unmaps only chunks whose handle
  is set. A pooled arena unmaps its span in one call and releases the pool
  handle once. Host tests catch ordering errors and leaks through the CUDA
  stub: its `cuMemAddressFree` refuses a reservation that still has mappings
  (it ignores the size argument), and `test_weightd`, `test_weightd_attach`,
  `test_weightd_churn` and `test_weightd_expert_stress` assert
  `spark_stub_cuda_outstanding_allocs() == 0` after teardown.
- The reclaim paths free only arenas without attach references or leases.
  The failed-attach paths free an arena that the same request created.
  `SparkWeightdServerDestroy` closes every connection, which releases its
  leases and attach references, before it frees the remaining arenas.

### Chunk export (fd tier)

Replies that carry file descriptors:

| Request | Descriptors in the reply |
|---|---|
| `ATTACH_LAZY`, `ATTACH_LAZY_SHARED` | The mesh region fd when the mesh is ready, then the pool fd of a pooled arena (0 to 2). |
| `EXPORT` | Up to `SPARK_WEIGHTD_EXPORT_BATCH_MAX` (64) chunk fds of an eager arena, from chunk `batch_offset` on. |
| `EXPORT_LEASE` | Up to 64 chunk fds of one lease's chunk union. None for a pooled arena: its consumer maps the pool fd from the attach reply. |
| `EPOCH_EXPORT` | The lazy arena's epoch page. |
| `MESH_MAP`, `MESH_STAGING_MAP` | The mesh region or the staging area. |

Each chunk fd comes from `cuMemExportToShareableHandle` and is set
close-on-exec (`SparkWeightdServerExportOne`). If one export in an `EXPORT`
or `EXPORT_LEASE` batch fails, every fd staged for the reply is closed and
the reply is `IO_ERROR`. A failed `EPOCH_EXPORT` closes every fd staged for
the reply and returns `NOT_FOUND`.

Access:

- `EXPORT`, `EXPORT_LEASE` and `EPOCH_EXPORT` serve only an arena whose
  generation this connection holds an attach reference for, and return
  `NOT_FOUND` otherwise. `EXPORT_LEASE` also requires the lease to belong to
  this connection's owner. The attach reference is the export capability.
- `EXPORT` also returns `NOT_FOUND` when a chunk in the batch has no
  handle. `EXPORT_LEASE` returns `IO_ERROR` in that case.
- `SparkWeightdServerListen` sets the socket to mode 0600 after `bind`.
  There is no peer-credential check, so any process running as the daemon's
  user, or as root, can connect, attach by identity and then export.

Daemon side:

- Exported fds wait in `response_fds` (`response_fd_count`) until the reply
  is sent.
- `SparkWeightdServerFlushResponse` sends a reply that has staged fds
  through `sendmsg` with one `SCM_RIGHTS` control message, starting at byte 0
  of the frame.
- Once `sendmsg` succeeds, the kernel holds its own references for the
  receiver. The daemon then closes its copies, whatever the byte count, and
  the rest of the frame goes out through `write`. The fds therefore cross
  exactly once, always with the frame's first bytes.
- On `EINTR` the send is retried. On `EAGAIN` nothing was sent, and the same
  fds go with the next attempt on a later step. On any other error the
  connection is closed, and `SparkWeightdServerCloseConnection` closes the
  unsent fds. Server teardown closes them the same way.

Mesh fd on lazy attach:

- `SparkWeightdServerStageMeshFd` runs on the server thread after the
  worker has produced a successful lazy attach reply. It puts the mesh fd
  ahead of the pool fd.
- `mesh_send_buffer_addr` is the daemon's own address of the mesh region and
  means nothing in another process. The daemon fills it only when the mesh is
  ready. `SparkWeightdClientAttachLazyKind` replaces it with the client's
  own mapping of the mesh fd (`SparkWeightdMapMeshFd`), or zeroes it when the
  mesh is not ready.
- If the mesh fd cannot be staged, every staged fd is closed and the reply
  becomes `IO_ERROR`.

Client side (`SparkWeightdClientReadFrameWithFds`):

- The client collects fds only from its first `recvmsg` of a frame. Later
  reads of the same frame use `recv`.
- The control buffer has room for at least 253 descriptors, the Linux
  per-message limit, so every fd the kernel delivers is seen, even past the
  protocol cap. A truncated control message (`MSG_CTRUNC`) fails the read.
- Each fd is made close-on-exec. Fds beyond the caller's capacity (64
  through `SparkWeightdClientExportExchange`, 2 for a lazy attach) are
  closed and fail the read.
- Any failure closes every fd received so far and returns `IO_ERROR`, so the
  caller never gets a partial set. Failures include over capacity, a
  truncated or malformed control message, an expired deadline or failed
  `poll`, and EOF or a receive error before the frame is complete. The
  exchange then closes the client socket.
- The caller then checks the fd count against the frame.
  `SparkWeightdClientExportBatch` requires the request id, the batch offset,
  `batch_count <= 64` and `batch_count` equal to the fds received. A lazy
  attach requires exactly one fd for each of `mesh_ready` and
  `pool_fd_staged`. A mismatch closes every fd and the socket. The export
  returns `SCHEMA_ERROR`, or the header validation error when the header
  itself is wrong; the lazy attach returns `SCHEMA_ERROR` in both cases. A
  batch is exactly what its frame declares, or nothing.
- `SparkWeightdAttachImportMap` (`runtime/spark_weightd_attach.c`) closes
  each chunk fd right after `cuMemImportFromShareableHandle` succeeds. A
  failed import closes the batch's remaining fds.

### Request dispatch

- The server thread (`SparkWeightdServerStep`) polls the listen socket, the
  worker's notify pipe and every open connection. The poll timeout is
  `SPARK_WEIGHTD_SERVER_POLL_TIMEOUT_MS` (20 ms).
- `HELLO`, `MESH_WRITE`, `MESH_BROADCAST`, `MESH_ACTIVITY` and `MESH_STATUS`
  run inline on the server thread. Every other kind marks its connection
  `request_ready`.
- The worker (`SparkWeightdWorkerSubmit` with
  `SparkWeightdServerDispatchWork`) runs one ready request at a time. The
  server thread picks the next ready connection round-robin, starting at
  `dispatch_next`. When the worker finishes it writes one byte to
  `dispatch_notify`, and `SparkWeightdServerCompleteWork` finishes the reply
  on the server thread.
- `HELLO` reports `resident_bytes` and the arena count from the published
  atomics. `SparkWeightdServerPublish` refreshes those only when no dispatch
  is running.
- If dispatch produces zero reply bytes, the connection is closed without a
  reply. That happens on a second `HELLO` on the same connection, any other
  kind before `HELLO`, or an exhausted owner counter
  (`next_owner == UINT64_MAX`).

### Connection lifecycle

Accept and read:

- When the listen socket is readable, the server accepts pending
  connections into free slots of the connection table
  (`SPARK_WEIGHTD_CONNECTION_COUNT_MAX`, 128) until `accept` fails or the
  table is full, and makes each socket non-blocking. When the table is full
  it accepts nothing. The peer waits in the listen backlog (also 128) and
  fails on its own client deadline.
- `SparkWeightdServerCreateUnbound` sets `SIGPIPE` to `SIG_IGN` for the
  process, so a write to a dead peer returns an error instead of killing it.
- `SparkWeightdServerHandleReadable` stops reading a connection while its
  request waits for the worker or its reply is not fully sent. A connection
  therefore holds at most one request and one reply. A peer that does not
  read its replies stalls only its own connection.

Framing:

- The connection is closed without a reply when the header has the wrong
  magic, an ABI that `SparkWeightdIpcAbiServed` refuses for that kind or
  that differs from the connection's earlier frames (see ABI 9: per-peer
  routes, served ABI range, row cap), a kind that is not a request kind, or
  a body larger than `SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX` minus the header.
- Once the body has arrived, `SparkWeightdIpcValidateHeaderVersion` requires
  the exact body size of the kind. A mismatch also closes the connection
  without a reply.

Write:

- `SparkWeightdServerFlushResponse` writes what the socket takes. On
  `EAGAIN` it returns `PENDING` and retries on the next step. Only a real
  write error closes the connection.
- A connection is not flushed while the worker runs its request.

Attach references (`SparkWeightdServerAttachRegister`):

- A connection holds at most one attach reference per arena. A second attach
  to the same arena on the same connection returns `DUPLICATE`.
- A connection holds at most `SPARK_WEIGHTD_ATTACHES_PER_CONNECTION_MAX` (8)
  references. Past that, an attach returns `CAPACITY_EXCEEDED`.
- Each reference increments the arena's `refcount`.
- Warm hit: an `ATTACH` whose identity is already resident registers a
  reference and returns the existing `device_base` with
  `loaded_from_pack = 0`. It neither opens nor hashes the pack.
- An `ATTACH` whose identity is resident as a lazy arena returns
  `INVALID_ARGUMENT`.
- If a cold load succeeds but the connection cannot take the reference, the
  new arena is freed at once instead of being left cold.

Close:

- A connection is marked closed on EOF (consumer death), a receive error,
  `POLLHUP`/`POLLERR`/`POLLNVAL`, a framing fault, a refused dispatch or a
  write error.
- `SparkWeightdServerCloseConnection` runs once no worker dispatch is in
  flight. It runs earlier only if the running dispatch belongs to another
  connection and the closing one holds no attach references and no lanes.
- `SparkWeightdServerCloseConnection` does the following:
  - quarantines an active mesh lane (see Concurrent mesh lane
    reservations);
  - releases the owner's leases and drops every attach reference (see
    Leases and consumer mapping). Each drop decrements its arena's
    `refcount`, so a dead consumer leaves no reference behind;
  - closes the socket and releases owned lanes;
  - closes any staged fds that were never sent.

Stop and teardown:

- `SparkWeightdServerRun` repeats `SparkWeightdServerStep` until `*stop` is
  nonzero or a step returns an error. It reads the flag with a sequentially
  consistent atomic load.
- `node/weightd.c` sets the flag from its `SIGINT`/`SIGTERM` handler. Host
  tests set it from another thread.
- A step returns within one poll timeout plus its own inline work, so the
  flag is checked at least that often.
- `SparkWeightdServerDestroy` does the following, in order:
  - waits for the worker to go idle and destroys it. If destroying the
    worker fails, it returns without freeing anything else;
  - closes every connection and frees every arena;
  - closes the listen socket and the notify pipe;
  - unlinks the socket path, but only if this instance bound it.

### Client deadlines and daemon loss

- One deadline covers a whole exchange, write and read. It comes from the
  caller's timeout, or `SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS` (10 s) when
  the caller passes none.
- `SparkWeightdClientReadAll` reads exactly the frame size. An expired
  deadline or a failed `poll` returns `BUSY`. EOF or a receive error before
  the frame is complete returns `IO_ERROR`: the daemon is gone or broke
  protocol.
- `SparkWeightdClientExchange` closes the client socket and sets it to -1
  when any step fails, including a header or request-id mismatch. Later
  calls on that client fail with `IO_ERROR`, and the client object does not
  reconnect.
- This is the client half of the crash semantics in the perf notes: once the
  daemon is lost, the next exchange fails, and so does every later call on
  that client.

### Darwin build

On Apple builds `runtime/spark_weightd.c` defines `_DARWIN_C_SOURCE` before
its includes. Darwin's `struct stat` names the nanosecond modification time
`st_mtimespec` only when `_POSIX_C_SOURCE` is unset or `_DARWIN_C_SOURCE` is
defined. The file reads modification times only through
`SparkWeightdPackMtimeNs` and `SparkWeightdPackStatSame`. Both live in
`runtime/spark_weightd_receipt.c`, which defines `_DARWIN_C_SOURCE` itself.

## Operator tools

Most of these tools are clients of a running weightd. They take its socket
from the command line or from `SPARK_WEIGHTD_SOCKET`. The exceptions:
`weightd_expert_segments` and `weightd_lazy_consumer prepare` work on files
only, `weightdctl status` and `weightd_warm --identity-print` never connect,
and `sparkpipe_weightd_vmm_verify.sh` starts its own server.

`weightd_warm`, `weightdctl`, `weightd_smoke`, `weightd_loadall` and
`weightd_lazy_consumer` send the PACK path unchanged, and the daemon opens it
relative to its own working directory, so pass absolute paths.

`build/weightd_warm`, `build/weightdctl` and `build/weightd_lazy_consumer` have
Makefile targets. Only `tools/weightd_lazy_bake.sh` compiles `weightd_smoke`
and `weightd_expert_segments`. Nothing builds `weightd_loadall`.

### weightd_warm

Lazily attaches one rank pack under an identity and loads its experts, or a
recorded working set, into the arena. A resident that later attaches with the
same identity, manifest and pool finds the arena warm. The tool also runs the
reclaim commands.

```
weightd_warm SOCKET PACK SHA256 REVISION TOPOLOGY [LAYERS [EXPERTS [TIMEOUT_S]]]
weightd_warm SOCKET PACK SHA256 REVISION TOPOLOGY --wset FILE [TIMEOUT_S]
weightd_warm SOCKET --reclaim
weightd_warm SOCKET --reclaim-pack PACK|SHA256 [...]
```

`--family NAME`, `--world-rank R` and `--identity-print` can appear anywhere.
The tool removes them before it reads the positional arguments. At most 24
arguments may remain, counting the program name.

Warm mode:

- `SPARK_WEIGHTD_EXPERT_POOL_BYTES` is required except with `--identity-print`.
  It must be a positive decimal no larger than
  `SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT`, and it becomes `expert_pool_bytes`.
  Set it to the pool the resident will declare.
- SHA256 must be 64 hex digits. The tool accepts upper-case digits, but the
  attach identity check accepts only lowercase, so an upper-case SHA256 fails
  the attach (exit 1). The tool does not hash the pack; the daemon verifies it
  when it creates the arena. For every family `arena_bytes` is the pack's file
  size, because the daemon rejects a lazy attach whose `arena_bytes` differs
  from the pack size.
- The tool loads `PACK.experts` itself, bounded by the pack size, and refuses
  an empty manifest.
- Without `--wset`, every manifest group must have a layer below LAYERS
  (default 45) and an expert below EXPERTS (default 288, at most
  `SPARK_WEIGHTD_EXPERT_COUNT_MAX`). Otherwise the tool refuses the run. It
  then takes one lease per layer covering all of that layer's groups, releases
  it, and logs `layer N WARM experts=M`. A layer with more than
  `SPARK_WEIGHTD_LEASE_GROUPS_MAX` (512) groups fails the run. TIMEOUT_S
  (default 1800) bounds the attach and every acquire and release.
- `--wset FILE` replays a recorded working set instead (see Working-set
  recording). FILE must be a non-empty regular file of whole 8-byte
  (layer, expert) pairs, at most 1,048,576 pairs, and every pair must be in the
  manifest. Duplicates are dropped and the order of first occurrence is kept.
  The manifest geometry check against LAYERS and EXPERTS is skipped.
  TIMEOUT_S defaults to 300.
- Releasing a lease only unpins its groups; the loaded chunks stay in the pool
  (see Pool sizing for when they are evicted).
- Because the daemon records acquired keys (see Working-set recording), a full
  warm writes every manifest key to `PACK.wset`, and an acquire that adds new
  keys fails when the daemon cannot write that file.

Family identities. `--family` replaces the identity fields. For
`dsv41_flash`, `k3` and `dsv4_pro`, and with no `--family` for glm5_next, the
result equals the identity that family's resident sends:

| `--family` | model | revision | topology | geometry |
|---|---|---|---|---|
| none | `glm5_next_stage` | REVISION | TOPOLOGY | 0 |
| `dsv41_flash` | `dsv41_flash_stage` | REVISION | TOPOLOGY | 0 |
| `ling` | `ling_stage` | REVISION | TOPOLOGY | 0 |
| `k3` | `kimi-k3` | `mxfp4` | TOPOLOGY (4 or 16) | 0 |
| `dsv4_pro` | `dsv4` | empty | derived | derived |

- With no `--family`, the identity matches the glm5_next module's lazy attach:
  `SPARK_GLM5_NEXT_MODULE_TAG`, the stage `model_revision` and `tp_degree`.
- `dsv41_flash` matches the dsv41_flash module's lazy attach:
  `SPARK_DSV41_FLASH_MODULE_TAG`, the stage `model_revision` and `tp_degree`.
  Pass the stage config's revision and TP degree as REVISION and TOPOLOGY.
- `ling` sets the model to `SPARK_LING_MODULE_TAG`. The ling module reads its
  pack directly and sends no weightd attach, so no ling resident uses this
  identity.
- `k3` matches the k3 runner's lazy attach. TOPOLOGY is the runner's
  `tp_degree`: 4 for TP4xPP4 or 16 for TP16. Any other value exits 2.
- `dsv4_pro` requires `--world-rank R` (0-15) and derives the identity that
  `SparkDsv4ModuleWeightdAttach` sends:
  - Topology is the low 32 bits of the `configuration_hash` that
    `SparkDsv4TpDeriveNodeConfig` returns for a TP4xPP4 shape with
    `tp_rank = R % 4` and `pp_stage_index = R / 4`.
  - Geometry is an FNV-1a chain (`SparkHashBytes`, offset basis
    1469598103934665603). It covers, in this order: `format_version`,
    `codec_abi_version`, `linear_weight_codec`, `expert_weight_codec`,
    `kv_cache_codec`, `tensor_count`, `first_layer_index`, `layer_count`,
    `total_layer_count`, `hidden_dimension`, `vocab_count`,
    `routed_expert_count` and `mtp_layer_count` (u32 each), then `file_bytes`
    (u64). The tool reads these fields from the 80-byte
    `SparkDsv4StagePackHeader` at the start of PACK.
  - REVISION and TOPOLOGY must still be given (TOPOLOGY as a positive
    number), but the derived values replace them.
  - `tests/test_dsv4_pro_weightd_warm_identity.py` checks the field order
    against the module and the rank-to-shape mapping against the serving
    adapter.
  - The identity matches, but the module cannot use the arena. The module
    attaches through `SparkWeightdAttachMappedPack`, and `weightd_warm` always
    creates a lazy arena. An eager attach to a lazy arena returns
    `INVALID_ARGUMENT`. With `SPARK_WEIGHTD_ATTACH_LAZY` set, the helper makes
    a lazy attach with a pool of pack size + 2 MiB, which succeeds only when
    the warm run declared that pool. The helper's import map then needs
    whole-arena `EXPORT`, which rejects lazy arenas (see Leases and consumer
    mapping).

`--identity-print` prints
`identity model=… revision=… topology=… geometry=… arena_bytes=… pack_sha256=…`
on stdout and exits 0 without connecting. PACK must still exist and SHA256 must
still be 64 hex digits.

Reclaim. Pack-scoped reclaim (above) covers what `--reclaim` and
`--reclaim-pack` free, along with the exit codes for `--reclaim-pack`. An
attach with a matching identity always goes to the existing arena and never
creates a new one, and the daemon frees cold arenas on its own only when a new
arena needs their device memory. A cold arena created with another
`expert_pool_bytes` (see Pool sizing), or whose manifest ranges differ from
the current `PACK.experts` (`HASH_MISMATCH`), therefore keeps refusing the
correct attach until a reclaim frees it. `--reclaim` logs
`RECLAIM freed=… arenas=… resident=… busy=…`, where `arenas` is the number of
arenas freed. It exits 0 when the daemon answers and 1 on a connect or daemon
error. `--reclaim-pack` takes at most 21 PACK or SHA256 arguments (more is a
usage error) and lowercases a hex SHA256 argument.

Exit codes in warm mode:

- 0: every layer or the whole working set warmed, or `--identity-print`
  succeeded.
- 1: manifest load failure or an empty manifest; a manifest expert outside
  LAYERS or EXPERTS; an invalid wset; a connect or attach failure; a layer over
  512 groups; an acquire or release failure.
- 2: usage error. This covers a bad argument count; a missing, zero,
  non-numeric or too large number (EXPERTS above
  `SPARK_WEIGHTD_EXPERT_COUNT_MAX`, TOPOLOGY or LAYERS above 32 bits); an
  invalid `--world-rank`; a missing, non-numeric or out-of-range pool
  variable; an unknown family; `dsv4_pro` without `--world-rank`; a PACK
  that is missing, not a regular file or empty; an overlong PACK path; a
  malformed SHA256; an overlong REVISION; a failed `dsv4_pro` derivation; a
  `k3` TOPOLOGY other than 4 or 16.

### weightdctl

Attaches one pack by hand through the resident attach helper
`SparkWeightdAttachPack`.

```
weightdctl load   PACK MODEL REVISION
weightdctl unload PACK MODEL REVISION
weightdctl status
weightdctl reclaim
```

- `load` attaches PACK with the identity (MODEL, REVISION, topology 0,
  geometry 0, `arena_bytes` = pack size). It prints
  `ATTACHED load arena_bytes=… generation=… cold=… refcount=…` and closes the
  connection, which drops this attach reference. The arena stays resident
  until a reclaim frees it, or until the daemon frees cold arenas, oldest
  first, because a new arena needs their device memory.
- `unload` runs the same code as `load`; only the printed mode differs. It
  neither detaches nor frees anything, and it cold-loads the pack when no
  arena matches. To free arenas, use `reclaim` or
  `weightd_warm SOCKET --reclaim-pack`.
- `status` does not contact the daemon. It prints one line,
  `requested=N socket=…`. N is 1 when `SparkWeightdAttachRequested` accepts
  the environment (`SPARK_WEIGHTD_SOCKET` non-empty and `SPARK_WEIGHTD_ATTACH`
  unset or `1`), else 0. `socket=` shows the variable's value, or `(unset)`.
  It always exits 0.
- `reclaim` sends `RECLAIM`, which frees every cold arena on the node, and
  prints `RECLAIM status=… reclaimed_bytes=… arenas=… resident=…`. Here
  `arenas` is the number of arenas left afterwards. `tools/weightsd_deploy.sh`
  runs `reclaim` as its health check, so that check frees cold arenas.

Environment for `load` and `unload`:

- `SPARK_WEIGHTD_SOCKET` is required and must be non-empty.
  `SPARK_WEIGHTD_ATTACH` must be unset or `1`; `0` or any other value fails.
  `SPARK_WEIGHTD_SHARE` must be unset or empty; any other value fails with
  `UNSUPPORTED`.
- `SPARK_WEIGHTD_PACK_SHA256` is the digest when set. When it is unset, the
  tool hashes the whole pack. Set to an empty value, the attach fails.
- `SPARK_WEIGHTD_IDENTITY_MODEL` and `SPARK_WEIGHTD_IDENTITY_REVISION` replace
  MODEL and REVISION when they are non-empty.
- A non-empty `SPARK_WEIGHTD_ATTACH_LAZY` switches to a lazy attach with a pool
  of pack size + 2 MiB. `PACK.experts` is then required and `cold` always
  prints 0.
- `SPARK_WEIGHTDCTL_TIMEOUT_NS` is the attach timeout in nanoseconds. The
  default is 300 s; an empty, non-numeric or 0 value also gives the default.

Exit codes:

- `load` and `unload`: 0 attached; 1 attach failed; 2 usage error, unreadable
  or empty pack, or hash failure. The tool also exits 3 when an attach
  succeeds without a client connection, which `SparkWeightdAttachPack` does not
  return.
- `reclaim`: 0 the daemon answered; 1 no daemon or the exchange failed; 2
  `SPARK_WEIGHTD_SOCKET` unset.

### weightd_lazy_consumer

A synthetic probe of the lazy lease and consumer-mapping path that can run as
several processes at once. It builds its own fixture and never reads a model
pack.

```
weightd_lazy_consumer prepare PACK
weightd_lazy_consumer SOCKET PACK consumer 0|1|2
weightd_lazy_consumer SOCKET PACK pressure
```

- `prepare` creates PACK and refuses when PACK or `PACK.experts` already
  exists. PACK holds four 2 MiB chunks, and chunk e is filled with byte e+1.
  `PACK.experts` is a version 2 manifest with layer 0 and expert e, and two
  4096-byte ranges (kinds 0 and 1) at the start of each chunk.
- `consumer F` runs these steps:
  1. Initialize CUDA and hash PACK.
  2. Create a `SparkWeightdLazyPack` with model `lazy-consumer-probe`,
     `arena_bytes` 8 MiB and a pool of three chunks.
  3. Check that the daemon's chunk size is 2 MiB.
  4. Acquire experts F and F+1 in one lease.
  5. Start an 8 KiB device-to-host copy from each expert and record
     completion.
  6. Print `READY` and wait up to 10 s for the byte `G` on stdin.
  7. Synchronize, check every byte, release the lease and destroy the pack.
  8. Print `PASS consumer-local lazy reads`.
- `pressure` acquires expert 3 alone and expects `CAPACITY_EXCEEDED`, because
  other consumers' pinned chunks fill the pool. If the daemon grants a lease
  anyway, the probe releases it and fails. On success it prints
  `PASS pinned pool rejects fourth chunk`.

On any failure the probe prints `lazy probe failed: <code>` and exits without
releasing its lease, because GPU work may still be reading the leased chunks;
the daemon releases it when the connection closes (see Leases and consumer
mapping). Code -15 means the probe did not receive `G` as the first stdin byte
within 10 s. Exit codes: 0 pass, 1 failure, 2 usage.

`tools/weightd_lazy_pair.py` prepares a fixture, starts its own daemon with an
8 MiB device budget, and uses the probe in this order:

1. Run consumers 0 and 1 concurrently (sets {0,1} and {1,2}, so 3 of 4 chunks
   are pinned).
2. Run `pressure` while both wait at `READY`.
3. Release both consumers with `G`.
4. Run consumer 2 to show that released chunks can be evicted.

`tools/weightd_execute_receipt.py` runs `consumer 0` alone, once per fixture
run.

### weightd_smoke

```
weightd_smoke PACK MODEL REVISION POOL_MIB TOUCHES
```

Lazily attaches PACK with a pool of POOL_MIB MiB. It then sends TOUCHES legacy
single-expert `ENSURE` requests for layer 0, spread evenly over expert ids 0 to
`expert_count - 1`, detaches, and prints
`SMOKE experts=… touches=… ensure_ns=… pool_mib=…`.

- The socket comes from `SPARK_WEIGHTD_SOCKET` only. The tool does not read
  `SPARK_WEIGHTD_ATTACH`.
- The digest comes from `PACK.sha256`. The tool never reads `PACK.experts`,
  but the daemon needs a version 2 one for the lazy attach.
- The identity is MODEL, REVISION, topology 16, geometry fingerprint
  `0x504F4355` and `arena_bytes` = pack size.
- The daemon answers every `ENSURE` with `UNSUPPORTED` (see Leases and
  consumer mapping), so even after a successful attach a run fails at the
  first touch with exit 1.
- `tools/weightd_lazy_bake.sh` runs `weightd_expert_segments` on each pack it
  discovers that has no `PACK.experts`, and then runs this tool on every
  discovered pack except those whose segment generation failed.

Exit codes:

- 0: success.
- 1: connect failure, invalid identity (including an unreadable pack), attach
  failure or ensure failure.
- 2: usage error, unset socket variable, zero or non-numeric POOL_MIB or
  TOUCHES, missing or malformed `PACK.sha256`, or an argument too long for its
  field.

### weightd_loadall

```
weightd_loadall ROSTER POOL_MIB TOUCHES
```

Each ROSTER line is `<pack-path> <model>`, split at the first space. Trailing
newlines, carriage returns and spaces are trimmed. The tool skips blank lines,
lines without a space or with an empty field, and lines whose pack path is
`SPARK_WEIGHTD_PATH_BYTES` (1024) bytes or longer or whose model is
`SPARK_WEIGHTD_ID_BYTES` (64) bytes or longer. It ignores every line after the
64th accepted arm.

On one connection it lazily attaches every arm with the identity: revision
`loadall`, topology 16, geometry `0x504F4355`, `arena_bytes` = pack size,
digest from `PACK.sha256`, and the same pool of POOL_MIB MiB for every arm. It
then sends TOUCHES `ENSURE` requests per arm, spread the same way as in
`weightd_smoke`, and prints
`LOADALL arms=… touches=… cold_faults=… ensure_ns=…`. It never detaches.
Closing the connection drops its attach references and leaves the arenas
cold. The socket comes from `SPARK_WEIGHTD_SOCKET`; the tool does not read
`SPARK_WEIGHTD_ATTACH`.

Daemon limits that apply:

- `SPARK_WEIGHTD_ATTACHES_PER_CONNECTION_MAX` (8): the ninth attach on the
  connection fails with `CAPACITY_EXCEEDED`.
- A new arena fails with `CAPACITY_EXCEEDED` when the daemon already holds
  `SPARK_WEIGHTD_ARENA_COUNT_MAX` (16) arenas, counting other clients' and
  cold ones.
- The same pack listed under two model names loads two arenas, and the same
  pack and model listed twice fails with `DUPLICATE`.
- `ENSURE` returns `UNSUPPORTED` (see `weightd_smoke`), so a run exits 1 at
  the first touch.

Exit codes:

- 0: success.
- 1: connect failure, missing or malformed `PACK.sha256`, invalid identity,
  attach failure or ensure failure.
- 2: usage error, unset `SPARK_WEIGHTD_SOCKET`, zero or non-numeric POOL_MIB
  or TOUCHES, or an unreadable roster or one with no accepted line.

### weightd_expert_segments

```
weightd_expert_segments PACK [SEGMENT_BYTES]
```

Writes `PACK.experts` without knowing the pack format, replacing any existing
file:

- PACK is cut into fixed segments of SEGMENT_BYTES (default 64 MiB, allowed
  range 2 MiB to 1 GiB). The last segment may be shorter.
- Each segment gets a ck128 digest. Its layer is 0 and its expert id is the
  segment index.
- Reads go through a single 8 MiB `pread` buffer, so memory use is bounded for
  any pack size. Except on macOS builds, the tool then drops the pack from the
  page cache (`POSIX_FADV_DONTNEED`).
- A pack may have at most `SPARK_WEIGHTD_EXPERT_COUNT_MAX` (40960) segments.

The tool prints `SEGMENTS file_bytes=… segment=… count=… ns=…`. It exits 0 on
success and 2 on any error.

The output is manifest version 1 (`SPARK_WEIGHTD_EXPERT_MANIFEST_VERSION`): a
16-byte header followed by 40-byte records (layer, expert, offset, bytes,
ck128) with no range kind. `SparkWeightdManifestLoad` accepts only the
version 2 format documented above, so the daemon refuses a lazy attach that
relies on this file with `PARSE_ERROR`, and `weightd_warm` refuses it as well.
Running the tool over a pack that has a version 2 `PACK.experts` replaces a
usable manifest with one the daemon refuses.

### sparkpipe_weightd_vmm_verify.sh

A GPU receipt for the eager VMM arena and the whole-arena fd export and import
on real CUDA. Host tests use a CUDA stub. No gate or Makefile target runs it.
Run it on a GB10 node:

```
tools/sparkpipe_weightd_vmm_verify.sh
```

The script prints `SKIP …` and exits 0 in any of these cases:

- `$CUDA_HOME/include/cuda.h` (default `/usr/local/cuda`) is missing.
- `cc` is missing.
- `nvidia-smi` is missing or fails.

Otherwise it writes `vmm_verify.c` to a temporary directory, compiles it with
weightd runtime sources against `libcudart` and `libcuda`, and runs it. On
success the program prints `VMM VERIFY PASS`, the script prints
`sparkpipe_weightd VMM receipt: green`, and the exit code is 0. A build or leg
failure exits 1. A failure after the pack is written leaves the per-pid pack
file in `/tmp`, and a failure after the server starts also leaves its socket
there.

The legs run in order, and any failure exits 1:

1. Granularity: `cuMemGetAllocationGranularity` (recommended, pinned device
   memory, POSIX fd handles) must be at least 2 MiB. The leg does not check
   the chunk size the daemon then uses.

   After this leg the program sets up the rest:
   - It writes a deterministic 8 MiB pack to
     `/tmp/spark_weightd_vmm_verify_<pid>.spack`. The byte at offset o is
     `(o*131 + o/512) mod 256`.
   - It uses the identity model `vmm-verify`, revision `gpu`, topology 1,
     geometry 1.
   - It starts an in-process weightd server thread
     (`SparkWeightdServerCreate`, device budget 32 MiB) on
     `/tmp/spark_weightd_vmm_verify_<pid>.sock`.
   - The per-pid paths let concurrent runs coexist.
2. Cold attach: an eager attach must load the pack (`loaded_from_pack` 1) and
   return a device address. A device-to-host copy from that address must equal
   the pack. The address is usable here only because the server runs in the
   same process.
3. Warm attach: an attach on a second connection must not reload and must
   report refcount 2. The leg does not compare the two handles.
4. In-process import: the leg sets `SPARK_WEIGHTD_SOCKET` and
   `SPARK_WEIGHTD_PACK_SHA256`, and `SparkWeightdAttachPack` must be a warm hit
   with refcount 3. `SparkWeightdAttachImportMap` then receives the arena's
   chunks as POSIX fds through whole-arena `EXPORT`, checks that they cover
   8 MiB, imports them, reserves a local VA range and maps them read-write. A
   copy from that mapping must equal the pack. Releasing the helper closes its
   connection, which drops its attach reference.
5. Cross-process import: the program re-executes itself through
   `/proc/self/exe` with `--consumer PACK HEX`. The child attaches through
   `SparkWeightdAttachPack` with the digest from the inherited environment; it
   does not use the HEX argument. The attach must be warm. The child maps the
   arena through `SparkWeightdAttachImportMap` in its own process, receiving
   the chunk fds over the socket (`SCM_RIGHTS`) as in leg 4, and compares its
   mapping with the pack file it read itself. It prints
   `consumer import map verified bytes=8388608` and must exit 0.
6. Re-attach after the child exits: an attach on a new connection must be
   warm and keep the arena generation from leg 2.
7. Teardown: every connection closes, the server thread stops and
   `SparkWeightdServerDestroy` runs, then the pack and socket files are
   removed. The leg passes if the process gets this far and exits 0. Driver
   state after the destroy is not inspected.

Legs 4 and 5 go through `SparkWeightdAttachPack`, which reads the environment.
Run the script with `SPARK_WEIGHTD_ATTACH` unset or `1`, and with these
variables unset or empty:

- `SPARK_WEIGHTD_ATTACH_LAZY`
- `SPARK_WEIGHTD_SHARE`
- `SPARK_WEIGHTD_IDENTITY_MODEL`
- `SPARK_WEIGHTD_IDENTITY_REVISION`

Otherwise those legs take another path or fail.

The compile line links only `runtime/spark_weightd.c`,
`runtime/spark_weightd_attach.c`, `src/spark_sha256.c` and
`src/spark_status.c`. `runtime/spark_weightd.c` calls functions defined in
other files, for example `SparkWeightdManifestLoad`
(`runtime/spark_weightd_manifest.c`), `SparkWeightdWorkerCreate`
(`runtime/spark_weightd_worker.c`) and `SparkCk128Initialize`
(`src/spark_ck128.c`), so the link fails and the script exits 1 before any
leg runs.
