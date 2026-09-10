# Weightd-owned mesh: the deploy cycle floor

## The reframe

RDMA queue pairs are kernel objects that die with their process. Putting them
in the residentd — the shortest-lived process on the node — means every deploy
invalidates all 450 QPs fleet-wide and forces a full record exchange before
the mesh can reform. This is the 10-15 second floor of every deploy cycle.

Moving the mesh to weightd (the persistent daemon, restarted only on binary
updates or node reboots) makes records static per node boot. Deploys become
pure compute state changes: kill residentd, start new one, attach to the
already-wired mesh. Target: **5 seconds from manifest to serving**.

## Architecture

```
node boot:
  weightd starts
    → opens RDMA device, allocates spine/expert pool
    → creates QPs for all peer routes (tree topology: ~7 routes × 4 sessions)
    → publishes QPNs to hub (HTTP POST, once)
    → awaits peer QPNs (HTTP GET, 15s budget — happens once)
    → transitions QPs to RTS
    → mesh ready flag set

deploy cycle:
  agent detects manifest change (1s)
  agent kills old residentd (<1s)
  agent starts new residentd (<1s)
  residentd → weightd lazy attach (~100ms warm, no SHA, no spine load)
    → gets: spine pointer + expert pool + mesh-ready + QP proxy
  residentd loads module (~1s, no session opens)
  residentd reports ready
  total: ~5s
```

## The QP sharing problem

RDMA QPs are per-process kernel objects — the residentd cannot use weightd's
`ibv_qp*` pointers directly. Three options:

1. **Work request proxy**: residentd sends "post this RDMA write" to weightd
   via the existing unix socket (~2µs IPC). The GPU data path is unchanged
   (GPU writes to mapped memory that the NIC reads). Only the work submission
   goes through weightd. Net cost: ~2µs per collective op × ~91 ops/token =
   ~180µs/token — negligible against the current 1.5s/token.

2. **Shared completion queue**: weightd creates the QPs AND the completion
   queues, exposes them via `ibv_get_cq_event_fd()`. The residentd polls the
   same CQ (fd passing through the unix socket). The residentd posts work
   directly (no IPC per op), but QP state transitions still need weightd
   (they're done once at wiring time, not per-op).

3. **Separate QPs, shared memory keys**: weightd registers the memory regions
   (getting stable rkeys), but each residentd creates its own QPs. Records
   still change per process, but the rkeys are stable and the RDMA paths are
   pre-warmed (ARP caches, GID resolution, route caches). This is the
   incremental step — doesn't eliminate record exchange but makes it faster.

## Implementation order

1. **weightd mesh init at daemon start** (create + wire QPs, publish records)
2. **Lazy attach returns mesh-ready + QP info** (extend the existing protocol)
3. **Residentd transport proxy** (option 1: work requests through weightd)
4. **Deploy cycle = attach + module load** (no session opens, no record exchange)

## What this eliminates

- Record exchange on every deploy (the 10-15s floor)
- Session open timeouts (QPs are wired once at boot)
- The HTTP fetch storm during boot (no per-process records to fetch)
- Route maintainer in the residentd (weightd maintains the mesh)
- The "fake ready" problem (mesh-ready means actually ready)
