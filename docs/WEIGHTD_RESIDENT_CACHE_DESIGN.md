# weightd: resident multi-model weight cache (design of record, 2026-09-01)

Operator direction, after the wave-collision incident: **weightd stays
resident all the time** — it owns eviction and swap-in itself, so a
model switch or fleet wave never again destroys memory residency.

## Today (single-shot seam)

weightd is spawned per residentd startup, holds ONE pack resident
(sha256 sidecar keyed), and dies with the serving units. Any fleet
switch = full cold reload (~21.7 GB/rank from NVMe, minutes). The
incident: a p0 wave task ran ~3 minutes before cancel and killed
weightd fleet-wide; the restore paid the full cold reload for nothing.

## Target (the operator's model)

weightd is a **persistent per-node service** — started once (system
unit), never owned by any model's lifecycle:

1. **Multi-model registry**: several packs resident concurrently, keyed
   by pack sha256. The SparkStageModuleLoadDeviceRegion seam already
   addresses by sidecar digest — the attach path generalizes from
   "the pack" to "a pack".
2. **Memory budget + eviction**: a node budget (default: leave headroom
   for KV + activations). Admission of a new pack evicts by policy —
   LRU among non-pinned packs, PARTIAL eviction first (evict the
   minimum set that frees the needed bytes; large packs can be
   fractionally resident: the seam's region-based attach makes a
   partial pack a hit-or-miss per region, minimizing thrash when two
   big models nearly fit).
3. **Pinning**: the actively-serving model's pack is pinned (unevictable
   while its serving task holds the nodes). Pin/unpin rides the queue:
   the serving task's dispatch/unload = pin/unpin events.
4. **Swap-in on scheduled runtime**: the queue knows what gets runtime
   next (a queued all-16 wave for model X is a pre-stage hint). On
   dispatch of X's build/wave task, weightd begins warming X's pack
   (NVMe read at idle priority) so the switch costs compute, not load.
   Concretely: the dispatcher emits a `warm <pack-path>` command per
   node as a side effect of task dispatch (queue change, small).
5. **Contract safety**: the fallback path stays exactly as-is — no
   daemon, refusal, digest mismatch, or partial hit on a REQUIRED
   region = direct load (correctness never depends on the cache).

## What this buys

- Fleet window switches: seconds (compute) instead of minutes (cold
  reload) when the target was pre-staged or partially resident.
- The experimental loop keeps its one-minute debug cycle across ANY
  fleet churn — the residency survives serving restarts and waves.
- Automatic multi-model coexistence when memory allows (small dev
  models + the serving model together).

## Implementation slices (each independently landable)

- S1: detach weightd from residentd — a persistent user unit per node;
  residentd's spawn becomes "ensure running" (idempotent). Fallback
  contract unchanged.
- S2: multi-pack registry + budget accounting + LRU-with-pin eviction.
- S3: partial residency (region-granular hit/miss) — the thrash
  minimizer; needs the seam's region map exposed per pack.
- S4: queue integration — pin on dispatch of the holding task, warm
  hints from queued wave tasks.

Owners: S1-S3 are the weightd lane (runtime/spark_weightd.c, the seam
in runtime/stage_module_common.c); S4 is the coordinator (queue).
