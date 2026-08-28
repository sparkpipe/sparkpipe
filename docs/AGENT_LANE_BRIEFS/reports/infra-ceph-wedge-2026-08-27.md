# INFRA ESCALATION 2 (2026-08-28 ~10:3x): sparke FULLY DOWN — osd.14 with it. SYSADMIN NEEDED

sparke: no ping (100% loss from the controller), ssh timeout from two
vantage points (controller + spark0). Node power/console/fabric check
needed. It runs **osd.14** — degraded PGs until it returns (replication
serves; expect recovery churn). Second node-down incident today after
spark3's reboot pair — worth checking for a common cause (power feed,
fabric switch, thermal).

FULL CEPH TOPOLOGY (inventoried 10:3x — corrects the earlier partial map):
- mons: spark0, spark7, sparkf (quorum alive)
- MDS PAIR: mds.ds4warm.spark2 + mds.ds4warm.spark3 (both up; spark3's
  reboots have been bouncing one half — the transient ENOENT blips are
  consistent with MDS pair failover, seen again at 10:3x on spark6)
- OSDs: one per node (osd.0 spark0, osd.1 spark1, osd.2+16 spark2,
  osd.3+17 spark3, osd.4 spark4, osd.5 spark5, osd.6+18 spark6,
  osd.7 spark7, osd.8 spark8, osd.9 spark9, osd.10 sparka, osd.11 sparkb,
  osd.12 sparkc, osd.13 sparkd, osd.14 SPARKE (DOWN), osd.15 sparkf)

REVISED storage-risk rule (replaces the two-host version): EVERY node is
a storage node. Strict no-GPU-work applies to ACTIVE-MDS hosts (spark2,
spark3 while they hold the mds.ds4warm pair). All other nodes run GPU
work under a MEMORY ENVELOPE (~85-90 GiB device ceiling; the spark3 crash
was ~104 GiB next to mds+2 OSDs) so a GPU job cannot OOM the local OSD.
Better long-term: sysadmin adds systemd MemoryMin to ceph units on ALL
nodes, then the envelope relaxes.

spark3 rebooted twice inside one hour (~08:13 and ~08:59). UPDATED ROOT
CAUSE (from the knee-sweep lane): reboot #2 coincided exactly with a
residentd launch at resident_sequence_capacity=128 / max_sequence_positions=8192
— a ~104+ GiB unified-memory KV pool on the SAME node as mds.ds4warm +
osd.3 + osd.17. GPU unified-memory pressure can starve/OOM the ceph
daemons → node crash + fleet-wide storage impact. Reboot #1 (~08:13)
predates all GPU work that morning — unexplained, possibly residual from
yesterday's NVRM/smmu instability. This is a NOISY-NEIGHBOR design
conflict, not (necessarily) flaky hardware: production storage roles and
GPU jobs share one 119 GB unified memory.

Asks for the sysadmin:
1. Confirm in the journal (~08:5x): OOM kill of ceph daemons / kernel
   panic signature. And check the 08:1x reboot's cause.
2. MOVE `mds.ds4warm.spark3` to a host without GPU lanes (sparkc runs
   osd.12 and has been rock solid). Fleet storage health is coupled to
   nodes we GPU-stress; every spark3 reboot restarts the model-warm MDS
   cold → transient ENOENT + slow uncached reads fleet-wide (NOT a new
   ceph wedge; wait 10-15 min for MDS warm-up and retest).
3. If storage roles stay on GPU nodes: systemd memory protections
   (MemoryMin/MemoryMax) for the ceph units so a GPU job cannot OOM
   them.

Lane policy effective now (in the README): STORAGE-HOST nodes (spark3:
mds+osd.3+osd.17; sparkc: osd.12) host NO GPU lane work — the queue will
not dispatch GPU runs there; agents do not reserve them. The knee-sweep
lane is granted a ONE-TIME exception on spark3 at the VERIFIED 71.1 GiB
envelope only (its 104 GiB config is dropped), one-B-per-session with
checkpoints; it stops if the node reboots twice more.

## ADDENDUM 09:4x — object-specific stalls = recovery aftermath (sysadmin: `ceph -s`)

Ceph is NOT globally slow anymore (shard 60: 145 MB/s direct on sparke),
but kimi-k3 shard 12 has hung a builder in D-state (folio_wait_bit_common)
for 25+ min — single-object stall with healthy neighbors. This is the
recovery-backfill signature: spark3's reboots took osd.3+osd.17 down, so
PGs are degraded/recovering, and objects on recovering PGs stall until
backfill completes. Ask: `ceph -s` (degraded PGs, recovery progress/ETA);
expect self-heal. No cold bypass exists (staged cold kimi-k3 = 6/96
shards, shard 12 absent). The K3 lane TERMs the builder at its journal
and probe-then-resumes shard 12 when it serves.

---

# INFRA: ceph /mnt/model-warm wedge — RESOLVED 2026-08-28 (root cause: spark3, not ceph)

RESOLUTION + CORRECTION: ceph itself was never wedged. spark3 hosts
`mds.ds4warm.spark3` (the cephfs METADATA SERVER for model-warm) plus
osd.3/osd.17. When spark3 wedged (GPU incident below), the MDS went with
it: every client's UNCACHED read stalled on metadata retries (cached
pages needed no MDS and flew at RAM speed) — which masqueraded as an
"OSD-path" wedge fleet-wide. spark3's reboot restored everything;
verified uncached direct reads: sparke 336 MB/s, sparkd 272 MB/s.
Corollary for future incidents: spark3 (or whichever host runs the
model-warm MDS) being unresponsive = fleet-wide slow uncached reads.
Check the MDS host's health BEFORE diagnosing ceph.

The advisory below is retained for the pattern and the recovery test.

---

# INFRA: ceph /mnt/model-warm wedged on OSD path — fleet lane advisory (2026-08-27)

Status: ACTIVE incident, sysadmin escalated. Uncached reads from
/mnt/model-warm crawl at ≤17 MB/s fleet-wide (normal 663 MB/s). Cached
(page-cache) reads still serve at RAM speed — a fast small read does NOT
mean recovery. Evidence: 256MB `dd iflag=direct` on kimi-k3 shard 80
timed out at 15s on sparke; K3 lane saw a 30+ min D-state stage build
and a 20s-incomplete 256MB read on sparkd AND sparke (multi-client =
not a per-node mount wedge). osd.12 on sparkc is healthy (S-state,
load 1.07) — fault likely elsewhere in the cluster (mon quorum, PG
backfill, another OSD host).

## What your lane should do right now
- DEFER all model-warm content hashing / digest verification / sha256
  sweeps. Pin small files + names + sizes if you need a provisional
  manifest; mark digests deferred.
- Builds reading the warm source: pause at the next journal point (TERM,
  never KILL — D-state processes exit when the I/O unwedges; do not
  stack retries). Your journals resume safely.
- Code work (kernels, headers, packers on local NVMe, synthesized
  packs, tests) is UNAFFECTED — do that.
- Anything already staged to local NVMe (K3 stage-3 rank packs, prior
  stage outputs) reads at full speed.

## Recovery test (run on any node, ~15s)
    timeout 15 dd if=/mnt/model-warm/kimi-k3/model-00080-of-000096.safetensors \
        of=/dev/null bs=4M count=64 iflag=direct; echo rc=$?
rc=124 or a MB/s figure in the double digits = still wedged.
A completion at hundreds of MB/s = recovered; resume ceph work and note
it in your next report. The coordinator will also message lanes directly
when confirmed.
