# INFRA ESCALATION (2026-08-28 09:05): spark3 reboot-crash-loop — SYSADMIN NEEDED

spark3 rebooted twice inside one hour (~08:13 and ~08:59, "up 3 min" at
09:02; third reboot in ~24h). Yesterday's GPU wedge showed arm-smmu-v3
CMD_SYNC timeout storms and an NVRM full-chip-reset assertion — the
recurring reboots fit that same hardware/driver instability signature.
Two asks for the sysadmin:
1. Investigate spark3 (hardware/driver — smmu/NVRM; crash logs will be
   in the journal around 08:1x and 08:5x).
2. MOVE `mds.ds4warm.spark3` to a stable host (sparkc runs osd.12 and
   has been rock solid). Fleet storage health is currently coupled to
   the least stable node: every spark3 reboot restarts the model-warm
   metadata server cold → transient ENOENT blips + slow uncached reads
   fleet-wide for minutes (NOT a new ceph wedge — do not diagnose it as
   one; wait 10-15 min for MDS warm-up and retest).
Until cleared: spark3 is UNSTABLE — the knee-sweep lane has been told
to checkpoint every measurement and stop if it reboots twice more.

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
