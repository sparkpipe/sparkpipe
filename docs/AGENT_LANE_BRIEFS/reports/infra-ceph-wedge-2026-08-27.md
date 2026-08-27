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
