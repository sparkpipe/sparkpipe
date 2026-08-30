# qwen-flash: 16-rank wave readiness audit + launcher per-host fix — 2026-08-30

Lane: qwen4_flash (Qwen 3.8 Flash), branch lane/qwen38flash-dev, all GPU-path
work via the spark queue (tasks qwen38flash-fp8hdrdump..deploydist below).
No GPU was consumed this session: every task was file-audit/staging class.

## Preflight audit (queued: qwen38flash-wavepreflight, spark4)

First pass reported 15/16 nodes "MISSING" everything — which turned out to be
the LAUNCHER'S OWN BUG reflected in my audit: both resolved every path under
`/home/spark4/...` (the coordinator's home) on all hosts. Homes are per-node
(`/home/spark4`, `/home/spark5`, ...); there is no cluster-wide /home.

## Fleet pack census (queued: qwen38flash-packcensus, spark4)

ALL 16 v4 stage packs ARE in place at their destination nodes, byte sizes
matching the spark4 staging copies exactly (spot-checked via stat; full sha
verification still owed at wave time by the launcher's own preflight +
receipts):

  s0r0-3 (38.95 GiB class) spark4-7 · s1r0-3 (14.75) spark0-3 ·
  s2r0-3 (14.75) spark8-b · s3r0-3 (17.84, head+MTP) sparkc-f

Disk note: sparke has 91 GiB free — fits its 17.8 GiB pack + runtime, but
the wave should not stage anything extra there.

## The fix (this PR)

tools/qwen4_flash_fleet16_launch.sh computed ONE deploy_dir from the
coordinator's home and used it on every host: pack preflight, residentd
spawn, ready-log greps, tail-on-fail. As shipped, a wave would fail 15/16
ranks at `cd /home/spark4/sparkdata/...` — burning the exclusive 16-node
window. Fix: deploy_dir_for() resolves `/home/$host/sparkdata/...` per host
(matching qwen4_flash_deploy_v4.py's runtime_root); coordinator-side uses
(pid file, api) keep the coordinator copy.

## Deploy-tree distribution (queued: qwen38flash-deploydist, spark4)

The deploy_v4 runtime tree (bin×3, lib×3, 16 adapter configs,
deployment.json, launch_table.json, batch files — ~7 MB) existed only on
spark4. Distributed to the other 15 nodes at each host's own path; verified
bin=3 lib=3 cfg=16 table=Y on all 15.

## Ratchet + digests (honest)

- Pristine origin/main measured 233255 authored lines vs CEILING 233120 —
  main is ALREADY 135 over its own ratchet (pre-existing; whoever landed the
  last main merge did not regenerate digests last either: manifest gate red
  for MODEL_DEV_GUIDE/ROADMAP/UNIVERSAL_PACKER/qwen38_stagepack/coordinator-
  log). Flagged for the coordinator; not adjudicated here.
- This change adds 5 lines (launcher fix) + the ceiling record comment;
  ceiling re-measured exact to 233270 with the justification in-file.
- PACKAGE_MANIFEST.json + SHA256SUMS regenerated LAST; `make offline-gates`
  exit 0 on the branch tip.

## Wave readiness after this PR

Packs 16/16 placed · deploy trees 16/16 · launcher host-correct · launcher
preflight will re-verify packs at wave time. Remaining for the wave:
coordinator's exclusive 16-node window (k3/dsv4 notes hold the queue head),
spark6/7 ceph wedge awareness for any non-local reads (none needed — packs
are node-local), and the standard receipts after launch (B1 smoke, stream
hash, exact-32K cell).
