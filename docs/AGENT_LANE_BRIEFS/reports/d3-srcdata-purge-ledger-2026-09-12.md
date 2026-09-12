# D-3 srcdata purge ledger — 2026-09-12

Operator's storage-place law: internal NVMe holds only what inference needs;
source checkpoints and shard staging live on warm/cold ceph; staging may sit
on internal NVMe only while shards are actively generated, and must be purged
once generated + copied to all sparks + validated.

Targets: spark1:/home/spark1/srcdata (66G), spark2:/home/spark2/srcdata (69G),
spark3:/home/spark3/srcdata (747G). All deletions below are logged here before
execution, with pre-deletion manifests written to each node's
/tmp/d3-purge-manifest-20260912.txt.

## Inventory before purge

spark3:/home/spark3/srcdata
- kimi_k3.mxfp4.pp13 — 474710 MiB — rank-3 PP13 layer-stage partition of
  Kimi-K3 (52 files incl. shards 1-30 + 94-96 + tokenizer), generated
  2026-08-09/15 by the ds4 layer-stage tooling (manifest:
  .ds4_layer_stage_manifest.json, rank 3 of 13). Zero writes since Aug 15.
- dsv4_pro.fp8.pp13 — 83066 MiB — rank-3 PP13 stage partition of
  DeepSeek-V4-Pro (shards 1,17-21,63,64), generated 2026-08-09. Zero writes
  since Aug 10.
- .glm52.fp8.pp13.inputs + glm52.fp8.pp13 — 86370 MiB unique (hardlink pair;
  all files share inodes) — glm_moe_dsa fp8 staging + inputs, Jun 17-Jul 5.
- dsv4_flash.fp8.pp13 — 10231 MiB — stage-source v2 extract of
  DeepSeek-V4-Flash-0731 shards 11-13, Aug 6.
- dsv4-tp4-correct-f8c8f001-rank03.spstage — 41584841708 bytes, Aug 13 —
  orphan rank-3-of-4 stagepack; no sibling ranks anywhere in the fleet, no
  repo reference, "f8c8f001" matches no commit.
- releases/ — empty.
- sparkqueue/ — 70G of lane-synced job checkouts (id/sha layout).

spark1:/home/spark1/srcdata — sparkqueue only, 66390 MiB across 26 job dirs.
spark2:/home/spark2/srcdata — sparkqueue only, 69748 MiB across 25 job dirs.

Queue state at inventory: 3 queued jobs (all cwd=$HOME, none touch srcdata),
2 invalid, zero running; all 43 sparkqueue-*.service units on the three nodes
terminal (failed). srcdata job ids appear in zero results.jsonl records —
they are lane-synced checkouts, not queue-dispatched jobs. Activity gate:
files written since 2026-09-10 12:00 exist only in spark2 gate-d1-sm121a,
spark3 glm53full-s4 and glm53full-maincheck.

## Deletion ledger (purge)

spark3 model-shard staging — 654378 MiB:
- kimi_k3.mxfp4.pp13, 474710 MiB — warm /mnt/model-warm/kimi-k3 holds the
  complete 96-shard source (1560998895805 bytes) INCLUDING the tokenizer
  files warm lacked on Aug 30 (tiktoken.model, tokenization_kimi.py,
  tokenizer_config.json verified present); k3 fleet deploy receipted
  (k3-finish-2026-08-30); k3 queue jobs invalid; no open handles.
- dsv4_pro.fp8.pp13, 83066 MiB — full source intact at
  /mnt/model-warm/deepseek-v4-pro-0813-ga (66 shards); superseded deployed
  packs exist fleet-wide (~/sparkdata/dsv4_pro.tp16/packs/rankN.spstage,
  dsv4_pro.tp4pp4); pack backup refresh 16/16 receipted (coordinator firing
  262, 2026-09-06).
- dsv4_flash.fp8.pp13, 10231 MiB — all three staged shards verified present
  in /mnt/model-warm/deepseek-v4-flash-0731 (48 shards); deployed
  dsv4flash.tp16 packs fleet-wide; dsv4flash-tp16-packs report 2026-09-01.
- .glm52.fp8.pp13.inputs + glm52.fp8.pp13, 86371 MiB unique — built packs
  exist at spark1:~/sparkdata/.glm52.fp8.pp13.release-source/; glm52 lane
  archived (docs/archive/GLM52_LAN_API_GATEWAY.md); glm52-smoke and
  glm52-validator-fix results exit 0. Warm holds no glm-5.2 copy — the
  partial staging is re-fetchable from the upstream HF repo if the lane
  revives; recorded as accepted data loss per the storage-place law.
- sparkqueue stale job dirs, ~70300 MiB: glm-paired-pack-main, glm-pack-main,
  glm-pipeline-main, glm-distributed-main (7 sha-checkouts of a Sep 9 sweep),
  glm-beta/conv/eos/hc/hc-round/measure/operands/perf/reduction/rms/serving/
  tree/verify-main, tp4-native-main, tp-chains-main, tp-credit-292720b,
  tp-credit-40d9ca9, tp-offset-e2841c1, tp-fanout-main,
  parallel-debug-20260908 (lane doc committed). All stale since Sep 9-10;
  their lanes' PRs/reports are merged or committed.

spark1 — 65941 MiB: same stale job-dir set (glm-paired-pack-main 20975,
glm-pack-main 20889, glm-pipeline-main 17837, glm-distributed-main 1906,
plus the ~180-280M stale dirs), minus preserved dirs below.

spark2 — 69076 MiB: glm-paired-pack-main 41676, glm-recovery-main 20894,
glm-pack-main 669, glm-distributed-main 1906, plus the stale ~180-280M dirs,
minus preserved dirs below.

## Preserved inventory

- spark2 gate-d1-sm121a, 438 MiB — D-1 lane, written today 17:58. ACTIVE.
- spark3 glm53full-s4 (879 MiB) + glm53full-maincheck (190 MiB) — glm53full
  lane, written Sep 11 04:4x. ACTIVE within 36h.
- spark1 k3-lane-d-m907-s1, 215 MiB — last write Sep 10 09:02; owner unknown;
  misses the 48h gate by hours. Needs owner ruling before deletion.
- spark1/2/3 qwen-flash-cuda5c, 234 MiB each — last write Sep 10 09:11 on all
  three nodes (multi-node lane); owner unknown. Needs owner ruling.
- spark3 dsv4-tp4-correct-f8c8f001-rank03.spstage, 41584841708 bytes —
  UNKNOWN: no fleet sibling (rank00-02 absent everywhere), no repo reference,
  no matching commit. Cannot establish a validated destination, so it stays
  until an owner rules.
- spark3 releases/ — empty; kept as a path convention.
- srcdata/sparkqueue roots on all three nodes — queue sync infra, required
  by tools/spark_queue.py sync (cwd layout srcdata/sparkqueue/<id>/<sha>).

## Execution gates (per target, checked immediately before rm)

1. lsof +D over the node's srcdata tree returns empty (no open handles —
   serving, weightd, or any process).
2. lsattr shows no 'i' (immutable) flag on any target top-level dir or on the
   large files inside the spark3 model-staging trees.
3. systemd sparkqueue units remain all-terminal; queued jobs still cwd=$HOME.
4. Nothing under spark3 serving paths or the weightd service is touched;
   purge is filesystem-only inside /home/spark3/srcdata.

Expected freed: spark3 ~724700 MiB, spark1 65941 MiB, spark2 69076 MiB;
~840 GiB total against the 2TB target.

## Execution record (2026-09-12, post-ledger commit 741f927)

All gates passed on all three nodes before each rm batch: zero running
sparkqueue units, zero open handles under srcdata (lsof +D, run from $HOME —
two initial aborts were the gate detecting its own session cwd, fixed by not
cd-ing into the tree; the misfiring immutable check matched the letter "i" in
path names ending -main, fixed by testing only the lsattr flags field), zero
immutable flags at directory and file level (lsattr -R over the spark3 model
trees and the large job dirs). Pre-deletion manifests on each node at
/tmp/d3-purge-manifest-20260912.txt.

Freed (df -m on /, before -> after):
- spark1: 1807493 -> 1741119 MiB used = 66374 MiB freed; avail 1841239 ->
  1907614 (50% -> 48%). Preserved: k3-lane-d-m907-s1, qwen-flash-cuda5c.
- spark2: 2761366 -> 2691632 MiB used = 69734 MiB freed; avail 887367 ->
  957100 (76% -> 74%). Preserved: gate-d1-sm121a, qwen-flash-cuda5c.
- spark3: 3171808 -> 2447995 MiB used = 723813 MiB freed; avail 476924 ->
  1200737 (87% -> 68%). Preserved: dsv4-tp4-correct-f8c8f001-rank03.spstage,
  releases/, sparkqueue/{glm53full-s4, glm53full-maincheck, qwen-flash-cuda5c}.
- Total freed: 859921 MiB (839.8 GiB) against the 2TB fleet target.

spark3 serving verified alive and untouched post-purge: sparkpipe_weightd
(/tmp/spark_weightd.sock) and sparkpipe_model_residentd running; lsof proved
neither held any srcdata handle pre-deletion.
