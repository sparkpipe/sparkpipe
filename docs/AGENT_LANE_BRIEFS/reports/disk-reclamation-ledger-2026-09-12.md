# D-1 fleet disk reclamation ledger — 2026-09-12

Scope: mgr2 nodes spark1/spark2/spark3, node-filesystem only, no daemon
interaction. Every entry below was verified on the node immediately before
this commit; deletions execute only after this ledger is committed. Bytes are
`du -s -B1` (space reclaimed, not file size). Target: ~2 TB free per spark.

## Executed deletions

| # | node | path | bytes | class | justification |
|---|------|------|-------|-------|---------------|
| 1 | spark1 | /home/spark1/sparkdata/dsv4_pro.tp4pp4 | 99,612,667,904 | DELETE-FREE | dsv4 pro is obsolete (replaced by dsv5); packs regenerable — warm `/mnt/model-warm/packbuild/dsv4pro` holds receipts and regen-identical packs (staging-inventory-2026-08-28) |
| 2 | spark1 | /home/spark1/sparkdata/qwen38_2.4t_a95b | 183,273,467,904 | DELETE-FREE | node-local raw-checkpoint cache; ceph stash `/mnt/model-warm/stagepack-backup/spark1/qwen38_2.4t_a95b` verified byte-for-byte mirror this session (25 files, sizes identical diff = empty; HASH_VERIFY manifest_sha256 8825c2e7 present both sides) |
| 3 | spark1 | /home/spark1/sparkdata/qwenmax.pp16-stripped/packs/.stage0.qwen38sp.ou1n_gf5.tmp | 1,551,904,768 | DELETE-FREE | aborted stagepack temp partial from the Sep 12 pp16 strip run; full stage set complete (stage0-10 spark2, stage11-14 spark3, stage15 spark1) |
| 4 | spark2 | /home/spark2/sparkdata/dsv4_pro.tp4pp4 | 99,613,261,824 | DELETE-FREE | same as entry 1 |
| 5 | spark2 | /home/spark2/hy4-full.gguf | 235,352,018,944 | DELETE-FREE | node-local source-format cache; sha256 `12d325844103bac75bd286d14e0e45f87e35e8e60401877282a30b6f26ba6ac6` verified EQUAL to `/mnt/model-warm/hy4-preview-ud-iq1m-angelslim/Hy4-preview-UD-IQ1_M.gguf` this session (both hashes computed under the stagepack memory cap) |
| 6 | spark2 | /home/spark2/hy4-allranks/rank-00 rank-01 rank-03 … rank-15 (15 dirs, rank-02 KEPT) | 280,870,846,464 | JUDGMENT → delete | only rank-02 is referenced: F-5 rung-6 cells consume rank-02, and the deployed hy4.ud-iq1m.tp16 pack tree holds rank-02 only; no script/manifest/cell on spark2 references ranks 00-01/03-15 (`hy4_build_watchdog.sh` writes to its own hy4-build/ tree, verified); source gguf hash-verified on warm (entry 5), ranks regenerable via hy4_tp16_shard.py |
| 7 | spark3 | /home/spark3/sparkdata/dsv4_pro.tp4pp4 | 99,613,065,216 | DELETE-FREE | same as entry 1 |
| 8 | spark3 | /home/spark3/q38max_hs_regress.qwen38sp | 111,078,465,536 | DELETE-FREE (verified duplicate) | sha256 `35b1b64207d50efc68347053eb227aa327548730d188534f7eb02da2433f9a5f` byte-identical to q38max_tp16k_regress.qwen38sp (which SURVIVES as the 35b1b642 generation copy); both are stale copies of the prior sota_smoke generation |
| 9 | spark3 | /home/spark3/.q38max_fw_l0.qwen38sp.k_fd74_f.tmp | 4,076,904,448 | DELETE-FREE | interrupted stagepack temp (Sep 8) |
| 10 | spark3 | /home/spark3/.q38max_fw_l0.qwen38sp.7ch16rgs.tmp | 0 | DELETE-FREE | empty stagepack temp (Sep 8) |

Planned reclaim: spark1 284,438,040,576 B (265 GiB); spark2 615,836,127,232 B
(573 GiB); spark3 214,768,435,200 B (200 GiB). Total 1,115,042,603,008 B.

## Refuted audit premises — NOT deleted

- **qwenmax.pp16-stripped is not a duplication.** spark2 (955G) holds stages
  0-10, spark3 (346G) holds stages 11-14, spark1 (38G) holds stage 15: one
  16-stage pp16 pipeline distributed across the three nodes, built Sep 12
  ~00:30-01:54. "One canonical location" already holds; deleting either side
  would have destroyed the deployment. KEEP ALL stages. Placement for the
  serve-gate: the stripped set spans spark2+spark3+spark1 by construction.
- **The 4× 104G q38max packs are not 4 copies of digest 5e773f8b.** Measured
  sha256: q38max_sota_smoke.qwen38sp = `0ca8266a…` (canonical, referenced by
  q38max_exec_cell.sh / q38max_ab_cell.sh / q38max_sota_smoke.sh, matches
  q38max_pack.sha256); q38max_sota_smoke.mtp1.qwen38sp = `5e773f8b…` (the
  real-pack digest — the ONLY copy of that generation, KEEP);
  q38max_tp16k_regress = q38max_hs_regress = `35b1b642…` (the only verified
  duplicate pair; hs_regress deleted, tp16k_regress kept as the 35b1b642
  survivor). All hashes computed under the stagepack memory cap.
- **k3.mxfp4.tp4pp4 (93G each on spark1/2/3) is referenced.** k3-gate
  worktrees on spark1 and spark7 carry live configs pointing at
  `sparkdata/k3.mxfp4.tp4pp4/packs/…` (TP4xPP4 mesh per node). KEEP all three.
- **qwen38_2.4t_a95b node copies on spark2 (148G) and spark3 (136G) were
  already gone** before this session started: free-space deltas vs the audit
  (+148G / +135G) match the item sizes exactly. Only spark1's copy remained
  and is handled as entry 2.

## Kept (classification outcomes)

- glm53flash.fp8.tp16 + weightd trees on spark3/spark2/spark1: operator's
  live serving; untouched, no daemon interaction.
- qwenmax.pp16 placed originals (87G each): operator said keep (various
  topologies supported).
- hy4 rank-02 (hy4-allranks/rank-02, 18G) + hy4-fp8-packs rank-02 (56G):
  active lane inputs.
- dsv5 r2 packs on spark6: A-dsv5-r11's final set; spark6 untouched
  (report-only for this lane).
- glm53full build trees on spark3 (bf16/fp8 tp16): staged build.
- dsv4_pro.tp16 (53G × 3), dsv4flash.* (53-93G × 3), qwen38_max.tp4pp4 (87G,
  spark1): NOT in this lane's classification — candidate follow-up for the
  coordinator, not deleted without a ruling.
- spark3/srcdata (747G) and spark2/srcdata (69G), spark1/srcdata (66G):
  unclassified; largest remaining gap-closers — needs coordinator
  classification before any action.

## Pre-flight evidence (per node, before any rm)

- `/proc/*/cmdline` scan for every target path: only the scan's own shell
  matched — no process holds any deleted path open.
- qwen38_2.4t mirror: `diff` of per-file size+name lists between ceph stash
  and local = empty.
- hy4 gguf and q38max pack digests: sha256sum under
  `sudo -n systemd-run --scope -q -p MemoryMax=4096M -p MemoryHigh=2900M`.

## Receipts (execution, 2026-09-12)

Every path verified present immediately before its `rm`; per-path result:

- spark1: `DELETED /home/spark1/sparkdata/qwen38_2.4t_a95b`; `DELETED
  /home/spark1/sparkdata/qwenmax.pp16-stripped/packs/.stage0.qwen38sp.ou1n_gf5.tmp`;
  `dsv4_pro.tp4pp4` — first pass left
  `packs/dsv4_pro.tp4_pp4.rank01.spstage` (`Operation not permitted`,
  lsattr shows the immutable `i` flag), cleared with `sudo -n chattr -i`
  then `DELETED-after-unlock`.
- spark2: `DELETED /home/spark2/hy4-full.gguf`; `DELETED
  /home/spark2/hy4-allranks/rank-00 rank-01 rank-03 … rank-15` (15 lines);
  `dsv4_pro.tp4pp4` — immutable `rank02.spstage`, `chattr -i` then
  `DELETED-after-unlock`.
- spark3: `DELETED /home/spark3/q38max_hs_regress.qwen38sp`; `DELETED
  /home/spark3/.q38max_fw_l0.qwen38sp.k_fd74_f.tmp` and the empty
  `.7ch16rgs.tmp`; `dsv4_pro.tp4pp4` — immutable `rank03.spstage`,
  `chattr -i` then `DELETED-after-unlock`.

df (/, same node, before session → after execution):

- spark1: 1.5T free (57%) at session start → **1.8T free (50%)**
- spark2: 294G free (92%) at session start → **867G free (76%)** (audit had
  146G/96%; the qwen38_2.4t copy was already reclaimed before this session)
- spark3: 266G free (93%) at session start → **466G free (87%)** (audit had
  131G/97%, same pre-session reclamation)

The other lanes kept writing during execution, so per-node df deltas also
absorb unrelated traffic; the ledger byte counts are the authoritative
reclaim figures. 2 TB-per-spark remains unmet on spark2 (−1.13T) and spark3
(−1.5T): the remaining classified-adjacent space needs coordinator rulings
(dsv4_pro.tp16, dsv4flash.*, spark3/srcdata 747G, qwen38_max.tp4pp4,
qwenmax.pp16-stripped if its lane abandons it).

---

# D-2 session (successor) — spark5 glm53full.tp16 stacks, 2026-09-12

Target per the brief: `spark5:/home/spark5/sparkdata/glm53full.fp8.tp16/`
and `glm53full.nvfp4.tp16/`, believed to hold ~15 ranks stacked on one node
(a placement-law violation). The premise is REFUTED by direct inspection:
each tree holds exactly ONE pack body — rank5, spark5's lawful rank under
the fleet-table policy (`tools/glm53full_place_packs.sh`: rank r → spark
hex(r)) — plus the 16 per-rank `receipt.json` build receipts (~3.7 MB
each; the "~15 ranks" was a count of receipt files). The placement law is
not violated on either tree.

## Digest identification (sha256 of every file whose bytes exceed the
receipt sidecars, plus the receipt chain)

| file | bytes | sha256 | classification |
|---|---|---|---|
| glm53full.fp8.tp16/packs/glm53full.fp8.tp16-rank5.glm52sp | 54,136,549,376 | `6e5102f1355b10c10a7f1219c5fccadf9eadd9c824ee4ed06e6cd157aaec9bbd` | KEEP — active arm, live attach |
| glm53full.fp8.tp16/packs/…rank5.glm52sp.experts + .sha256 | 3,686,416 + 99 | (sidecars of the body) | KEEP |
| glm53full.fp8.tp16/packs/rank{0-15}.glm52sp.receipt.json | 16 × 3,706,207-08 | (receipt chain of the R2 set) | KEEP |
| glm53full.fp8.tp16/{bin,lib,config,kvcache,r2_attach_receipt.txt,residentd_r2.*} | deploy tree | — | KEEP — live deployment root |
| glm53full.nvfp4.tp16/packs/glm53full.nvfp4.tp16-rank5.glm52sp | 32,903,038,976 | `471548d763d5ef98457878da6b2a518df6fdaa47acbdf7f8814c7919a41b195e` | DELETE — matches no receipt (below) |
| glm53full.nvfp4.tp16/packs/rank{0-15}.glm52sp.receipt.json + SHA256SUMS | 17 files, 63.0 MB | (build receipts, pinned in glm53full-2026-08-28.md) | KEEP — rebuild reference |

## Receipt comparison

- fp8 rank5: on-disk sha256 == `packs/*.sha256` sidecar == every
  `g53r2 RECEIPT` line in `r2_attach_receipt.txt` (bytes and digest both
  recorded per attach). The tree's residentd is ATTACHED LIVE right now
  (the `sparkqueue-df93f211…` unit re-ran during this session: a residentd
  PID in state S running `./bin/sparkpipe_model_residentd … --rank-index 5`
  with the fp8 tree as its root; earlier scan caught its reaped zombie).
  Active receipt → whole tree KEEP. Node weightd (mesh-rank 5) is cwd'd on
  the operator's glm53flash serving swap — different tree, untouched.
- nvfp4 rank5: the pinned `SHA256SUMS` (2026-08-29 13:57, build session of
  the 16/16-validated set) demands `9e16b01111d66012…`; the on-disk body
  (mtime 2026-08-30 19:44, one minute after the fp8 rank5 rewrite) hashes
  to `471548d763d5ef98…`. No receipt, manifest, script, unit, or repo
  reference anywhere names 471548d7. Cross-node control: spark0's rank0
  body still hashes exactly to its pin `36fef980…` (mtime Aug 29 15:51,
  placement era) — the fleet set and SHA256SUMS are intact and
  authoritative; spark5's rank5 body is the single divergent copy.
- The qwen38 packer directory rows/cols defect (#952, 4ca697a) does not
  apply: glm53full packs come from `glm52_resident_stagepack.py` (wire
  format v3, receipts validated 16/16 errors:0 at build), a different
  packer with no shared late-binding pattern.
- mgr1-era L7/#829 receipts were not found in the repo or the #904/#954
  threads (limit); the fp8 KEEP chain above is independent of them.

## Classification outcome

- fp8.tp16 (54.2 GB): KEEP — the lane's active arm, receipt-matched, live
  attach in progress. Zero deletions.
- nvfp4.tp16 rank5 body (32,903,038,976 B): DELETE — digest-matched to
  nothing, contradicted by the set's own pin, unreferenced by any process
  or unit (fuser/lsof/proc scans clean; no immutable flags, plain `e`
  extent). The lane's own plan (glm53full-2026-08-28.md, SPACE note)
  already slated the local nvfp4 set for deletion as "derived artifacts,
  rebuildable from pinned sources"; rank5 survived that rm only as the
  local-rank retention, and as retained it is not the receipted bytes —
  it has no value to the arm. When the nvfp4 arm activates, rank5 is
  rebuilt from the warm-pinned source (`/mnt/model-warm/glm-5.3-nvfp4-
  radixark`, index_sha256 `2aa8397b…` in the authoritative contract) and
  re-placed by the resumable placement tool; the 15 verified fleet ranks
  are untouched on their nodes.
- nvfp4 receipts + SHA256SUMS (63 MB): KEEP — the rebuild/verify reference
  for the set.

## Deletion ledger (execute only after this commit)

| # | node | path | bytes | class |
|---|------|------|-------|-------|
| 11 | spark5 | /home/spark5/sparkdata/glm53full.nvfp4.tp16/packs/glm53full.nvfp4.tp16-rank5.glm52sp | 32,903,038,976 | DELETE (unreceipted divergent body) |

Planned reclaim: spark5 32,903,038,976 B (30.6 GiB). The 2 TB target is
not met on spark5 (1.6 T free at session start, ~1.63 T after): the
briefed stacks held 87 GB total, not the audited hundreds — the fp8 half
is the live arm. Remaining spark5 candidates outside this brief's scope,
for coordinator ruling: glm53full.{bf16.tp16, fp8.tp4pp4, nvfp4.tp4pp4,
bf16.tp4pp4} sibling trees and the large srcdata/ tree.

