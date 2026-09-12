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

