# Fleet staging inventory + disk-hygiene manifest — 2026-08-28

Lane: staging (worktree /tmp/lane-staging, branch lane/staging-v2). Mission:
every served model's packs on the right nodes per the all-16 policy
(docs/AGENT_LANE_BRIEFS/README.md fleet table); disk pressure eliminated as a
recurring incident class. This report = S1 inventory (gap table) AND the S2
destructive-action manifest, committed BEFORE any deletion is executed (the
queue's denylist ethos: per-node manifest first, anything ambiguous stays).

All claims below = command + output (run from the controller unless noted;
`ssh -o BatchMode=yes sparkN '<cmd>'`). Node-local date at measurement:
Aug 29 01:2x KST (= Aug 28 16:2x UTC).

## S1.1 Disk state (`df -h /`; /home is on the root fs)

```
spark0  3.7T  1.8T  1.7T  52%     spark8  3.6T  460G  3.0T  14%
spark1  3.7T  2.2T  1.3T  63%     spark9  3.6T  762G  2.7T  22%
spark2  3.7T  1.6T  1.9T  46%     sparka  3.6T  509G  3.0T  15%
spark3  3.7T  2.8T  761G  79%     sparkb  3.6T  2.0T  1.4T  59%
spark4  3.6T  943G  2.5T  28%     sparkc  3.6T  528G  2.9T  16%
spark5  3.7T  1.7T  1.8T  49%     sparkd  3.6T  618G  2.8T  18%
spark6  3.6T  2.6T  909G  74%     sparke  3.6T  1.3T  2.2T  38%
spark7  3.6T  1.9T  1.6T  56%     sparkf  3.6T  405G  3.1T  12%
```

spark0's 100%-full incident was already rescued (585G→1.7T free,
coordinator log 21:1x). Current pressure points: spark3 (79%), spark6 (74%),
sparkb (59%), spark1 (63%), spark7 (56%).

## S1.2 Pack gap table (model x node; every cell = measured this session)

Sizes are exact bytes from `stat -c %s`. "verified" = sha256 compared against
a receipt or a recorded digest this session.

| Model (policy set) | State | Evidence |
|---|---|---|
| glm5_next TP16 (all 16) | **16/16 PRESENT, size-exact 21,706,046,976 B each; rank0 full-sha verified** | real packs at `~/glm53_packs/glm5_next_stage.tp16.rank{r}.g5nsp`; runtime root `~/sparkdata/glm5_next.tp16/packs/` symlinks to them (all 16 readlink-checked). rank0 sha256 `ca0e427ce9c33f0f6b536984bfbf4a50ae3498ba814ab7310e5f77aa24e0e6c1` (NEW receipt — the lane report's `4ba9b31f` was the pre-header-patch emission prefix; full-set digest sweep in flight, results appended in §S3) |
| glm52 TP8 (donor band spark8-f ONLY — not a fleet-pack model per README) | **8/8 PRESENT, size-exact 102,835,957,760 B each** | `~/sparkdata/glm52.tp8.fp8/packs/glm52_tp8_rank{0..7}.fp8.glms52sp` on spark8..sparkf (rank r = host index). STRAY outside band: spark1 holds `glm52_tp8_rank04.fp8.glms52sp` (duplicate of sparkc's rank04; sha cross-check in §S2) |
| dsv4_pro TP4xPP4 (all 16) | **16/16 PRESENT; 14 with on-node receipts; spark1/2 sha-verified this session** | spark0..f each hold `~/sparkdata/dsv4_pro.tp4pp4/packs/dsv4_pro_tp4_pp4_stage.spstage`. Receipts on 14 nodes: `validated:true`, shas byte-match the dsv4pro report (spot: spark0 rank0 `490c5cdc...`, spark3 rank3 `1071e9ad...`, spark4 `db58f92a...` ... sparkf `889276e4...`). spark1/spark2 packs are Aug-17 vintage WITHOUT receipts — hashed this session: spark1 `e9de954be40628dcc4f0d2d88fe1b2391b50ba7910ca9a8e5ba249b6a53b9879`, spark2 `acc6d761cb79174abe2ec8e8b53e8bf38ca98dde87a2ad939d3beccd95ad5863` — **exactly equal to the warm stash receipts** `/mnt/model-warm/packbuild/dsv4pro/rank{1,2}.receipt.json` (deterministic packer; Aug-17 bytes == regen bytes, as the report proved for rank3). dsv4_pro is DONE — the only fill owed is the two receipt files. (bin/lib/config on spark1/2 are pre-PR721 vintage — driver-lane scope, NOT staging's.) |
| K3 TP4xPP4 stage-3 (sparkc-f; stages 0-2 still building on sparke — K3 lane's) | **4/4 PRESENT + sha-verified in place** | `~/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage3.rank0{0..3}.pack`, 98,119,908,864 B each: sparkc `23df8aa6...`, sparkd `d9deda6a...`, sparke `b1727bec...`, sparkf `02882b94...` — all equal to `/mnt/model-warm/packbuild/k3/receipts/stage3_warm_sha256.txt` and the K3 report's deploy digests. (The K3 report's "sparkc/d/f copies removed" note is STALE — packs are present and hash-correct.) |
| qwen4_flash TP4 v3 (node's own rank on spark4-7; v3 = the verified set) | **4/4 PRESENT (per-node rank), 60,194,156,288 B each; receipts on spark4 only** | `~/sparkdata/qwen4_flash.tp4/packs_v3/`: spark4=r0(+r1-3 hub copies), spark5=r1, spark6=r2, spark7=r3. `deploy_v3/config/adapter-r{0..3}.json` references `../packs_v3/...` — per-node own-rank is the serving layout. v3 receipt shas (spark4): r0 `d08ccfec...`, r1 `183bf7fc...`, r2 `d3d66c6a...`, r3 `4fbc9336...`. fills owed: receipt copies + sha-verify on spark5/6/7 (LAST — active lane). **packs_v2 and packs (v1) are superseded generations still on disk (see §S2).** |
| Qwen 3.8 27B TP1 (dev: spark2 + co-resident spark9/a; fleet policy = re-emit sharded, pending) | **3/3 PRESENT, 30,135,214,592 B** | `~/sparkdata/qwen38.fp8.tp1/packs/qwen38-fp8.tp1.qwen36sp` on spark2/9/a (sha trio in §S3). Extra copies: spark3, spark8 (same size, other lanes' use), spark5 (OLD size 29,948,282,112 B). |
| dsv4_flash TP16 (deployed fleet-wide; not in staging brief) | **16/16 present, BUT sparkf's pack differs: 10,072,381,148 B vs 9,013,048,832 B on the other 15** | `~/sparkdata/dsv4_flash.fp8.tp16.b1/packs/dsv4_flash_stage.spstage`. sparkf's is a different generation — flagged to the coordinator/dsv4flash lane (staging does not arbitrate un-briefed models). |
| qwen38max TP4xPP4 wire-v2 (all 16 once COMPLETE) | **PENDING — 15/16 packs on spark7 ONLY, 12/16 receipted; `qwen38max.pp3.tp4-rank3.qwen38sp` MISSING; rank3 receipts missing for pp0-2** | `~/sparkdata/qwen38max.tp4/packs/` on spark7: pp0 r0-3 (90,290,595,840 B), pp1 r0-3 + pp2 r0-3 (86,212,642,816 B), pp3 r0-2 (98,359,464,192 B). Receipts (`kind: sparkpipe.qwen38.stagepack-receipt.v2`, source quark-mxfp4 @ warm `packbuild/qwen38max/amd-mxfp4`) exist for r0-2 of every stage only. Build interrupted Aug 28 15:12. The brief's "NOT BUILT" was stale — the correct state is "15/16 built on one node, 12/16 verified". **No distribution until the set completes** (a half-set baked onto 16 nodes would be worse than the hoard); owner: qwen38max-shard lane / coordinator. spark0's `~/sparkdata/qwen38max.tp4/packs/` is an EMPTY scaffold dir. |

Pending (not gaps against current policy): K3 stages 0-2 (building, K3 lane);
27B re-emit sharded; qwen38max completion + fleet distribution; dsv4_flash
sparkf generation question.

## S1.3 Fleet pack budget (measured, current-policy packs only)

```
glm5_next  21.7 GB x16 = 347 GB    dsv4_pro  ~88-93 GB x16 = 1,433 GB
glm52      102.8 GB x8 = 823 GB    K3 st.3   98.1 GB x4  = 392 GB (stages 0-2 to come: +~294 GB)
qwen4_flash v3 60.2 GB x4 = 241 GB  27B TP1  30.1 GB x3  = 90 GB
dsv4_flash tp16 9.0 GB x16 = 144 GB
```

## S2 DESTRUCTIVE-ACTION MANIFEST (per node, EXACT paths)

Executed only after this report is committed+pushed. Guards on every delete:
(1) `lsof +D <dir>` shows nothing open (skipped if anything holds a file),
(2) for duplicates: sha equality against the kept copy confirmed first,
(3) qwen-flash deletions run LAST (active S5/S6 lane on spark4-7) and only
after that node's v3 pack sha-verifies against its receipt.
NOT in this manifest = stays (see §S2.2 keep-list).

### spark1 (frees ~336G)

| Action | Path | Size | Justification (receipt/command evidence) |
|---|---|---|---|
| DELETE (if sha==sparkc's) | `/home/spark1/sparkdata/glm52.tp8.fp8/packs/glm52_tp8_rank04.fp8.glms52sp` | 102.8G | duplicate of the band's rank04 (sparkc); glm52 band = spark8-f per glm52-packs report + README (not a fleet model). Cross-sha in §S3; if unequal → KEEP+flag |
| DELETE | `/home/spark1/sparkdata/glm52.fp8.pp13/` | 56G | Jul 7-10 per-layer dataset experiment (mtimes; `fp8_moe_pack_manifest.json`); glm52 lane COMPLETE (PR 728) |
| DELETE | `/home/spark1/sparkdata/dsv4_pro_ga.val0p3.spstage` | 85.5G | Aug 17 03:17 validation-era build; superseded by the `validated:true` regen deployed ON THIS NODE (sha `e9de954b...` verified above) |
| DELETE | `/home/spark1/sparkdata/dsv4_pro_ga.val0p4.spstage` | 99.4G | Aug 17 03:13; same class (99,371,524,236 B != rank bytes — a different stale build) |

### spark2 (frees ~135G)

| Action | Path | Size | Justification |
|---|---|---|---|
| DELETE | `/home/spark2/sparkdata/glm52.fp8.pp13/` | 56G | same July-era class as spark1's |
| DELETE | `/home/spark2/sparkdata/qwen38.fp8.tp1.oldaug26/` | 79G | self-identifying OLD generation (`oldaug26`); current 27B pack (30,135,214,592 B) present on-node |

### spark3 (frees ~57G)

| Action | Path | Size | Justification |
|---|---|---|---|
| DELETE | `/home/spark3/sparkdata/qwen38.fp8.tp1.oldstack/` | 57G | self-identifying OLD stack; current pack present on-node (knee lane used it, PR 730 merged) |

### sparkb (frees ~1,052G)

| Action | Path | Size | Justification |
|---|---|---|---|
| DELETE | `/home/sparkb/sparkdata/dsv4_pro.full.spstage` | 864.9G | Aug 15 14:05 pre-GA FULL master (864,875,157,944 B); the current generation = 16 rank packs, receipts `validated:true` on every node + my spark1/2 sha equality; rebuild path proven deterministic from the complete warm checkpoint (892,762,507,005 B, DOWNLOAD_STATUS complete) |
| DELETE | `/home/sparkb/sparkdata/dsv4_pro.val4.spstage` | 57.4G | Aug 15 partial validation build, superseded |
| DELETE | `/home/sparkb/sparkdata/dsv4_pro.valtail.spstage` | 73.2G | Aug 15 partial, superseded |
| DELETE | `/home/sparkb/sparkdata/dsv4_pro_ga_hdr.val4.spstage` | 57.4G | Aug 17 00:30 GA-header validation build, superseded (identical size to val4 = same-era slice) |
| DELETE (empty dirs) | `/home/sparkb/sparkdata/dsv4_pro.rankwork/`, `/home/sparkb/sparkdata/dsv4_pro.regenwork/` | 0 | empty (ls: no entries); regen work moved to spark6 per the dsv4pro report |

### sparka (frees ~31G)

| Action | Path | Size | Justification |
|---|---|---|---|
| DELETE | `/home/sparka/sparkdata/qwen4_flash.tp4/packs/qwen4_flash_full.tp4-rank0.qwen4_flashsp` (+ its `.receipt.json`) | 31G | v1 generation (33,069,091,328 B, Aug 27 19:27) — superseded by v2 then verified v3; the same v1 bytes also exist on spark4-7 (deleted there too, below); build-hub residue |

### spark4, spark5, spark6, spark7 — LAST, per active-lane courtesy (frees ~246G per node)

| Action | Path (per node) | Size/node | Justification |
|---|---|---|---|
| DELETE | `~/sparkdata/qwen4_flash.tp4/packs/` (v1: 4 x 33,069,091,328 B + receipts) | ~123G | v1 superseded twice (v2 merged 7c28c72; v3 = S4 COMPLETE 4/4 verified, receipts on spark4, deploy_v3 references packs_v3 only) |
| DELETE | `~/sparkdata/qwen4_flash.tp4/packs_v2/` (4 x 33,165,429,248 B + receipts) | ~124G | brief's explicit class: "old v2 when v3 verified" — v3 verified this session per-node before any v2 unlink |

Precondition per node (logged before deletion): the node's own v3 pack
sha256 == its receipt sha; `lsof +D ~/sparkdata/qwen4_flash.tp4` shows no
process holding v1/v2 files (the lane's S5/S6 daemons reference deploy_v3).

### Nothing-to-do classes (verified empty this session)

- `*.predebug` backups: `find /home /tmp -maxdepth 4 -name "*predebug*"`
  returned NOTHING on all 16 nodes — the brief's assumed backups do not exist.
- stale `.tmp`/`.part` packs: only
  `spark1:~/sparkdata/qwen38_2.4t_a95b/checkpoint/model-00022-of-00213.safetensors.part`
  — inside a KEEP-flagged tree (see below), so it stays with it.

## S2.2 KEEP list — big items that STAY (ambiguous / other lanes' / deliberate)

| Node(s) | Path | Size | Why it stays |
|---|---|---|---|
| spark1/2/3 | `~/sparkdata/qwen38_2.4t_a95b/` | 171G/148G/136G | old qwen-max source-download scratch (213 shards + a `.part`); superseded as source by warm `amd-mxfp4`, but it is a DIFFERENT source variant (a95b) — coordinator call, 455G at stake |
| spark1/2/3 | `~/sparkdata/k3.mxfp4.tp16/` (rank0{1,2,3}.pack, 99,562,379,520 B each) | 93G each | K3 TP16 = refused topology, but K3 lane ACTIVE — their artifacts to retire |
| spark1/2/3 | `~/sparkdata/k3.mxfp4.tp4pp4/` (stage0 rank0{1,2,3}.pack, 53,564,269,824 B, Aug 16) | 50G each | pre-reconciliation stage-0 generation; the replacement is mid-build on sparke — retire when it deploys |
| spark0-3 | `~/sparkdata/dsv4_flash.fp8.{pp13,pp13.b16,pp13.b8,tp16.b1.128,tp16.b1.gpudirect,tp4.b1,tp4.b1.hostrdma}/` | ~110G/node | old dsv4_flash topologies, NO receipts found to arbitrate "newest verified" — not staging's brief |
| sparkb | `~/sparkdata/qwen38max.tp4pp4/` | 574G | parked full-width scale proof (coordinator arbitration, qwen38max merge note) |
| spark7 | `~/sparkdata/qwen38max.tp4/` | 1.3T | the in-progress v2 rank set (15/16) — the completion source |
| spark2 | `/tmp/glm53_mid_stage{0,1}.g5nsp` | 42G | glm53 lane's synthesized packs (ACTIVE lane on all 16) |
| spark4-8 | `/tmp/dsv4bisect-*`, `/tmp/dsv4_flash_stage_v4.rank2.spstage` (spark6) | ~270G | deliberately kept for the open 32K-prefill investigation (PR #732 note) |
| sparka | `/tmp/lane-qwenflash/` | 25G | ACTIVE qwen-flash lane worktree |
| spark9 | `~/sparkdata/lane-glm52/` + `/tmp/g5n_test` | 2.5G | glm52 follow-up residue, small |
| spark0 | `/tmp/rdma_debug_build` | — | MINE per brief — keep |
| spark3/8 | `~/sparkdata/qwen38.fp8.tp1/` copies; spark5 old-gen 27B pack | 29-31G each | dev/p1a/knee usage; 27B not yet a per-node fleet model |
| spark0-3 | `~/sparkdata/dsv4_compress_ingest_*` | 38G/node | ingest-control dataset (da7f910-pinned), unknown owner |
| ALL | running daemons, lane worktrees, queue state | — | READ-ONLY per lane rules; no process touched by this lane |

Warm-side note for the coordinator (not acted on): the dsv4pro stash
`/mnt/model-warm/packbuild/dsv4pro/rank{1,2}.spstage` (186G) is now
redundant — both packs verified deployed — the dsv4pro report itself says
"delete after ranks 1-2 deploy". Warm is shared storage; deletion is the
coordinator's call.

## S3 FILL PLAN (post-manifest)

1. dsv4_pro receipts: write `rank{1,2}` receipt.json (from the warm stash
   bytes, already sha-matched) next to the packs on spark1/2.
2. qwen4_flash v3: copy `rank{1,2,3}.receipt.json` spark4→spark5/6/7, then
   sha256 each node's own pack and compare to the receipt (spark4's own r0
   too). LAST (active lane; avoid saturating mid-measurement).
3. glm5_next: pin the full 16-rank sha256 manifest (running; appended below)
   and drop a receipt.json next to each pack recording it.
4. NO byte fills needed anywhere: every policy pack is present (dsv4_pro 1/2
   were already byte-exact; K3 4/4 verified; qwen v3 per-node ranks present).
5. qwen38max: NO distribution (set incomplete — see gap table).

## Commands (representative; full session log in lane records)

```
$ for h in spark0..sparkf: ssh $h 'df -h /home | tail -1; du -sh ~/sparkdata/*'
$ ssh spark1 'sha256sum ~/sparkdata/dsv4_pro.tp4pp4/packs/dsv4_pro_tp4_pp4_stage.spstage'
e9de954be40628dcc4f0d2d88fe1b2391b50ba7910ca9a8e5ba249b6a53b9879  ...
$ ssh spark2 'sha256sum .../dsv4_pro_tp4_pp4_stage.spstage'
acc6d761cb79174abe2ec8e8b53e8bf38ca98dde87a2ad939d3beccd95ad5863  ...
$ ssh sparkc 'sha256sum ~/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage3.rank00.pack'
23df8aa64e02cbc62fdf318bc0189cbef18192faaf55ce464b05e14f1bcfabf9  ...   (x4 nodes, all match warm receipts)
$ ssh spark0 'sha256sum ~/glm53_packs/glm5_next_stage.tp16.rank0.g5nsp'
ca0e427ce9c33f0f6b536984bfbf4a50ae3498ba814ab7310e5f77aa24e0e6c1  ...
$ ssh spark4 'cat ~/sparkdata/qwen4_flash.tp4/packs_v3/*.receipt.json'   (shas in gap table)
```
