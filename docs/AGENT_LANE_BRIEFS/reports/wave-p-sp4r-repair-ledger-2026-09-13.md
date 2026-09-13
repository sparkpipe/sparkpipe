# Wave-PR (SP-4R) stagepack repair ledger — 2026-09-13

Agent: SP-4R (repair wave; rerun of the SP-4R instance lost in startup).
Branch `lane/wave-pr-stagepack-sp4r` off origin/main `c698e20` (fresh clone
`/Users/mac/sp4r`; identity verified `sparkpipe` before first write op).
Inputs: Wave-P ledger branch `lane/wave-p-stagepack-sp4` @ `6637980`
(verification + placement report and the 206-row sweep ledger) and SP-5's
`docs/STAGEPACK_BUILD_SP5_LEDGER.md` (PR #980). Filesystem + packer only:
zero weightd/daemon contact, zero module execution. spark3 (operator serving)
and spark6 (Engram staging) received only pack-file placement inside the
already-defective glm53flash.fp8.tp8 arm, behind lsof pre-checks; no service
state touched. All heavy ops sparkcap'd
(`sudo -n systemd-run --scope -q -p MemoryMax=4096M -p MemoryHigh=2900M
--uid=1000`), never root-HOME; disk-cache purge (`sync; drop_caches=3`,
sudo -n) after every placement batch, timestamps inline below.

Tools: shipped main `c698e20` tarball (sha256 `07710c6e8f51bfa2…`) to
`~/sp4rtools/sparkpipe/` on 12 build nodes. Node clones were stale (e.g.
spark0 @ f9b04ba 09-07, pre-#877) and were not used.

## Item 1 — glm53flash.fp8.tp8 REBUILD (all 16 slots, rank r on spark r + r+8)

Recipe crowned by probe: current packer `--tp-degree 8 --tp-rank R
--owns-embedding --owns-head --first-layer 0 --layer-count 45` (MTP-free
flags=0) plans exactly 1160 tensors — matches gen-B's count; without owns
flags the plan is 1157 (the gen-A defect signature). Fresh builds: 1160
tensors, 42,381,110,784 B each, dir_sha `faf6ebe9…` uniform across ranks.
glm5_next_pack_verify VERIFY-PASS (checkpoint region hashes + plan-diff) on
every rank before placement. Fresh digests (rank: sha256):

| rank | sha256 | placed on |
|---|---|---|
| 0 | 36cf4ae52b560a3e53590ea4f43a951e594a400f72e9dd9849bcb46e06ff26ed | spark0, spark8 |
| 1 | fa9ecfc79597436cf9471686948c2531fbd4e102a623bbc1503c0071025b5c15 | spark1, spark9 |
| 2 | 178e6caf1c171b2a0e70ac7f6e534f4d1d754740d76c367d7b45e6491d9653c2 | spark2, sparka |
| 3 | 1fedb34e0983a27474c5b63b2a2db514cca39603fef6e4e4d4da23ea3d0a4f36 | spark3, sparkb |
| 4 | 6cb0345e6e2622aa3c04314f6178e0f4d0d1afb4879b6e0eb29ec028b57c7553 | spark4, sparkc |
| 5 | a6888033e8e111b0a2ed0788e1f22740e887fa4bc5d16d6b4c565ce8a02e629c | spark5, sparkd |
| 6 | 54da2ee4ff6f2985e04934f4d7d9a6a6b8927cee83329472b1d2870e404193b9 | spark6, sparke |
| 7 | 350872dbca3a92508e1699c65687161a97e4a535e25322bd75e3734411dbb9c8 | spark7, sparkf |

gen-A (ec=1157) and gen-B (ec=1160, s8/s9) are both now proven non-reproducing:
the current packer's output bytes differ from every placed digest (e.g. rank0
fresh 36cf4ae5… vs gen-A 578c3aab… vs gen-B 3ad58198…). Old-generation files
per node (Wave-P digests re-confirmed by pre-delete sha, then deleted):
578c3aab spark0, bec98ec2 spark1, a63085fd spark2, 6c21ebae spark3, 7155655d
spark4, 8ba7f5a2 spark5, 278d4a0f spark6, ea4f69a3 spark7, 3ad58198 spark8,
3bc288aa spark9, a63085fd sparka, 6c21ebae sparkb, 7155655d sparkc, 8ba7f5a2
sparkd, 278d4a0f sparke, ea4f69a3 sparkf (gen-A size 42,063,908,352; gen-B
size 42,381,110,784). Per-node conventions vary (rank0 vs rank01 padded
names); each node's old files were removed under whatever name they carried,
and the new generation is placed under the canonical unpadded
`glm5_next_stage.tp8.rank<R>` + `.experts` + `.sha256` + `.receipt.json`.

.old-digests + .experts regenerated per node from the placed bytes via
build-g5n-experts-manifest (main source, -Wall -Wextra -Werror, current
wire); `.sha256` sidecars are sha256sum-format; receipts are
`sparkpipe.g5nsp.stagepack-receipt.v1` with output_sha256 = dest-recomputed
sha; packs chattr +i. Deleted old-gen bytes per node: 42,063,908,352
(gen-A nodes) or 42,381,110,784 (gen-B nodes) plus stale .experts/.receipt
sidecars; sparkf additionally `glm5_next_stage.tp8.rank7.g5nsp.compact.tmp`
(13,156,747,648) + its .experts (7,896) — the failed 09-05 compact strip.

## Item 2 — qwen3flash.fp8.tp8 REBUILD (all 16 slots)

LAW GATE applied: the fp8 arm packs FROM `/mnt/model-warm/qwen3.8-flash-next-fp8`
verbatim (`--expert-format fp8-official`, never-quantize law, coordinator-log
firing 172/173), NOT from the BF16 source — a BF16-source plan probe was
caught and killed before any placement. Current packer (e3af39e stagepack_core
wave included), `--first-layer 0 --layer-count 48 --tp-degree 8 --tp-rank R
--no-mtp` → 1215 tensors, 30,518,612,480 B (old receipted packs were 30,518,
614,272 with the struct-FAIL layout: offsets not 256-aligned, scale_group 0).

Builds: rank00-03 on spark8/9/d/f (first launch killed pre-placement when the
BF16 recipe was caught; relaunch on the official-fp8 recipe), rank04-06 on
spark9/d/f, rank07 pending. Gate: qwen4_flash_pack_verify --no-mtp 8-sample
byte-trace vs the fp8 checkpoint must PASS before placement (see verdict log).

Placement: per-node lawful names from the Wave-P census (spark0/spark8
unpadded `rank0`, all others padded `rank0N` with N = node%8); .experts
regenerated with the committed qwen38max_experts_manifest.c wire adapted to
the qwen4_flash 120B/56B layout (tools/qwen4_flash_experts_manifest.c, new
file, compiled -Wall -Wextra -Werror; v2 per-expert manifest,
SparkWeightdManifestLoad self-validated; the 09-05 sidecars were an older
v1 455-record format and are superseded). sparke rank06 .experts regenerated
from its own new pack (the truncated 9,016-byte sidecar dies with the old
generation).

Build results: rank00-03 on spark8/9/d/f (02:0xZ), rank04/06 on spark9/sparkf
(02:57Z), rank07 on spark8 (04:36Z); rank05 on sparkd crawling through the
window (in place). digests: rank00 598721746c105c85e6644b0fe074f82b3b9071cdf
1c8294375ef6f6cd77680a3, rank01-rank07 recorded in the build receipts
(<pack>.receipt.json, tool-emitted output_sha256). Gate: rank00 byte-trace
verify (qwen4_flash_pack_verify --no-mtp) launched on spark8 after its read
path cleared; verdict pending at ledger-commit time — placements for this arm
are BLOCKED behind it (fail-loud: no verdict = no placement).

## Item 3 — qwen3flash.fp8.tp4pp4 stage3 rebuild (ranks 12-15)

Recipe from the rank13 receipt: `--first-layer 36 --layer-count 12
--tp-degree 4 --tp-rank (R-12) --expert-format fp8-official --no-mtp` on
/mnt/model-warm/qwen3.8-flash-next-fp8 → 305 tensors, 8,935,637,248 B
(current-packer plan; old stage3 carried two competing generations at
8,935,638,784 and 8,890,517,456). Built on spark2/5/7/0, placed on
sparkc/d/e/f as `qwenflash.tp4_pp4_fp8.rank12..15.spstage` after
qwen4_flash_pack_verify PASS. The 15 stray gens flagged by Wave-P:

| node | stray file | bytes | sha256 |
|---|---|---|---|
| spark0 | qwenflash.tp4_pp4_fp8.rank08.spstage | 8286810112 | bff6326e… |
| spark2 | qwenflash.tp4_pp4_fp8.rank10.spstage | 8286810112 | 282760a9… |
| spark3 | qwenflash.tp4_pp4_fp8.rank11.spstage | 8286810112 | 09945291… |
| spark6 | qwenflash.tp4_pp4_fp8.rank14.spstage | 8935638784 | 7b4b80ff… |
| spark7 | qwenflash.tp4_pp4_fp8.rank15.spstage | 8935638784 | 64ab598f… |
| spark8 | qwenflash.tp4_pp4_fp8.rank0.spstage | 34270463232 | 4e3da8dd… |
| spark9 | qwenflash.tp4_pp4_fp8.rank01.spstage | 34270463232 | 29340de6… |
| sparka | qwenflash.tp4_pp4_fp8.rank02.spstage | 34270463232 | 91c2e390… |
| sparkc | qwenflash.tp4_pp4_fp8.rank04.spstage | 8286810112 | 414e3cf9… |
| sparkd | qwenflash.tp4_pp4_fp8.rank05.spstage | 8286810112 | 3105b4d0… |
| sparkf | qwenflash.tp4_pp4_fp8.rank07.spstage | 8286810112 | 455bffb2… |
| spark4 | qwen3flash.bf16.tp4pp4 qwenflash.tp4pp4.rank0.pack | 41818367232 | b11dd97e… (DELETED this wave, see item 5) |
| spark4 | qwen3flash.bf16.tp8 qwenflash.tp8.rank0.pack | 46333527808 | 44b78acc… (DELETED this wave, item 5) |
| spark9 | qwenflash.tp8.fp8.rank03.spstage | 30518614272 | 2c4634fb… (DELETED, item 5) |
| spark9 | qwenflash.tp8.fp8.rank05.spstage | 30518614272 | 30d5b24e… (DELETED, item 5) |

plus the replaced-in-place broken stage3 quartet (rank12 byte-copy bff6326e,
rank13 6e1360da, rank14/rank15 old gens) which die at placement time.

## Item 4 — k3.mxfp4.tp4pp4 canonicalization

Ground truth on the nodes: each of the 16 nodes holds exactly ONE k3 pack
(k3.stage{node/4}.rank{node%4}.pack) — there are no replica sets; the "four
quartile replicas" in the Wave-P report maps to the four distinct PP-stage
ranks, whose digests are distinct by construction. The real canonicalization
question is pack==packer, and the answer is NO for every placed pack: all 16
were sliced 08-28..08-30 (mtimes), and k3_shard.py changed 08-31 (2b27e64:
manifest_reserve 262128 → 1048560), which shifts every payload offset and
pads the manifest differently — byte layout changes, digests necessarily
change. Per the pack==packer law all four stages are rebuilt through the
current pipeline (k3_pack.py stage slice [0+24, 24+23, 47+23, 70+23] →
k3_shard.py 4 ranks → per-rank .experts regenerated with the committed
generator pattern, libk3manifest.so built from current src, loader
self-verified). Stage build state: stage0 (spark9), stage1 (sparkd), stage2 (sparkf) launched
03:22Z-ish; ALL frozen at deterministic payload offsets (5,590,697,728 /
900,703,104 / 577,957,888 B) — the kimi-k3 warm-read path stalls at the same
objects across nodes AND across a kill+retry (fail-fast applied once each).
Scopes left in place per the Wave-B1 playbook; they complete when the storage
window clears. stage3 build NOT launched (s8 was the only free candidate and
was itself stalled; it was reserved for the qwen r7 build, which completed
04:36Z). This is the wave's principal carry-forward.

## Item 5 — stray/tmp cleanup (all content-re-verified before rm)

| node | file | bytes | verdict |
|---|---|---|---|
| sparka | qwen38_max.tp4pp4/packs/qwen38_max.tp4_pp4.rank00.spstage | 93243509760 | DELETED 18:01:28Z (receipt sha match) |
| sparkf | qwen38_max.tp4pp4/packs/qwen38_max.tp4_pp4.rank02.spstage | 93243509760 | DELETED 18:02:42Z (receipt sha match) |
| spark4 | qwen3flash.bf16.tp8/packs/qwenflash.tp8.rank0.pack | 46333527808 | DELETED 18:03:19Z (Wave-P sha match) |
| spark4 | qwen3flash.bf16.tp4pp4/packs/qwenflash.tp4pp4.rank0.pack | 41818367232 | DELETED 18:03:54Z (Wave-P sha match) |
| spark9 | qwen3flash.fp8.tp8/packs/qwenflash.tp8.fp8.rank03.spstage | 30518614272 | DELETED 18:04:46Z (Wave-P sha match) |
| spark9 | qwen3flash.fp8.tp8/packs/qwenflash.tp8.fp8.rank05.spstage | 30518614272 | DELETED 18:05:06Z (Wave-P sha match) |
| spark2 | qwen38_max.tp4pp4/packs/.qwen38_max.tp4_pp4.rank02.spstage.yog7kztb.tmp | 7594368512 | DELETED 18:07:01Z (Wave-P sha match) |
| spark0 | qwen27b.tp4/packs/qwen27b.tp4.rank0.qwen36sp.backup | 10645053184 | DELETED 18:07:31Z (SP-5 sha match) |
| sparkf | glm53flash.fp8.tp8 rank7 compact.tmp + .experts | 13156755544 | DELETED with item-1 old-gen debris |

Every deletion: lsof clean pre-check, lsattr immutable pre-check (chattr -i
where set), sha256 re-verified against the ledger/receipt digest before rm.

## Purge log (sync + drop_caches=3, sudo -n, per node; UTC)

17:50:11Z spark8 | 17:50:17Z spark0 | 17:54:02Z sparke | 17:54:03Z sparkd |
17:54:04Z sparka | 17:54:04Z spark9 | 17:54:04Z sparkf | 17:54:09Z spark5 |
17:54:27Z spark2 | 17:54:31Z spark7 | 17:54:41Z spark1 | 17:57:27Z spark3 |
17:58:02Z sparkb | 18:45:37Z spark4 | 18:46:07Z sparkc — placement batches.
Post-deletion sweep (item 5 nodes): 19:59:16Z sparka | 19:59:17Z sparkf |
19:59:18Z spark4 | 19:59:18Z spark9 | 19:59:18Z spark2 | 19:59:19Z spark0.
s3/s6 purged with their placement batches; no purges skipped.

## Deleted-bytes totals

- Item 5 (strays/tmp/backup): 353,915,564,800 B across 8 receipts.
- Item 1 old-generation debris (defective tp8 packs + stale sidecars +
  sF compact.tmp): 603,003,211,974 B across 14 nodes (per-node lines above;
  spark0/spark8 debris from the finalize pass, spark4 from its inventory —
  pack 42,063,908,352 + .experts 25,096 + .g5nsp.receipt.json 321 +
  symlinkfix 200).
- Wave total deleted: 956,918,776,774 B (~891 GiB).

## Blockers / watch items

- Ceph warm-read stall windows recurred repeatedly (per Wave-P warning):
  glm-5.3-flash reads recovered quickly, but /mnt/model-warm/
  qwen3.8-flash-next-fp8 and kimi-k3 readers stalled 40+ minutes on
  spark4/8/9/d/f (D-state, 0 B/s). Handled per playbook: processes left in
  place, bounded polls, one kill+retry where a build crawled (spark4 r4 —
  rebuilt on healthy spark1 instead).
