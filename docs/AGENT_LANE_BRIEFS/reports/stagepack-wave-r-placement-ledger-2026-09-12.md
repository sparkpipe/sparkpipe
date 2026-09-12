# Stagepack Wave-R placement ledger — 2026-09-12

Wave-R of the mgr2 stagepack campaign: the four cheap re-copies from verified
on-fleet sources plus the muse.tp16.bf16 fan-out. Every copy is rsync over
ssh, pushed from the holding node under the sparkcap law
(`sudo -n systemd-run --scope -q -p MemoryMax=4096M -p MemoryHigh=2900M
--uid=<node-user>`), batched per destination node, with the receiving-side
disk-cache purge (`sync; echo 3 > /proc/sys/vm/drop_caches` via `sudo -n`)
after every batch. Pre-copy checks on every source and overwritten
destination: `lsattr` (immutable flag, the D-1 chattr lesson) and
`lsof`/`fuser` (open handles). All timestamps are node-local (+0900);
`placement time` = the destination file's status-change time (`stat -c%z`).

## Re-copy ledger

### 1. sparkb qwen27b.tp4 rank3 (divergent body replaced)

- destination: sparkb:/home/sparkb/sparkdata/qwen27b.tp4/packs/qwen27b.tp4.rank3.qwen36sp
- source: sparkf:/home/sparkf/sparkdata/qwen27b.tp4/packs/qwen27b.tp4.rank3.qwen36sp
- arm: qwen27b.tp4, rank 3 (TP4 placement law: rank 3 -> every quartile;
  sparkb = node 0xB, quartile 3)
- bytes: 10,524,421,440
- sha256 source (pre-copy, sparkcapped): 4d49902bfa1cfbba588108a97e77d3ded86ba252ff9cddc2c2482a6e2c5e29a2
- sha256 destination (post-copy, sparkcapped): 4d49902bfa1cfbba588108a97e77d3ded86ba252ff9cddc2c2482a6e2c5e29a2
- receipt: 4d49902bfa1cfbba588108a97e77d3ded86ba252ff9cddc2c2482a6e2c5e29a2 / 10,524,421,440 (sparkf receipt; receipt file also relayed to sparkb)
- placement time: 2026-09-13 00:58:10 (receipt 00:58:29)
- replaced body: the SP-1-flagged divergent 10,645,053,184 B body (SP-1
  digest d4505f2e; the sweep matrix's "10,645,051,840" is off by 1,344 B from
  the on-disk size — actual bytes recorded here). Overwritten in place
  (--inplace) after lsattr/lsof pre-checks came back clean.
- spark3 was not touched (its copy is identical per SP-1; sparkf used as the
  source to keep read load off the coredev serving node).

### 2. sparkd glm53flash.fp8.tp8 rank5 (truncated body replaced)

- destination: sparkd:/home/sparkd/sparkdata/glm53flash.fp8.tp8/packs/glm5_next_stage.tp8.rank5 (+ .experts + receipt)
- source: spark5:/home/spark5/sparkdata/glm53flash.fp8.tp8/packs/glm5_next_stage.tp8.rank5
- arm: glm53flash.fp8.tp8, rank 5 (TP8 law: rank r -> spark r and spark r+8;
  sparkd = node 0xD = 5+8, the replica slot)
- bytes: 42,063,908,352 (replaces the truncated 18,618,679,296 B body)
- sha256 source (pre-copy, sparkcapped): 8ba7f5a2ca00fabc8b1528ba4c0a19375459e9e3259ef4e0d9b3830ae4cfb478 — equals the spark5 g5nsp receipt
- sha256 destination (post-copy, sparkcapped): 8ba7f5a2ca00fabc8b1528ba4c0a19375459e9e3259ef4e0d9b3830ae4cfb478
- .experts sha256 source/destination: a6e5583be3a72d42139fece6c7f6f67dc5bc669470535962eea0e54f5303c4ee (25,096 B, replaces the mismatched 11,136 B file)
- placement time: 2026-09-13 01:06:23
- transfer note: first rsync attempt exited 23 — `--inplace` also applied to
  the root-owned receipt file (permission denied); body and .experts had
  already completed and verified; the receipt was re-placed without
  `--inplace`.
- GEN-SKEW EVIDENCE PRESERVED FOR WAVE-G: sparkd's old receipt, verbatim:
  `{"bytes_reclaimed": 1098478024, "file_bytes": 42381066808, "locked": true,
  "mtp": "stripped", "mtp_entries_dropped": -57, "output_sha256":
  "a5eca1e58e51303aaf25c74640b57dd479eee0cbba97d318ccfcc544c2b1d16e"}` —
  a 42,381,066,808 B generation that matches no on-disk body. The node now
  holds the spark5-gen receipt (g5nsp v1, 8ba7f5a2…, 42,063,908,352 B) so
  receipt and body agree; the old receipt lives on here as evidence.
- source note: spark5's rank5 body carries `chattr +i` (immutable) — read as
  source only; flag left in place and reported.

### 3. spark5 qwen3flash.fp8.tp8 rank05 (missing body restored)

- destination: spark5:/home/spark5/sparkdata/qwen3flash.fp8.tp8/packs/qwenflash.tp8.fp8.rank05.spstage (+ .experts + receipt)
- source: sparkd:/home/sparkd/sparkdata/qwen3flash.fp8.tp8/packs/qwenflash.tp8.fp8.rank05.spstage
- arm: qwen3flash.fp8.tp8, rank 05 (TP8 law: node-index naming, spark5 = 5+8
  replica slot of rank 5)
- bytes: 30,518,614,272
- sha256 source (pre-copy, sparkcapped): 30d5b24ef26e079951fcc51ae24f81f01170847a82be934c3c8db3238d446f26 — equals the receipt's output_sha256
- sha256 destination (post-copy, sparkcapped): 30d5b24ef26e079951fcc51ae24f81f01170847a82be934c3c8db3238d446f26
- .experts sha256 source/destination: 1f56572f43bf8be7e5b4307e3570885377758001bc96404ca508fc6887864aa7 (18,216 B)
- placement time: 2026-09-13 01:08:46
- litter left untouched, needs a D-2-class ruling:
  spark5:/home/spark5/sparkdata/qwen3flash.fp8.tp8/packs/.qwenflash.tp8.fp8.rank05.spstage.bfc2vvya.tmp
  — 8,262,400,000 B stale partial from the failed Sep 4 build, no open
  handles (verified). Rank 05 was NOT in the gen-skew hold list (ranks
  1/4/6/7), so this copy is in scope; the held ranks were not touched.

## Muse fan-out (15/15)

Source: sparkc:/home/sparkc/sparkdata/muse.tp16.bf16/packs (the SP-1-verified
build stack; muse.tp16.bf16, 3,639,537,920 B/rank). Placement law TP16: rank
= node index, one rank per node, no replication; sparkc (node 12) already
holds the stack. Each destination received its own rank's pack plus the
packer's `.receipt.json`. Source files pre-checked (no immutable flags, no
open handles). Every destination sha256 below was computed post-copy on the
receiving node (purge first, then sha256sum; sparkcapped) and matches the
pack receipt's `pack_sha256` — which by construction also proves each
destination equals the receipted packer output even though the source files
were not re-hashed on sparkc this wave (SP-1's content-level verification is
the source-side record; a drifted source would have failed the receipt gate).

| rank | node | pack | bytes | pack_sha256 (dest verified) | placement time |
|---|---|---|---|---|---|
| 00 | spark0 | muse_tp16_rank00.bf16.gsmu | 3639537920 | 928f4e55b37dbfcbdd41afc1058005cf3a2cf61f12b52c5559fba23a7201ac66 | 2026-09-13 01:09:51 |
| 01 | spark1 | muse_tp16_rank01.bf16.gsmu | 3639537920 | 7c58d388e6c1ca2eff58677054c561795bc5314204e9cd2706f80a349bf6aaef | 2026-09-13 01:09:57 |
| 02 | spark2 | muse_tp16_rank02.bf16.gsmu | 3639537920 | a9a0b15792ec3ac38af5bd56eb867c5e1841c8f40c2622cc7db9b16b75343467 | 2026-09-13 01:10:03 |
| 03 | spark3 | muse_tp16_rank03.bf16.gsmu | 3639537920 | eadac8d75d722ad50380148e39f749e58fb7cffc89288d56dac6c348fabba218 | 2026-09-13 01:10:10 |
| 04 | spark4 | muse_tp16_rank04.bf16.gsmu | 3639537920 | 55ca88d5f53c358c16db2e253fa3610a5e6b360a92dd86073a8832155c7df38d | 2026-09-13 01:10:16 |
| 05 | spark5 | muse_tp16_rank05.bf16.gsmu | 3639537920 | 33e4233f5c97ad66d029a1e76d0873738e64a37882ded47bc4615e1292f7e470 | 2026-09-13 01:10:37 |
| 06 | spark6 | muse_tp16_rank06.bf16.gsmu | 3639537920 | d2237a0b785cdbc6a070e8db8100f43aa010c04540f8750609a1d4e46d764c69 | 2026-09-13 01:10:43 |
| 07 | spark7 | muse_tp16_rank07.bf16.gsmu | 3639537920 | 67f32a8eabae5365615938a3c2026dce878596fb636286a0f45ad05352bf0a81 | 2026-09-13 01:10:50 |
| 08 | spark8 | muse_tp16_rank08.bf16.gsmu | 3639537920 | 789babd0219c5eb75d7cad08695955d4e4fee7fe7d2beedb86d0d24b4f79294c | 2026-09-13 01:10:56 |
| 09 | spark9 | muse_tp16_rank09.bf16.gsmu | 3639537920 | 8544a7f4d47d38d81db5ece83a934dae4fa4c4ec333950c01a9bb1b8d8be846f | 2026-09-13 01:11:02 |
| 10 | sparka | muse_tp16_rank10.bf16.gsmu | 3639537920 | 3340de74249e68588cab92172d9a53e5ee528a876ec3616a036e303d65f39e6f | 2026-09-13 01:11:20 |
| 11 | sparkb | muse_tp16_rank11.bf16.gsmu | 3639537920 | b68ecfe16235e025461522f3177c7c57c89ae5e8826647822d142f4b104093d0 | 2026-09-13 01:11:26 |
| 13 | sparkd | muse_tp16_rank13.bf16.gsmu | 3639537920 | 5724d2a65addb38a88a730f7c75daa3971aaae6187f6c839f683ae819e675426 | 2026-09-13 01:11:32 |
| 14 | sparke | muse_tp16_rank14.bf16.gsmu | 3639537920 | 9ca4663364435f35ecd48479a2c32d8dbb395b623c3c616c2d9bbceaf9d6be23 | 2026-09-13 01:11:38 |
| 15 | sparkf | muse_tp16_rank15.bf16.gsmu | 3639537920 | d166cc74cc5494c2f66a36937f5a5e6a2f950a6641c6618096832929852e2edf | 2026-09-13 01:11:44 |

All 15 receipt files confirmed present next to their packs. Total placed:
15 × 3,639,537,920 = 54,593,068,800 B (50.8 GiB).

## Disk-cache purge log

Every entry is `sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"`,
verified by exit status (all OK):

- sparkb — after receiving re-copy 1 (post-purge buff/cache ~1 GiB)
- sparkf — after serving re-copy 1
- sparkd — after receiving re-copy 2 (post-purge free 103 GiB)
- spark5 — after serving re-copy 2
- sparkd — after serving re-copy 3
- spark5 — after receiving re-copy 3
- spark0, spark1, spark2, spark3, spark4 — before each muse sha256 verify (muse batch 1)
- spark5, spark6, spark7, spark8, spark9 — before each muse sha256 verify (muse batch 2)
- sparka, sparkb, sparkd, sparke, sparkf — before each muse sha256 verify (muse batch 3)
- sparkc — after serving all 15 muse ranks (post-purge free 103 GiB)

## Boundaries honored

- spark5's glm53full.nvfp4.tp16 rank5: untouched (D-2 deleted it; nothing
  resurrected).
- spark5's glm53full.fp8.tp16 stack: untouched (pending the Wave-G ruling).
- glm53flash.fp8.tp8 gen-skew ranks 0/1 and qwen3flash.fp8.tp8 gen-skew
  ranks 1/4/6/7: nothing copied.
- No daemon interaction anywhere; no serving-path changes on spark3 (the
  only spark3 write is the stagepack placement into
  sparkdata/muse.tp16.bf16/packs).
- Node split: this wave touched sparks 0-9, a, b, d, e, f as destinations
  plus sources sparkc/f/5/d — all sanctioned by the Wave-R brief; no mgr1
  lane work touched.
- /Users/mac/sparkpipe untouched; the ledger lives in this wave's own
  worktree (/Users/mac/lane-waver, branch lane/stagepack-wave-r).
