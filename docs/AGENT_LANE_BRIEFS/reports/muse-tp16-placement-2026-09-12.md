# muse.tp16.bf16 placement + verification ledger — M-2, 2026-09-12

Verdict: muse.tp16.bf16 (muse_glimmer, dense GQA, bf16, tp_degree 16) is
PLACED AND VERIFIED 16/16. One pack per node at
`/home/<host>/sparkdata/muse.tp16.bf16/packs/muse_tp16_rank%02d.bf16.gsmu`
with the packer's `.receipt.json` beside it (the muse packer emits a receipt,
not an `.experts` manifest — muse is dense, no expert shards exist).

## State found vs brief

The brief said 1/16 (sparkC only, created-unverified). The fleet reality at
campaign start: all 16 rank masters staged on sparkc plus every node already
holding its own rank (placed 2026-09-11 03:23-03:37 +0900, master and
placement share identical mtimes = one preserved-copy pass). This campaign
did not move or rewrite any pack byte: it verified the entire chain, proved
packer parity against the lane tooling, applied the per-node cache purges,
and stamped the ledger. Nothing was killed.

## Packer

- Entry: `tools/muse_glimmer_stagepack.py` (lane muse, byte-identical at the
  merge point 8d59273 and at fleet tip 6c6caf52).
- Invocation per rank:
  `python3 tools/muse_glimmer_stagepack.py --source /mnt/model-warm/muse-glimmer-30b --output <pack> --tp-degree 16 --tp-rank <r>`
  Emits the pack plus `<pack>.receipt.json` (pack_sha256, index/config sha256,
  two-pass placement proof, census).
- Warm source: `/mnt/model-warm/muse-glimmer-30b`; index sha256
  `7d817b4dccb1b123fc6c1939356c65cee3a0ad462a5b821ac88280990a27d1ba`, config
  sha256 `5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14`
  (recomputed on sparkc ceph — both match every receipt).

## Verification chain

1. Structural walk, all 16 staged packs: magic 0x47534D55, version 1,
   header 120 B, entry 56 B, tensor_count 419, file_bytes 3,639,537,920;
   419 contiguous entries, first payload at align_up(120+56*419,256)=23808,
   walk ends exactly at file size; kind census 3 globals + 8 classes x 52
   layers; kv_head_of_rank 0 for ranks 0-7, 1 for ranks 8-15. 16/16 PASS.
2. sha256, all 16 staged packs == receipt pack_sha256. 16/16 PASS.
3. Per-node sweep (pack sha, receipt sha byte-identical to staged master,
   receipt's inner pack_sha256 == actual file sha, size exact): 16/16 PASS.
   `sudo -n sync; echo 3 > /proc/sys/vm/drop_caches` applied on every node
   after its verification read.
4. Packer parity: `tools/muse_parity_sample.py` re-derives payloads IN PLACE
   with the lane packer's own `write_records` (globals + layer 51) from the
   warm source onto a copy of the placed pack; an unchanged whole-file
   sha256 after the rewrite proves byte parity for the re-derived regions.
   - rank 12 (the sparkC reference): copy sha stayed `b9236b9c...` — job
     muse-parity-r12-sc, exit 0. PARITY PROVEN.
   - rank 00 second sample: copy sha stayed `928f4e55...` — job
     muse-parity-r00-s0, exit 0. PARITY PROVEN.
5. Full 16-rank rebuild-from-source re-derivation is NOT yet done: ceph is
   degraded fleet-wide (cold 1 GiB read = 394 s = 2.6 MB/s on spark0 and
   similar on sparkc vs the healthy ~1.2 GB/s), and one rank costs ~51 GiB
   of ceph reads in this packer (full-plane reads for qgkv/gate_up/down).
   Retry command per rank (fits the ttl-15 queue window once ceph recovers):
   `python3 $HOME/muse_tp16_build/tools/muse_glimmer_stagepack.py --source /mnt/model-warm/muse-glimmer-30b --output $HOME/muse_tp16_build/muse_tp16_rank<NN>.bf16.gsmu --tp-degree 16 --tp-rank <r>`
   then compare sha256 against the ledger row. Tooling is staged and
   hash-verified on sparkc and spark0 (`~/muse_tp16_build/tools/`).

## Placement ledger (16/16, verified in place 2026-09-12T16:20-16:21Z)

| rank | node | bytes | pack sha256 | verified (UTC) |
|---|---|---|---|---|
| 00 | spark0 | 3639537920 | 928f4e55b37dbfcbdd41afc1058005cf3a2cf61f12b52c5559fba23a7201ac66 | 2026-09-12T16:20:50Z |
| 01 | spark1 | 3639537920 | 7c58d388e6c1ca2eff58677054c561795bc5314204e9cd2706f80a349bf6aaef | 2026-09-12T16:20:52Z |
| 02 | spark2 | 3639537920 | a9a0b15792ec3ac38af5bd56eb867c5e1841c8f40c2622cc7db9b16b75343467 | 2026-09-12T16:20:55Z |
| 03 | spark3 | 3639537920 | eadac8d75d722ad50380148e39f749e58fb7cffc89288d56dac6c348fabba218 | 2026-09-12T16:20:57Z |
| 04 | spark4 | 3639537920 | 55ca88d5f53c358c16db2e253fa3610a5e6b360a92dd86073a8832155c7df38d | 2026-09-12T16:21:00Z |
| 05 | spark5 | 3639537920 | 33e4233f5c97ad66d029a1e76d0873738e64a37882ded47bc4615e1292f7e470 | 2026-09-12T16:21:02Z |
| 06 | spark6 | 3639537920 | d2237a0b785cdbc6a070e8db8100f43aa010c04540f8750609a1d4e46d764c69 | 2026-09-12T16:21:04Z |
| 07 | spark7 | 3639537920 | 67f32a8eabae5365615938a3c2026dce878596fb636286a0f45ad05352bf0a81 | 2026-09-12T16:21:06Z |
| 08 | spark8 | 3639537920 | 789babd0219c5eb75d7cad08695955d4e4fee7fe7d2beedb86d0d24b4f79294c | 2026-09-12T16:21:09Z |
| 09 | spark9 | 3639537920 | 8544a7f4d47d38d81db5ece83a934dae4fa4c4ec333950c01a9bb1b8d8be846f | 2026-09-12T16:21:11Z |
| 10 | sparka | 3639537920 | 3340de74249e68588cab92172d9a53e5ee528a876ec3616a036e303d65f39e6f | 2026-09-12T16:21:13Z |
| 11 | sparkb | 3639537920 | b68ecfe16235e025461522f3177c7c57c89ae5e8826647822d142f4b104093d0 | 2026-09-12T16:21:17Z |
| 12 | sparkc | 3639537920 | b9236b9c8eaf652192c345402aff8da13c0f97672d1f9103c803238066bd6daa | 2026-09-12T16:21:19Z |
| 13 | sparkd | 3639537920 | 5724d2a65addb38a88a730f7c75daa3971aaae6187f6c839f683ae819e675426 | 2026-09-12T16:21:22Z |
| 14 | sparke | 3639537920 | 9ca4663364435f35ecd48479a2c32d8dbb395b623c3c616c2d9bbceaf9d6be23 | 2026-09-12T16:21:24Z |
| 15 | sparkf | 3639537920 | d166cc74cc5494c2f66a36937f5a5e6a2f950a6641c6618096832929852e2edf | 2026-09-12T16:21:26Z |

Receipt sha256 per rank (byte-identical between the sparkc staging set and
every destination) is recorded in the campaign transcript; the receipt for
each rank embeds the pack sha256 above, the index/config sha256 of the warm
source, and the two-pass placement proof.

## Placement matrix (stagepack campaign row: muse.tp16.bf16)

spark0 ✅ spark1 ✅ spark2 ✅ spark3 ✅ spark4 ✅ spark5 ✅ spark6 ✅
spark7 ✅ spark8 ✅ spark9 ✅ sparka ✅ sparkb ✅ sparkc ✅ sparkd ✅
sparke ✅ sparkf ✅ — 16/16 placed, sha256-verified at destination,
receipts byte-identical, packer parity proven (rank 12 reference + rank 00
sample), caches purged per node.

## Queue jobs (queue v2, run-kind, ttl-15, explicit memory)

- muse-r12-parity / muse-r00-parity (sparkc, MemoryMax 4096M): cancelled —
  page-cache reclaim under the small cap made ceph reads crawl; superseded.
- muse-r12-parity-s0 / muse-r00-parity-s0 (spark0, 32768M): cancelled —
  ceph itself degraded (2.6 MB/s), full rebuild infeasible in-window.
- muse-parity-r12-sc (sparkc) / muse-parity-r00-s0 (spark0), 32768M:
  exit 0, parity proven.

## Blockers

- Ceph fleet-wide degradation (~2.6 MB/s cold) — blocks only the optional
  full 16-rank rebuild re-derivation; no placement work is blocked.
- sparkc free space dropped 869G -> 718G during the campaign from other
  writers; unrelated to this lane, flagged for the disk-reclamation owner.
