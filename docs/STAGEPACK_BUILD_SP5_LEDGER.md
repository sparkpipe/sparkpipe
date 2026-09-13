# SP-5 stagepack build wave ledger (manager-2, missing-pack build wave)

All times UTC. All sha256 are full-file digests. node paths are absolute on the
named spark. Justification for every entry: the SP-5 wave brief (fleet audit
build gaps 1-4) plus the placement law. Verification chain: source sidecar or
packer receipt -> placed bytes -> destination re-read.

## Prior state accepted (verified this wave, not modified)

| arm | rank | node | bytes | sha256 | chain |
|---|---|---|---|---|---|
| ling.bf16.tp16 | 0 | spark0 | 15725069824 | 1f0642fb3362202a53f95e90f565b599c270908d7b5e71c24a5f3861ec7e146d | == PROGRESS.md packer receipt rank0; sha256sum of placed file 2026-09-13 |
| ling.bf16.tp16 | f | sparkf | 15725069824 | 0df030880eec4d324249370e457f1a465e6a621f575924656a991eb5e512d6bd | == PROGRESS.md packer receipt rank15; sha256sum of placed file 2026-09-13 |

qwen3flash.bf16.tp8: all 8 replica pairs byte-equal (spark r == spark r+8),
sha256sum sweep 2026-09-13: rank0 44b78acc6af814b5ae04d6c26d18240c26a8969b33b56f729d1a2184a8f4293f
(spark0, spark8, and the stray on spark4), rank1 1c3875cdc507e6ff21a047fb6c9cb11cd3fa40479311ebff4a334ee27c46502a,
rank2 65a44e9b173512b23fcd21dd38f26f8a7994a74450f3e61999ef855ac9cd7600,
rank3 9d4d447050288133500aa8cbcb12882802cfdf464c392ab27262de4fcdbef5ea,
rank4 5b0d8ffcd65c7405c53f8c87f4f39e9cf5b6993554f4551be912529366f58638 (==
spark4/sparkc rank4.receipt deployment sha 2026-09-01T11:39:13Z),
rank5 6aa25ab0ed538d7f14a194bbe2b86c608807cf53245e4bd99c44e5ad8b6a021e (==
spark5 arm-root sidecar), rank6 f84ad41b631261ae90aa1a3163221f95ff9afb1c83f2b9e630bbfd963f0a0673,
rank7 52e9cb42c75f01a98be0120817bada71008a65cf3d243351396852d0f1afb20e.
qwen3flash.bf16.tp4pp4: all 16 stage-rank packs present, rank r -> spark r;
digests recorded in the sweep (unique per rank as expected for stage packs).

qwen3flash.fp8.tp8 rank05: NOT missing (audit finding stale). spark5 ==
sparkd == 30d5b24ef26e079951fcc51ae24f81f01170847a82be934c3c8db3238d446f26 ==
receipt output_sha256 (mtp stripped, 31 entries dropped, 30518614272 bytes).

## Operations

- 2026-09-13T05:14Z REPAIR sparkf:sparkdata/qwen3flash.bf16.tp8/packs/qwenflash.tp8.rank7.pack.experts
  18456 -> 27656 bytes, sha256 ab5d3632977be441f10cfd13b1258764d5795d4800be221a4b8461ab059c9537
  -> e776d3899f02984f0458d70435744168d0b365cdfaa6738e5792f42ceb4e016b (copied from
  spark7 sibling; pack bytes verified pair-equal so the sibling manifest is valid
  for this pack). Pre-checks: no open handles (lsof), no immutable flag (lsattr).
- 2026-09-13T05:28Z BUILD ling.bf16.tp16 rank1 on spark1 under
  systemd-run MemoryMax=4096M MemoryHigh=2900M, source
  /mnt/model-warm/ling-3.0-flash, packer tools/ling_stagepack.py @ main 2c73001
  (shipped to ~/lingbuild, dry-plan green: 730 tensors, census 63783 = 62230 +
  1553). Output ~/lingbuild/packs (staging).
- 2026-09-13T05:28Z BUILD ling.bf16.tp16 rank2 on spark2 (same chain as rank1).
- 2026-09-13T05:49Z BUILD COMPLETE rank1 spark1: 15725069824 bytes, sha256
  11169d972afbd549e700a2b32b30b8753edaff1dcf073727fa0e147e5cb1aee5, receipt
  rank1.json sha256 == sidecar == file. NOTE: identical to the pre-existing
  on-node sidecar digest -> the packer is deterministic (same inputs, same
  bytes) and the displaced pack was byte-genuine.
- 2026-09-13T05:49Z BUILD COMPLETE rank2 spark2: 15725069824 bytes, sha256
  e08e8c985a1fb13a394cdddc948ad6ea4d291578ff2a89df96b435e3cd504504 (same
  determinism note as rank1).
- 2026-09-13T05:55Z PLACE rank1 -> spark1:~/sparkdata/ling.bf16.tp16/packs/
  (pack + .experts + .sha256 + receipts/rank1.json; staged copy re-hashed
  before move; destination sha256sum -c PASS; pre-checks lsof/lsattr clean).
  Staging removed; cache purged on spark1 + spark2 05:56Z.
- 2026-09-13T05:55Z PLACE rank2 -> spark2:~/sparkdata/ling.bf16.tp16/packs/
  (same chain as rank1). SPARK_OK both.
- 2026-09-13T05:52Z BUILD (detached units sp5-ling-r3/sp5-ling-r4) rank3 on
  spark3, rank4 on spark4, same packer/source/memory caps; logs
  ~/lingbuild/r3.log r4.log.
- 2026-09-13T06:13Z BUILD COMPLETE rank3 spark3: 15725069824 bytes, sha256
  c1ffdc0c4cdc608a9371f9108c3614124a9094d88bdc0024beb72c8cb76fb118 (== the
  pre-existing sidecar; census closes; 730 tensors, 40960 manifest ranges).
- 2026-09-13T06:13Z BUILD COMPLETE rank4 spark4: 15725069824 bytes, sha256
  fbe7565e9947b0b832ef063764418c48cfda2261cb0eea5b488f0033889d9efe (== the
  pre-existing sidecar).
- 2026-09-13T06:15Z PLACE rank3 -> spark3, rank4 -> spark4 (SPARK_OK, staged
  copy re-hashed, destination sha256sum -c PASS, ownership normalized to the
  node user per the spark0 rank0 convention; placement script updated with
  chown + SPARK_FAIL 26). Cache purged on spark3 + spark4.
- 2026-09-13T06:18Z BUILD (detached units sp5-ling-r5/sp5-ling-r6) rank5 on
  spark5, rank6 on spark6.
- 2026-09-13T06:05Z IDENTIFY spark9:~/sparkdata/ling.bf16.tp16/packs/
  ling_stage.tp16.rank0.lspk 15725069824 bytes sha256 1f0642fb3362202a53f95e
  90f565b599c270908d7b5e71c24a5f3861ec7e146d == the rank0 packer receipt ->
  it is a duplicate of rank0 (correctly placed on spark0) under the retired
  ling_stage naming. This is the audit's "mislabeled pack" (a rank0-content
  file sitting in spark9's ling arm; spark9's actual rank9.sp sidecar
  cf62a572... is a different, unreceipted digest and gets REPLACED by the
  fresh rank9 build in its wave). Queued for removal in the spark9
  maintenance window: lsof clean + lsattr clean verified 06:20Z.

## Verification sweeps (filesystem-only; header/dir walk + sha256; no module execution)

- qwen38_max.tp4pp4: 18/18 files MATCH their .receipt.json output_sha256
  (16 law ranks + strays rank00@sparka, rank02@sparkf — stray CONTENT is
  receipt-genuine; placement-cleanup wave decides removal). One NO_RECEIPT
  item: /home/spark2/sparkdata/qwen38_max.tp4pp4/packs/
  .qwen38_max.tp4_pp4.rank02.spstage.yog7kztb.tmp 7594368512 bytes sha256
  083190e4fb05a31560f56b477aec7b5500831841d800c9c81a03873037b4a530 —
  interrupted partial, rank02 itself verified; litter for the cleanup wave.
- qwen38-27b.nvfp4a16.tp4: 16/16 MATCH receipts (tp4-rank00..03 quartile
  placement verified intact).
- qwen27b.tp4: 16/16 MATCH receipts. .backup files quartile-consistent
  (rank0 623b0365..., rank1 bc1a983b..., rank2 8c9c459b...,
  rank3 4d49902b...) EXCEPT spark0's backup 113931ed9e14f84af4d9a82be7a7784ee
  328ee443e27e972e56330fd0a6ec0e9 10645053184 bytes = stale-gen backup
  (reported, not deleted).
- qwen38_27b.tp4pp4: 16/16 packs present per law (rank r -> spark{hex r});
  NO receipts fleet-wide. Structural probe: placed rank00 copied to spark3
  scratch and verified with the current main module's verify() -> VERIFY_OK
  213 tensors 4359934976 bytes (copied back-clean). Current-main dry-plan for
  the same rank plans 213 tensors but 6163031040 bytes -> the placed arm is a
  DIFFERENT PAYLOAD GENERATION (same tensor inventory, smaller expert codec or
  fp8-source pass-through). Rebuild would NOT be receipt-exact; deferred to
  the gen-ruling session with the exact evidence above.

## Disk-cache purges (sync + drop_caches via sudo -n, all PASS)

- 2026-09-13T05:56Z spark1 spark2 (post-placement batch)
- 2026-09-13T06:00Z ALL 16 nodes (after the 2.2 TB quartet+backup hash sweeps)

## Reported, not acted on (outside SP-5 scope)

- qwen3flash.fp8.tp8 gen-skew: ranks 1/4/6/7 (plus rank3, same pattern) exist as
  two byte-generations (30518614272 vs 30425355496 bytes) across replica pairs;
  rank3 skew reported here for Wave-G since the brief listed only 1/4/6/7.
  sparke rank06 .experts is 9016 bytes vs 18176 elsewhere and was NOT repaired
  because the rank06 pair is gen-skewed; manifest must be regenerated from
  whichever gen Wave-G crowns.
- qwen38_max.tp4pp4 stage packs of two size generations coexist (93243509760 /
  89162287616 / 150055310080); stray rank00 (93243509760) on sparka and sparkf;
  stray rank0 duplicates on spark4 in both qwen3flash.bf16 arms; spark9 holds
  fp8 rank01+rank03+rank05. All left in place pending SP-4 refinement; nothing
  deleted outside SP-5's own arm targets.
- 2026-09-13T06:40Z BUILD COMPLETE rank7 spark7: sha256
  20de00d7f794461d5088e5380df774d793856f0156c0da04df9c2cce605cd15c (==
  pre-existing sidecar). PLACE SPARK_OK 06:42Z, spark7 cache purged.
- 2026-09-13T06:47Z BUILD rank8 on spark8 STALLED: two attempts (sp5-ling-r8,
  sp5-ling-r8b) both reached <300 MB then 0 MB/s over 45 s+ windows on the
  warm ceph read while spark7's paired build ran normally; ceph mount itself
  healthy (small reads fast). Stopped both; staging partials removed.
- 2026-09-13T07:12Z WAVE 5 contention finding: the paired rank8-staging build
  on sparke (sp5-ling-r8s) stalled the same way; ps showed SP-4's
  verification (qwen4_flash_pack_verify.py against /mnt/model-warm/
  qwen3.8-flash-next, D-state) consuming sparke's warm-ceph client at the
  same time. Yielded: stopped sp5-ling-r8s, removed its staging dir; the
  fleet heavy-read budget stays with the verification wave. Remaining ling
  builds run SEQUENTIALLY (1 stream).
- 2026-09-13T07:33Z BUILD COMPLETE rank9 spark9: sha256
  cf62a57278306601f7f1fbc8790860b1a5aef22f730c6a1cc8224a9336282800 (==
  pre-existing sidecar -> the placed rank9.sp was genuine; the audit's
  mislabel was the .lspk duplicate, not rank9.sp). PLACE SPARK_OK; REMOVED
  ling_stage.tp16.rank0.lspk (rank0 duplicate, 15725069824 bytes, sha256
  1f0642fb...; justification above); staging partials removed; spark9 cache
  purged.
- 2026-09-13T07:36Z BUILD rank10 on sparka (sp5-ling-ra, solo).
- 2026-09-13T07:50Z WORKTREE EVENT: /Users/mac/wave-b1 was deleted externally
  mid-wave. No work lost: both commits (e90b429, aa77753) were already pushed;
  worktree re-cloned from origin/lane/stagepack-build-sp5 at aa77753 and this
  entry recreates the only uncommitted ledger additions.
