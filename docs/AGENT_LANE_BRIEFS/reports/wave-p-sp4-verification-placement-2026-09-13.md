# Wave-P (SP-4) fleet stagepack verification + placement wave — 2026-09-13

Agent: SP-4 / Wave-P (fleet manager-2 stagepack campaign). Branch
`lane/wave-p-stagepack-sp4` off origin/main `2c73001fc9f03442cb8f967f087ade0c0b4445dd`.
Scope: the 8 unverified arm rows of the 09-11 placement matrix
(SHARED_DECISIONS.md): glm53flash.fp8.tp8, glm53flash.fp8.tp4pp4,
qwen3flash.{bf16,fp8}.{tp8,tp4pp4}, glm53full.{bf16,fp8,nvfp4}.tp4pp4,
k3.mxfp4.tp4pp4. Filesystem-only (no module execution, no daemon interaction).

## Method

Per node (16/16): full-census pass over the 10 arm dirs — sha256 of every pack
(8 MiB streaming, 206 files incl. strays/temps), family header census (glm5_next
v2 264B / qwen4_flash 120B wire headers), .experts + receipt + .sha256-sidecar
harvest, lsattr immutable-flag scan — under `sudo -n systemd-run --scope -q
-p MemoryMax=4096M -p MemoryHigh=2900M --uid=<node>`, followed by
`sync; echo 3 > /proc/sys/vm/drop_caches` (sudo -n) after each node's batch.
Family verifiers (main `2c73001`, staged to /tmp/sp4tools on each node):
glm52_validate_pack ×48, k3_verify_pack --quick ×16, qwen4_flash_pack_verify
per qwen pack (8 byte-trace samples vs /mnt/model-warm/qwen3.8-flash-next[-fp8]),
glm5_next_pack_verify spot-checks per fp8.tp8 generation. Artifacts:
wave-p-sp4-sweep-ledger-2026-09-13.{json,md} (206 rows: node, arm, rank, bytes,
sha256, entry_count, mtp, receipt verdict, experts sha256).

## Result matrix (✅=verified, S=struct+digest only, ❌=flagged, ?=verdict in flight)

| arm | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | sA | sB | sC | sD | sE | sF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glm53flash.fp8.tp4pp4 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| glm53flash.fp8.tp8 | ❌G | ❌G | ❌G | ❌G | ❌G | ❌G | ❌G | ❌G | ❌G | ❌G | ❌G | ❌G | ❌G | ❌G | ❌G | ❌G+tmp |
| glm53full.bf16.tp4pp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| glm53full.fp8.tp4pp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| glm53full.nvfp4.tp4pp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| k3.mxfp4.tp4pp4 | ❌R | ❌R | ❌R | ❌R | ❌R | ❌R | ❌R | ❌R | ❌R | ❌R | ❌R | ❌R | ❌R | ❌R | ❌R | ❌R |
| qwen3flash.bf16.tp8 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen3flash.fp8.tp8 | ❌F | ❌F | ❌F | ❌F | ❌F | ❌F | ❌F | ❌F | ❌F | ❌F | ❌F | ❌F | ❌F | ❌F | ❌F | ❌F |
| qwen3flash.bf16.tp4pp4 | ✅ | ✅ | ✅ | ✅ | ✅ | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| qwen3flash.fp8.tp4pp4 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 | ❌S3 |

No MISSING cells: every node holds its lawful rank for all 10 arms surveyed
(TP8 arms: rank r on spark r and r+8; TP4×PP4: rank r on spark r).

## Per-arm verdicts

1. **glm53flash.fp8.tp4pp4 — ✅ 16/16 VERIFIED.** Pack sha256 == Wave-B1
   receipt output_sha256 on all 16 (ranks 0-3 ec=272, 4-11 ec=287, 12-15
   ec=314, MTP-free, flags=0), .experts present, chattr +i locks confirmed.
   Content proof = Wave-B1's all-tensors round-trip (2026-09-12); this wave's
   digests re-anchor the placed bytes.

2. **glm53flash.fp8.tp8 — ❌ defective generation, repair-wave rebuild.** All 16
   structurally valid vs their own receipts (578c3aab on s0 …), header PASS,
   layout PASS. Content: gen-A (ec=1157, 42,063,908,352 B, 14 nodes) FAILS the
   current packer plan-diff at the entry-count check (1157 != 1160) — the
   pre-#877-pairing packer's output (packs dated 09-04/09-05; #877 = e38ebb2,
   09-09). gen-B (ec=1160, 42,381,110,784 B, s8+s9 only, dated 09-05 13:56,
   also pre-#877) is count-compatible but content-unproven (deep spot stalled
   on the ceph window; see watch items). Replica skews: rank0 s0 != s8, rank1
   s1 != s9. Per the SD-262 MTP ruling both generations are MTP-free
   (flags=0), so the newer digest (gen-B) would be the interim canonical —
   but the whole arm is the Wave-B1 defect class and must be REBUILT through
   the current packer, not re-pinned. Cleanup: sF holds a 13,156,747,648 B
   `glm5_next_stage.tp8.rank7.g5nsp.compact.tmp` leftover (the failed 09-05
   compact strip).

3. **glm53full.{bf16,fp8,nvfp4}.tp4pp4 — S ×48 STRUCT+DIGEST.** glm52_validate_pack
   rc=0 on all 48 packs (346 tensors each; tp4, rank = node % 4 correct;
   expert codec per arm: bf16=1 / fp8=5 / nvfp4=6). Stage size classes uniform
   per quartile (bf16 86.5/100.7/95.7/95.7 GB; fp8 46.7/53.9/51.2/51.2 GB;
   nvfp4 27.5/31.2/29.7/29.7 GB). This wave records the first sha256 anchors
   for these packs (no receipts existed; the .experts sidecars are present
   48/48). No content-vs-checkpoint digest anchors exist for this arm —
   candidate work for a later session; structurally and size-wise the arm is
   sound.

4. **k3.mxfp4.tp4pp4 — placed 16/16, struct-PASS 16/16, ❌ REPLICA SET NOT
   CANONICAL.** k3_verify_pack --quick PASS on all 16 packs (stage0 552
   tensors/24 layers, stage1 537/23, stage2 534/23, stage3 534/23 — uniform
   per stage). BUT every one of the 16 packs has a DISTINCT sha256: the four
   quartile "replicas" of each rank (e.g. stage0.rank00 on s0/s4/s8/sC =
   bef86d7e/2fd46f28/c2ee5a33/23df8aa6) are four different builds. Each pack
   is individually valid, so serving works per-node, but the placement law's
   replica contract (same rank == same bytes) is violated fleet-wide. Repair
   wave: digest-identify the four generations per rank (D-2 process), pick or
   build one canonical per rank, re-place the other three quartiles.

5. **qwen3flash.bf16.tp8 — ✅ 16/16 VERIFIED (MTP-embedded generation).** All 8
   ranks content-PASS (8 byte-trace samples vs checkpoint, header geometry,
   1246 entries) and every replica is digest-identical to its pair
   (44b78acc r0 ×3 incl. s4's stray, … 52e9cb42 r7 ×2). Census note: this arm
   embeds the MTP tail (mtp_layer_count=1); the MTP-free canonical question
   belongs to the gen-reconciliation session per the sidecar law — the packs
   as placed are valid for their build generation.

6. **qwen3flash.fp8.tp8 — ❌ STRUCTURAL FAIL vs current format contract,
   repair-wave rebuild.** qwen4_flash_pack_verify rc=1 on every completed pack
   (6/6 attempted before the ceph window: ranks 01/02/03/04/06 on
   s1/s2×2/s3/s4/s6), all with the same deterministic signature — hundreds of
   directory entries with payload offsets not 256-aligned and MXP4-group
   scale_group 0 != 128 (1,359 failing entries on s4's pack alone). The packs
   MATCH their build receipts, so this is an old-format generation, not
   post-build corruption. Replica skews (ranks 1, 3, 4, 6, 7 — sD/s9 strays
   match the sB-class generation), plus 31 hidden `.tmp` copy partials
   (~105 GB across the 16 packs dirs) and stray rank03/rank05 on s9.
   Placement-complete but content-invalid as a set: rebuild through the
   current packer; purge the .tmp partials.

7. **qwen3flash.bf16.tp4pp4 — stage0 ✅ (ranks 0-3 content-PASS), ranks 4-15
   digest-anchored, verdicts in flight.** Ranks 0-3 PASSed with 8 byte-trace
   samples each (311 entries, MTP-embedded); ranks 4-15 (300/336 entries,
   uniform per stage) are placed with fresh sha256 anchors and sound headers;
   their verifier runs were queued behind the stalled fp8 checkpoint reads on
   each node and had not emitted verdicts at report time (fail-loud: no
   verdict recorded = unverified-content for those cells until the runs
   complete; the scopes remain running and self-purge on exit). s4's extra
   rank0 pack = replica-generation stray, digest-identical to s0's.

8. **qwen3flash.fp8.tp4pp4 — ❌ stage3 quartet broken, repair-wave rebuild of
   ranks 12-15.** Ranks 0-11: uniform stage sizes (34.27/8.29/8.29 GB),
   receipt-match 16/16, MTP-free. Stage3 is a broken generation:
   - rank12 (sC, 8,286,810,112 B, bff6326e…) is BYTE-IDENTICAL to rank08
     (s8) — a copy of stage2's rank0 pack, not a stage3 pack;
   - rank13 (sD) is 8,935,638,784 B while rank14 (sE) / rank15 (sF) are
     8,890,517,456 B — two competing stage3 generations, with the 8.94 GB
     class also present as strays on s6 (7b4b80ff) and s7 (64ab598f);
   - 10 additional cross-stage strays (rank08 on s0, rank10 on s2, rank11 on
     s3, rank14 on s6, rank15 on s7, rank0 on s8, rank01 on s9, rank02 on
     sA, rank04 on sC, rank05 on sD) — all receipt-matched, digest-identified,
     delete candidates for the repair wave's cleanup pass.

## Placement completions

No copies were required in this wave's scope: all 8 target arms already hold
their lawful rank on all 16 nodes (the matrix's ? cells were
created-but-unverified, not missing). The matrix's ❌ rows from earlier waves
were re-verified placed 1 rank/node ×16: ling.bf16.tp16 (c7d740c),
muse.tp16.bf16 (M-2; note sC additionally holds the 16-pack verified build
stack), dsv41flash.mxfp4.tp8 (SP-2; spark6's copy lives under
sparkdata/dsv41flash.mxfp4.tp8/packs-r2/ — the recorded path difference),
qwenmax.nvfp4.tp16 (SP-3). Had copies been needed, the per-law procedure
(pre-copy lsof/lsattr, sparkcap'd transfer, dest-sha vs source-sha vs receipt,
.experts, chattr +i, per-node purge) was proven by the census tooling.

## Disk-cache purge log (all via sudo -n after each node's batch)

PURGED spark0 rc=0 05:47:00Z; spark1 05:54:22Z; spark2 06:05:00Z; spark3
06:11:46Z; spark4 06:11:30Z; spark5 06:10:13Z; spark6 06:09:30Z; spark7
06:09:40Z; spark8 06:09:48Z; spark9 06:11:15Z; sparka 06:09:49Z; sparkb
06:09:05Z; sparkc 06:09:18Z; sparkd 06:09:16Z; sparke 06:09:01Z; sparkf
06:09:36Z (UTC; first census batch). Spot purges additionally after every
verifier run on spark0/spark8. The in-flight qwen verifier scopes purge on
completion via their launcher.

## Watch items / blockers

- **Ceph warm-read stall window recurred** (~06:20-07:00Z): verifier sample
  reads went D-state at 0 B/s on multiple nodes (sparkd 23 min frozen on
  model-00069, spark8's glm5_next deep spot frozen from launch, spark0's fp8
  sampling crawling at ~10 MB/min). Handled per the Wave-B1 playbook: no
  verdicts killed mid-emit, sparkcap'd processes left in place, bounded polls
  only; the window self-resolved on spark0 during the wave. The in-flight
  bf16.tp4pp4/fp8.tp4pp4 verdicts will land when the window clears — cells
  marked ? are digest-anchored but lack content verdicts until then.
- **stray/.tmp cleanup (~120 GB)** and the **k3 canonicalization** are
  repair-wave work (D-2's digest-identify process); nothing was deleted by
  this wave.
- The glm53full arms carry no content-vs-checkpoint digest anchors (none ever
  existed); the new sha256 anchors make future content spot-checks cheap.

## Self-check

- No code changed; no test was modified or deleted; no pack was written or
  deleted (read-only sweep + receipts harvested). The only writes were
  /tmp tool staging on the nodes and the ledger docs on this branch.
- Zero comments in any emitted code (census.py, analyze.py, gen_qwen_verify.py,
  batch/launch scripts are /tmp working files; the committed artifacts are
  docs only).
- Claims trace to artifacts: per-pack digests in the ledger JSON; verifier
  lines in per-node logs (/tmp/sp4verify_*, /tmp/sp4qwen_* on each node,
  fetched copies in the wave's local results dir).
