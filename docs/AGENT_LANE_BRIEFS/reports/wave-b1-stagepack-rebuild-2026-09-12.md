# Wave-B1 stagepack rebuild ledger — glm53flash.fp8.tp4pp4 stage3 quartet + stage0-2 receipt re-pin (2026-09-12)

Agent: Wave-B1 (fleet manager-2 stagepack campaign). Branch `lane/wave-b1-stagepack-rebuild`
off origin/main `6c6caf529cf02a27a2de8fd87cdef8674c3efe22` (fresh clone
`/Users/mac/wave-b1`; `git rev-parse origin/main` == `git ls-remote origin refs/heads/main`
== `6c6caf529cf02a27a2de8fd87cdef8674c3efe22` at campaign start).

## Finding (SP-1 sweep, confirmed live 2026-09-12)

Arm `glm53flash.fp8.tp4pp4` (legacy pack names `glm5_next.tp4pp4.rankN`, rank r on spark-r,
16 ranks / 4 stages). Canonical split verified from the structurally-PASS stage0-2 packs:

| stage | ranks | hosts | layers | entries | flags |
|---|---|---|---|---|---|
| 0 | 0-3 | spark0-3 | 0 + 11 | 272 | 0 (MTP-free) |
| 1 | 4-7 | spark4-7 | 11 + 11 | 287 | 0 |
| 2 | 8-11 | spark8-b | 22 + 11 | 287 | 0 |
| 3 | 12-15 | sparkc-f | 33 + 12 | 314 | 0 |

Stage3 quartet is a broken generation: header `directory_offset` carries a stale
end-of-file value (file_bytes − 314×64: 107,211,675,136 on sparkc/d/e,
23,925,500,672 on sparkf) instead of the packer's 512; sparkc/d/e files carry a
sparse 107 GB tail (23 GB real blocks) from the failed 2026-09-05 compact strip;
receipts (2026-09-05, "compact file_bytes repair") claim first_layer 36 + 12 and
sha over bytes that no longer exist. The packs were rewritten in place on 2026-09-08
(mtimes 04:31-09:01 fleet-wide) after the receipts were written — stage0-2 receipts
are stale the same way (struct-PASS, sha mismatch, size exact).

Ruling per the packer's determinism (fixed plan, fixed transforms, no timestamps in
the wire): placed bytes that verify == the packer's output may be re-pinned; bytes
that cannot verify (stage3) are rebuilt from the source checkpoint.

## Tool staging (build node spark3 — source-local OSD reads; serving PIDs untouched)

- `spark3:~/wave-b1-stagepack/` staged from main `6c6caf52` tarball
  sha256 `29f4f3ffe31150c39fbf8e9fef2f149a826a4e426029ae5acd09a151e76b57eb`
  (verified both sides). Contents: glm5_next packer + verifier + spark_pack_common +
  name_map + module format header + experts-manifest tool sources.
- `build-g5n-experts-manifest` compiled: gcc 13.3.0, `-std=c11 -Wall -Wextra -Werror
  -O3` over tools/glm5_next_experts_manifest.c + runtime/spark_weightd_manifest.c +
  src/spark_ck128.c + src/spark_status.c — clean.
- dd-probe of the warm source before any heavy read: 268 MB @ 405 MB/s. All heavy ops
  under `sudo -n systemd-run --scope -q -p MemoryMax=… -p MemoryHigh=…` (sparkcap law);
  one concurrent warm stream; disk-cache purge after each heavy batch.
- Dry-plan stage3 rank0: `rank 0: 314 tensors planned` — matches the placed generation.

## Build ledger (stage3 rebuild)

| time (UTC) | op | detail |
|---|---|---|
| 2026-09-12T16:04Z | build rank12 | spark3, sparkcap MemoryMax=10240M/High=8192M, source /mnt/model-warm/glm-5.3-flash → DONE: glm5_next_stage.tp4.pp4.stage3.rank0.g5nsp, 314 tensors, 23,925,499,392 B, sha256 7e578703f75055830cc198b07028450f38d7497864e9fea63445d0ba4d84aa05 |

## Ceph stall incident (16:08-16:40Z) and pivot

The spark3 all-tensors verify + rank13 build both entered D-state on their first
checkpoint reads (0% CPU from start, partial stuck at 203,914,496 B). Probes: spark3
reads model-00062 at 2.4 MB/s and spark0 at 1.4 MB/s, sparkc reads the same file at
422 MB/s; `ceph -s` (spark0, read-only): HEALTH_WARN, 1 OSD experiencing slow
operations in BlueStore. Node-specific read-path stall, not a dataset stall. Actions:
stopped both wave scopes on spark3 (`systemctl stop run-r89f38… run-rc0a52…`),
purged spark3 disk caches (buff/cache 22G→1G), verified the killed verify emitted no
verdict (empty log — no false PASS). PIVOT: warm-reading work moved to sparkc
(276-422 MB/s re-probe) — a placement target, so its rank also places locally with
zero network hop. rank0 pack mesh-copied spark3→sparkc for verification there.
Fresh stage3 size (23,925,499,392 B) ≠ sparkf placed (23,925,520,768 B): the
compact-era layout differs from the current packer — rebuild (not re-pin) confirmed
mandatory for the whole quartet.

## Verification ledger

(filled per rank: verifier verdict, sha256, dir sha, .experts, splits)

## Placement ledger (stage3 quartet — complete)

| rank | node | file sha256 (dest-verified) | .experts sha256 | lock |
|---|---|---|---|---|
| 12 | sparkc | 7e578703f75055830cc198b07028450f38d7497864e9fea63445d0ba4d84aa05 | e6ad8a4b704eca5096470e2c7febefeb6178bc024ee0e2eff03ac7119e069998 | +i |
| 13 | sparkd | e4dafece459260c45d698cc0afd839866a46b72eaf7f4b5043c7de7c1f187563 | ef0dc3033f4e3a66a844a5a0dace15fad681f87cd593426739d211f1a664e9ba | +i |
| 14 | sparke | ffabeab8b791a3b85152411eea54f162bc65df0d6bce4fec346835e1dbd36778 | 2025e3038e31b1a766919287c090e72b32187249a5a29c535fd93b56b7812144 | +i |
| 15 | sparkf | a1392b4df4e62f604f0f75163e017a1184c7fcfc23205cd6f23c3a710a606023 | 46360b37f8ef4f84ba0af88b31f82fbd19960c9c32db4e3a17d80e450b307f49 | +i |

All four: 23,925,499,392 B, 314 entries, dir_sha256 fc9bfe454258206f0d17148043f2d964505f5ac12db44c32e8db67beb3ad5e0f
(uniform), flags=0 MTP-free, .experts manifests published via build/glm5_next_experts_manifest
(SparkWeightdManifestLoad-validated), receipt pairs written (.receipt.json user + .g5nsp.receipt.json
root, fleet convention). SPINE/EXPERT splits per rank: 290 spine entries 1,502,716,512 B |
24 expert entries 22,422,749,184 B | payload total 23,925,465,696 B. Disk-cache purge after
each placement batch (sync + drop_caches=3, sudo -n).

## Stage0-2: re-pin REVERSED — full rebuild required

The rank11 all-tensors verify (sparkb) FAILED: layer-22 (0x16) expert up_gate slab,
pack c64ba2d4766700043712427b3c0572fcf5b0eaaaa78cdc6ac14bd9e15976d80c != checkpoint
2fdafa710a12d67486a720976d13463b9d880550aa525f491757089091aa5467, plan-diff PASS.
ROOT CAUSE: the 2026-09-08 in-place fleet rewrite predates #877 (e38ebb2, 2026-09-09):
"Pair expert up and gate TP slices with down columns" — the pre-#877 packer gave a rank
only up or only gate rows and the local SwiGLU multiplied unrelated channels (75.4%
relative error at layer 4 per GLM_FLASH_HILLCLIMB 2026-09-09, which mandates
regeneration of affected TP packs; unchanged sizes cannot prove validity). Every one of
the 16 placed packs is therefore a defective-generation artifact; the receipt re-pin is
invalid by contract and all 16 ranks are rebuilt from the source through the current
packer (stage3 already done above).

SCRIPT DEFECT CAUGHT (self-inflicted, fixed): the re-pin driver's `| tail -2` swallowed
the verifier's exit code (set -e defeat — the firing-270 lesson) and wrote a PASS receipt
after a FAIL verdict. The false receipt pair + the regenerated .experts were removed from
sparkb (rank11 left unreceipted = fail-closed for weightd until the rebuild lands); the
rebuild driver verifies fail-closed (verifier rc captured; receipts written only on PASS).

## Stage0-2 rebuild ledger (12 ranks, waves of 2, on placement targets)

| time (UTC) | op | detail |
|---|---|---|
| 2026-09-12T18:20Z | probes | warm reads recovered fleet-wide after the stall window: spark1 408, spark2 511, spark3 440, spark5 469, spark6 574, spark7 503, spark9 548, sparka 504, spark4 164, spark8 513 MB/s |
| 2026-09-12T18:2xZ | dry-plan | stage0 272 tensors, stage1 287 — generation match |
| 2026-09-12T18:2xZ | wave 1 | rank0 (spark0) + rank1 (spark1), stage0 L0+11 owns-embedding — running |

## Final placement matrix (16/16, sweep-verified 2026-09-12)

All 16 packs: MTP-free (flags=0), directory_offset 512, header file_bytes == actual,
chattr +i locked, receipt pair (`.receipt.json` + `.g5nsp.receipt.json`) whose
output_sha256 equals the independently recomputed pack sha, `.experts` manifest
regenerated and recorded, verify = plan-diff + all-tensors round-trip PASS vs
/mnt/model-warm/glm-5.3-flash at main 6c6caf52. dir_sha256 uniform per stage:
stage0 f579fdf7…, stage1 4b8caff6…, stage2 8df28e2c…, stage3 fc9bfe45….

| rank | node | stage/layers | entries | bytes | output sha256 |
|---|---|---|---|---|---|
| 0 | spark0 | 0, 0+11 | 272 | 16,503,376,128 | 7007d579adfa727eb10bb3113efb0b4d90a79340a411ba238e6f604ec516b292 |
| 1 | spark1 | 0, 0+11 | 272 | 16,503,376,128 | 490f39e70f4e91f552f2b190e31b436f460f94132d000d26b4120eca3445e107 |
| 2 | spark2 | 0, 0+11 | 272 | 16,503,376,128 | b49959909d97905c6af7a3fe1a38a844508b646c399137b624f9c6bfd501afba |
| 3 | spark3 | 0, 0+11 | 272 | 16,503,376,128 | 55d3feecd03b7102b008e48f2eb67900d579f2a3e092ab704d9fb012a5a11f79 |
| 4 | spark4 | 1, 11+11 | 287 | 21,651,182,336 | 33df6607a46de0ecf4dce239f79c468a414ee9f84c985df7ab343a5cbe20d5b5 |
| 5 | spark5 | 1, 11+11 | 287 | 21,651,182,336 | 42c035f010c23ee3dc1385bd09b73d4e8cd1980da1a962fd4dc8a14a2a9c407b |
| 6 | spark6 | 1, 11+11 | 287 | 21,651,182,336 | be0eb252d289c287142ae21716637a210645136638d12553185854df849041c9 |
| 7 | spark7 | 1, 11+11 | 287 | 21,651,182,336 | 43d19d4f18017d56c13f9aab172368deeb59c7e0e70027434074297aa5b2cdab |
| 8 | spark8 | 2, 22+11 | 287 | 21,651,182,336 | f804adae3db324172ae609e2664a0ebeabb89429e9aa922ed911e6548f15b4e0 |
| 9 | spark9 | 2, 22+11 | 287 | 21,651,182,336 | beb3a24cb1c252c896dcb85f815b5a506133eccf5dad233e70a32d6552f2202c |
| 10 | sparka | 2, 22+11 | 287 | 21,651,182,336 | 51c36fd3992558dc755186f4016b6edd4de62e11fce86b589680106ce8268e64 |
| 11 | sparkb | 2, 22+11 | 287 | 21,651,182,336 | ca51deee64fcfe889a5916832398a070a06ddad2b977160d54fb5a4d1f671425 |
| 12 | sparkc | 3, 33+12 | 314 | 23,925,499,392 | 7e578703f75055830cc198b07028450f38d7497864e9fea63445d0ba4d84aa05 |
| 13 | sparkd | 3, 33+12 | 314 | 23,925,499,392 | e4dafece459260c45d698cc0afd839866a46b72eaf7f4b5043c7de7c1f187563 |
| 14 | sparke | 3, 33+12 | 314 | 23,925,499,392 | ffabeab8b791a3b85152411eea54f162bc65df0d6bce4fec346835e1dbd36778 |
| 15 | sparkf | 3, 33+12 | 314 | 23,925,499,392 | a1392b4df4e62f604f0f75163e017a1184c7fcfc23205cd6f23c3a710a606023 |

Relay notes: rank3 built+verified on spark9, mesh-shipped to spark3 (its build path
stalled twice — serving co-tenancy + OSD double-duty); rank6 built+verified on spark9
after spark6's build stalled twice. Both dest-sha verified before the receipt was
written. Sizes are byte-identical to the replaced generation (the #877 pairing
redistributes up/gate rows within unchanged shapes) — which is exactly why the old
packs' unchanged sizes could not prove validity.

## Cleanup

Staging build dirs (~270 GB: sparkc ~190 GB, spark9/spark3/spark6/spark8 transient
copies), per-node driver scripts, and /tmp tool tarballs removed after the sweep;
wave systemd scopes stopped; disk caches purged per batch throughout.

## Blockers / watch items

- The warm-ceph stall windows recurred three times (spark3 ×2, spark6 ×1, correlated
  with `ceph` reporting 1 OSD with slow BlueStore ops); all self-resolved. The
  read-probe-before-wave + relay-build mitigations carried the campaign through.
- Serving engines on the 16 nodes still hold the old pack inodes until their next
  restart (no daemon interaction was performed, per law); the next engine restart on
  each node picks up the corrected bytes. Weightd will fail-closed on any node whose
  engine state predates the swap until remapped.
- The 2026-09-08 in-place rewrite's origin and intent are unknown to this lane; its
  output was pre-#877-pairing bytes. Other arms built before 2026-09-09 may carry the
  same defect class and deserve the same probe (out of scope here).
