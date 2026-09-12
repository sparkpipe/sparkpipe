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
