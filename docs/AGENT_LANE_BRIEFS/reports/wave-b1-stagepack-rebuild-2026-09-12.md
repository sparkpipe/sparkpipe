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

## Placement ledger

(filled per rank: source/dest sha256, lock state, purge)

## Stage0-2 re-pin ledger

(filled per stage/rank)
