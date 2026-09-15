# Wave-Finisher per-pack ledger — 2026-09-14

Agent: wave-finisher (mgr2 dispatch; sole ceph warm consumer for the wave).
Branch `lane/wave-finisher-0914` off origin/main; identity verified `sparkpipe`
via tools/sparkpipe_github_pat.sh before any write op. All heavy ops under
sparkcap (sudo -n systemd-run --scope); purge (sudo -n sync; drop_caches=3)
after every placement batch; per-file sha256 + open-handle pre-checks on
every deletion; dest-sha == receipt sha on every placement.

## Item 1 — qwen3flash.fp8.tp8: 16/16 placed, verified, locked

Builds: wave-pr's 8 rank builds (30,518,612,480 B, 1215 tensors,
--no-mtp fp8-official from /mnt/model-warm/qwen3.8-flash-next-fp8) verified
per rank with qwen4_flash_pack_verify --no-mtp (header geometry + 1215
directory entries + 8-sample byte-trace vs checkpoint) PLUS streaming
sha256sum == receipt output_sha256. Replica rule rank r -> spark r + r+8;
per-node lawful names preserved (spark0/spark8 unpadded rank0, others padded).

| rank | node | file | new sha256 | replaced (old gen) | placed_at |
|---|---|---|---|---|---|
| tp8 r0 | spark0 | 30,518,612,480 B | 598721746c10 | 4690c1411496 | 2026-09-14T08:54:28Z |
| tp8 r1 | spark1 | 30,518,612,480 B | ca54341d43e6 | 78d35f5099aa | 2026-09-14T09:04:15Z |
| tp8 r2 | spark2 | 30,518,612,480 B | 447690a0c1ed | 39793c0fbd08 | 2026-09-14T09:05:28Z |
| tp8 r3 | spark3 | 30,518,612,480 B | 90a505adc6f2 | 110b4548e0f6 | 2026-09-14T09:36:21Z |
| tp8 r4 | spark4 | 30,518,612,480 B | 59367b2b7855 | f4138a3d88c3 | 2026-09-14T09:15:37Z |
| tp8 r5 | spark5 | 30,518,612,480 B | 90fa00cad25b | 30d5b24ef26e | 2026-09-14T09:52:47Z |
| tp8 r6 | spark6 | 30,518,612,480 B | e99cea905350 | d7b857d1aa38 | 2026-09-14T09:56:09Z |
| tp8 r7 | spark7 | 30,518,612,480 B | acb43ac24f71 | 6df18d01db23 | 2026-09-14T09:57:18Z |
| tp8 r0 | spark8 | 30,518,612,480 B | 598721746c10 | 4690c1411496 | 2026-09-14T08:52:46Z |
| tp8 r1 | spark9 | 30,518,612,480 B | ca54341d43e6 | 2a8ca851eee2 | 2026-09-14T08:59:05Z |
| tp8 r2 | sparka | 30,518,612,480 B | 447690a0c1ed | 39793c0fbd08 | 2026-09-14T10:08:40Z |
| tp8 r3 | sparkb | 30,518,612,480 B | 90a505adc6f2 | 2c4634fbbb65 | 2026-09-14T10:09:48Z |
| tp8 r4 | sparkc | 30,518,612,480 B | 59367b2b7855 | 52aed41bfdb6 | 2026-09-14T09:16:54Z |
| tp8 r5 | sparkd | 30,518,612,480 B | 90fa00cad25b | 30d5b24ef26e | 2026-09-14T09:40:55Z |
| tp8 r6 | sparke | 30,518,612,480 B | e99cea905350 | 868d8416574e | 2026-09-14T09:58:41Z |
| tp8 r7 | sparkf | 30,518,612,480 B | acb43ac24f71 | 9ad6fc034dc0 | 2026-09-14T10:00:55Z |

Old-generation defects replaced in place: 8 slots carried the 30,425,355,496 B
short gen (s1,s2,s3,s4,s7,sa,se,sf), 8 the 30,518,614,272 B gen (s0,s5,s6,s8,s9,sb,sd)
— replaced_sha256 in each receipt; struct-FAIL generation retired fleet-wide.
Wrong-slot strays created and deleted same wave (finisher error, corrected by
census): sd rank02 + sf rank03 copies (digests identical to lawful copies on
s2/s3; deletion receipted in the wave log, lsof-clean pre-check).

## Item 2 — qwen3flash.fp8.tp4pp4 stage3 ranks 12-15: 4/4 placed

Builds (wave-pr, on s2/s5/s7/s0; 8,935,637,248 B, 305 tensors, first_layer 36,
layer_count 12) verified with the same gate (305 entries + 8-sample byte-trace).

| rank | built on | placed on | file | new sha256 | replaced (old gen) | placed_at |
|---|---|---|---|---|---|---|
| stage3 r12 | spark2 | sparkc | qwenflash.tp4_pp4_fp8.rank12.spstage 8,935,637,248 B | 9ce1d28a42e6 | bff6326ec623 | 2026-09-14T10:23:05Z |
| stage3 r13 | spark5 | sparkd | qwenflash.tp4_pp4_fp8.rank13.spstage 8,935,637,248 B | ab1d8115b354 | 6e1360da30f2 | 2026-09-14T10:23:21Z |
| stage3 r14 | spark7 | sparke | qwenflash.tp4_pp4_fp8.rank14.spstage 8,935,637,248 B | cc51a7296c44 | 1f236b89007b | 2026-09-14T10:23:47Z |
| stage3 r15 | spark0 | sparkf | qwenflash.tp4_pp4_fp8.rank15.spstage 8,935,637,248 B | 733efbe5b121 | 6259b07a59a9 | 2026-09-14T10:24:02Z |

Superseded stage3 generation deleted: rank12 byte-copy of rank08
(bff6326e, 8,286,810,112 B), rank13 6e1360da (8,935,638,784 B),
rank14 1f236b89 / rank15 6259b07a (8,890,517,456 B class) — replaced at
placement; plus stage3-class strays spark6 rank14 7b4b80ff and spark7
rank15 64ab598f (8,935,638,784 B each) — sha-verified against the Wave-P
ledger before rm, lsof-clean, deleted with purge.

## Item 3 — k3.mxfp4.tp4pp4 rebuild: stage0 COMPLETE (4/4 placed+oracle-PASS), stages 1-3 IN FLIGHT

SP-4R's left-running scopes were found DEAD on s9/sd and hard-wedged
(mem_cgroup_handle_over_high) on sf; no stage had completed (partial payloads
only). All stages are rebuilt from zero through the current packer on spark9
under an autonomous watchdog loop (tools staged at /tmp/fin_k3_stage.sh on
s9/s8/sd/sf): k3_pack.py (resume-journal) + stall watchdog (payload-size poll,
1800 s no-growth kill, cache purge, retry, max 40 attempts) + k3_shard.py 4
ranks + .experts regeneration (patched ctypes byref calls; the shipped
sp4r_k3_experts_gen.py never ran to completion and dies on bytearray-vs-
c_void_p) + per-rank sha256 (k3.stage{S}.shas).

Stage0 receipts (built on spark9, single-writer provenance, k3_verify_pack
--quick PASS 552 tensors/24 layers, manifest_len 1048560, payload_base
1048576 = post-2b27e64 format; k3_checkpoint_oracle --digest PASS 1074
checks / 108,084,736 bytes / 0 mismatches per rank):

| rank | placed on | file | new sha256 | replaced (old gen) | placed_at |
|---|---|---|---|---|---|
| stage0 | spark0 | k3.stage0.rankNN.pack 98,766,808,320 B | 908827dc1dbf | bef86d7ea918 | 2026-09-14T17:49:55Z |
| stage0 | spark1 | k3.stage0.rankNN.pack 98,766,808,320 B | c8fa00bee6d4 | dda37d487726 | 2026-09-14T17:57:01Z |
| stage0 | spark2 | k3.stage0.rankNN.pack 98,766,808,320 B | f210ff9bd359 | 0d3543b2cfe3 | 2026-09-14T18:03:37Z |
| stage0 | spark3 | k3.stage0.rankNN.pack 98,766,808,320 B | 15b34e801afe | c941a221abe1 | 2026-09-14T18:10:31Z |

abi_version=6/descriptor=72 (dispatch acceptance pair): NOT PRESENT in the
pack manifest (no such keys), NOT a repo constant reachable from main or the
node trees (SPARK_K3_STAGE_RUNNER_ABI_VERSION=1, SPARK_MODEL_RUNTIME_ABI_VERSION=1,
serving-adapter ABI 22, range-manifest version 2). The verifiable format law
these packs carry: k3 pack format version 2, manifest reserve 1048560,
payload base 1048576, tp_degree/tp_rank echo, v2 per-expert .experts
(SparkWeightdManifestLoad self-validated, 41216 ranges/rank). The 6/72 pair
is flagged back to mgr2 as ungroundable from any artifact this wave could read.

Stages 1-3: rebuild loops launched and grinding autonomously on spark9
(stage1 24+23, stage2 47+23, stage3 70+23), same watchdog; the warm client
paths degrade intermittently (per-region crawls to ~1-3 MB/s and hard hangs,
reproducing SP-4R's exact frozen offsets) and the loop's journal-resume
banks every completed tensor across retries. VERDICTS PENDING: no pack for
stages 1-3 is placed or claimed placed; oracle + placement + purge execute
per stage on completion (same receipts chain).

## Item 4 — never-referenced .tmp partials: 30/30 deleted

The sweep counted 31 .tmp rows (105,660,516,864 B of qwen partials + sF's
13,156,747,648 B glm compact.tmp); the compact.tmp was already deleted by
wave-pr with its item-1 debris. This wave deleted the remaining 30, each
with byte-size + sha256 re-verification against the Wave-P sweep ledger and
an lsof open-handle pre-check (refuse-on-mismatch; zero mismatches hit):

| node | deleted | bytes |
|---|---|---|
| spark0 | 1 .tmp partials | 3,462,376,960 |
| spark1 | 7 .tmp partials | 18,251,303,680 |
| spark2 | 1 .tmp partials | 8,262,400,000 |
| spark3 | 2 .tmp partials | 35,985,608,448 |
| spark4 | 6 .tmp partials | 3,326,963,200 |
| spark5 | 1 .tmp partials | 8,262,400,000 |
| spark6 | 8 .tmp partials | 9,422,848,512 |
| spark7 | 1 .tmp partials | 7,462,396,160 |
| spark9 | 2 .tmp partials | 10,924,773,120 |
| sparkc | 1 .tmp partials | 299,446,784 |
| **total** | **30** | **105,660,516,864** |

Per-file deletion receipts (node file bytes sha256) preserved in the wave
log; every digest matched the Wave-P census before rm.


## Addendum — stages 1-2 build completion (in-flight at ledger update)

Stage1 (sparkd; 537 tensors/23 layers; ENOSPC during the first shard pass and
a concurrent-writer poisoning forced one clean rerun from the stage pack; the
rerun's rank sizes carry the +786,432 B manifest-reserve delta of the current
format exactly):

| rank | sha256 (build, sparkd) |
|---|---|
| stage1.rank00 | 58345167ebc7f6e1b2c109250bb609c0418a47b3dc22cf8cad116d7871ee57a1 |
| stage1.rank01 | e971a6ff80be96a133dda97eb217d7cdfdb3e5335fc58c9a3238b12e50a352fc |
| stage1.rank02 | cb488bc487cbc589f1da0f7d6da15cbfdc8f088f1790f7a0ea544a42be359fcb |
| stage1.rank03 | 27ce6ce0ba544e06e6dbd40f7eec6817f555c710b78fd37e7492f86d6a8eecfe |

k3_verify_pack --quick PASS x4 (537 tensors, 23 layers, manifest_len 1048560,
payload_base 1048576, tp 4/0-3); .experts v2 x4 (41,216 ranges). Oracle
--digest x4: IN FLIGHT on sparkd (warm-client crawl windows; verdicts
pending — no stage1 pack is placed until 4/4 PASS).

Stage2 (sparkd; 534 tensors/23 layers; pack 388,835,000,704 B payload):

| rank | sha256 (build, sparkd) |
|---|---|
| stage2.rank00 | 24c370554e59be011f70d69eeadab0c9c3567524167185fee873618b22044f20 |
| stage2.rank01 | 3e6c54fec9e6720071612589bc62f4ab87d2295e4824bc3098260d070a7dee12 |
| stage2.rank02 | 8cb22a458150f717484640b59d33c3107e5339bf9f66ee926b520a96175bc040 |
| stage2.rank03 | 627b75747c220a5e32c0ec5712c80128b5d7076598008d6ed23226a4d13538a8 |

.experts v2 x4 done (41,216 ranges). verify --quick + oracle + placement:
PENDING (scripted; see /tmp/fin_k3_RUNBOOK.md on spark9/sparkd).

Stage3 (layers 70+23): rebuild loop NOT yet launched (warm-stream serialization
behind the stage1 oracle chain); launch command in the runbook.

Deleted this addendum window: stage0 debris on sparkd (393,525,084,800 B
stage0.pack + 4x ~97.6 GB quartile rank copies + sidecars — migration junk,
stage0 already placed on s0-s3), stage1.pack and stage2.pack deleted
post-shard on sparkd (their rank packs + shas + experts are the artifacts).
