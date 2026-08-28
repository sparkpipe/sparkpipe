# Qwen 3.8 Flash S5/S6 lane report — 2026-08-29 (session: pack-complete under operator pause)

Worktree /tmp/lane-qwenflash-s6, branch lane/qwen-flash-s6 (off
origin/lane/qwen-flash @ 4c3ff91). Build nodes spark4/5 (spark6/7 lost to
a ceph wedge mid-session, see INCIDENT). The mission's S5/S6 live-serving
milestones are STAGED but NOT RUN: the operator pause directive landed at
pack-complete ("hold; housecleaning sprint owns the fleet; wave resumes on
my word"). Everything up to the one-command wave launch is done and
verified.

## Session decisions that changed the plan (all coordinator-ruled)

1. QUANT POLICY (commit 8e0aca7): the v3 packs' fp8 experts were
   SELF-quantized -> policy-void for serving. Consumed two ways:
   (a) the OFFICIAL publisher FP8 release
   (Qwen/Qwen3.8-Flash-Next-FP8, commit 970c569adaca6b35532111fd6b27351b2baefe50,
   131 shards, 172.8 GiB) was fetched and sha-verified (144/144 files vs
   HF LFS oids, provenance.json at
   /mnt/model-warm/packbuild/qwen3.8-flash-next-fp8/provenance.json) as
   the approved FAST-FOLLOW source; (b) per the coordinator's primary
   directive, packs_v4 are BF16 REPACKAGE-ONLY from the bf16 warm source.
2. TOPOLOGY: 16-rank TP4xPP4 primary (supersedes the 4-rank whole-stack
   framing). Stage ROTATION puts the PLE-heavy stage 0 on spark4-7 and
   the light stage 1 on spark0-3 so the active-MDS hosts spark2/3 carry
   only ~22 GiB-class ranks (coordinator-approved class, glm5_next
   precedent): chain 0-3=stage 0 on spark4-7, 4-7=stage 1 on spark0-3,
   8-11=stage 2 on spark8-b, 12-15=stage 3 on sparkc-f.

## Latent S5-staging bugs found and fixed BEFORE any launch

1. The staged deploy_v3 adapter hardwired a TP4xPP3 12-rank geometry
   (SERVING_STAGE_COUNT 12): a 4-node launch would have served a
   16-layer slice with NO final head. Now: HYBRID_TP_PP descriptor
   (stage_count 16 = node count per runtime/model_resident_deployment.c:451,
   parallel_group_size 4, per-node layer counts 12x16 summing per group
   to 48); module env STAGE_COUNT=pp_stage_count(4), STAGE_INDEX=PP stage.
2. The old launcher's teardown was `pgrep -f sparkpipe_model_residentd.*--deployment`
   — on shared nodes that pattern MATCHES SIBLING LANES' residentds
   (glm5_next runs exactly that command line). Replaced with a pid-file
   TERM-only design everywhere (spawn-recorded pids, zero pattern matching).
3. The adapter never set the module TP collective env; the residentd
   environment must carry SPARK_QWEN4_FLASH_TP_{DEGREE,RANK} +
   STAGE_TP_{BACKEND_PATH,IDENTIFIER,PORT_BASE,HOSTS,LOCAL_HOST} — the
   fleet launcher exports them per rank (TP port base 66840; PP hidden
   transport base 66640; control tcp 18180+chainpos).

## Evidence (commands + raw output, abridged per lane format)

bf16 grouped-expert path (NEW kernel surface — synth packs were fp8, so
this branch had never executed):

```
# synth --bf16 (bf16 experts over the F32B128-natural kinds ONLY; blanket
# --bf16 is invalid - the loader rejects bf16 over f32-natural kinds:
# "pack_entry_invalid kind=19" = A_log; fixed in the synth tool)
qwen4_flash_validation check=module_decode_vs_prefill elements=10240 relative_l2=0 cosine=1 max_abs=0
qwen4_flash_validation check=module_determinism bit_exact=1
qwen4_flash_validation PASS
```

Module publish (whole-stack TP4 standalone ladder, v3 pack — validation
use of policy-void weights is sanctioned; the serving packs are v4):

```
make publish ... STAGE_PACK_PATH=packs_v3/...rank0... TP_DEGREE=4 TP_STANDALONE=1
qwen4_flash_validation PASS
```

packs_v4 (16 ranks, bf16 repackage; sizes from dry-run then receipts):

```
s0 (0+12, PLE+embed): 4 x 38.95 GiB, 311 entries
s1 (12+12):           4 x 14.75 GiB, 300 entries
s2 (24+12):           4 x 14.75 GiB, 300 entries
s3 (36+12, head+MTP): 4 x 16.62 GiB, 336 entries
PASS qwen4_flash_v4.s0r0..s3r3 (16/16): header geometry, directory entries,
     12 byte-traced samples each, receipt=verified
```

(The verifier needed a new branch: bf16 experts at tp>1 crashed the trace
— the generic expected-read ignored the expert shard; fixed with the
packer-mirroring expert-slab byte compare. Rank-0-only strides had masked
it, same shape as the P2 verifier bug.)

Fleet placement (sha256 prefixes, per-rank on its node):

```
s0r0 b11dd97e96cb761e  s0r1 ee9b7566b1b1bc28  s0r2 951d261412c3b732  s0r3 5c7b02435bd381b4
s1r0 742c671f71f83360  s1r1 d710c4e5a35b5c38  s1r2 549116d21bda7768  s1r3 7fc2c2fe1702dc93
s2r0 4f875d1d11d9fbd7  s2r1 4f3b6ab1eb8c6b00  s2r2 b58bab72d588e4d2  s2r3 71d1c95a6b3c3c17
s3r0 a03cda5026d82b65  s3r1 c6ac7d0d2ccbd2f9  s3r2 0950039a70413376  s3r3 397da1a3acca78ba
```

Serving stack: module PUBLISHED, adapter .so (255,560 B) and
model_driver.so (4,846,008 B — larger than S5's, the bf16 kernel branches
embedded) built; deploy_v4 (bin/lib/config/deployment.json/launch_table/
LAUNCH-STATE.md/smoke batch/B1-32K batch) staged IDENTICALLY on all 16
nodes. The wave is one command:

```
tools/qwen4_flash_fleet16_launch.sh --table <deploy_v4>/launch_table.json
```

## INCIDENT (coordinator has it; 3rd instance of the class)

spark6+spark7 ceph clients WEDGED for bulk reads on /mnt/model-warm:
both stage-0 pack builds sat D-state 16+ min at the IDENTICAL
346,907,904-byte output offset (the first PLE ngram-shard read), 0 bytes
growth/30s, a 128MB dd on the shard also hung >90s. TERM'd my two packers
(own pids; D-state), rerouted all 10 remaining builds to spark4/5
(healthy, full speed). Zero fleet impact (residentds read local NVMe).
Pattern now spark5(P2)+spark6+spark7 — all bulk-read class under
multi-node concurrent reads of the same checkpoint; my MDS-pressure
theory is recorded as leading; reroute-around is now doctrine per the
coordinator.

Also fixed en route: my own `pkill -f pack_verify` caught its own ssh
wrapper on spark4 (the exact fuzzy-kill class the rules warn about —
no harm, one killed wrapper; noted honestly).

## Honest negatives

  * No live serving number: the operator pause stopped the session at
    pack-complete. The 16-rank wave, B1 smoke, token-stream hash, B1-32K
    perf cell, and COMPSEC-17 remain TODO on the coordinator's word.
  * The bf16 TILE-path grouped-expert branch (rows>=16) is compiled but
    UNEXERCISED (the tiers run <=8 rows). B>=16 cells will first-exercise
    it; the scalar path is proven bit-exact.
  * Per-stage whole-stack standalone on the REAL slice packs was not run
    (the synth bf16 ladder + publish ladder + byte-trace receipts cover
    the composition; if the coordinator wants it pre-wave it is one
    command per stage with the debug validator).
  * The old /tmp/lane-qwenflash staging on spark4 has NO .git (prior
    session synced files, not a checkout — unreproducible staging);
    this session staged everything from git bundles at /tmp/lane-qwenflash-s6
    on all four build nodes. Flagged for the staging agent.
  * v3 packs (packs_v3, 4x56.06 GiB) are policy-void for serving and
    marked validation-only; kept on spark4-7 for the publish ladder.

## INTEGRATION REQUEST

  1. (coordinator) Wave window arbitration per the pause: sequenced after
     housecleaning + either the MacStudio MDS move (spark2/3 clear) or
     glm5_next's TERM; spark8's p1a occupant must be TERM'd first too
     (stage-2 ranks 8-11 include spark8; ~78.6G occupant + 22G rank would
     exceed the envelope).
  2. (coordinator) spark6/7 ceph client wedge — sysadmin queued; note
     both nodes still host their v4 packs fine (local NVMe).
  3. (coordinator) Fast-follow: official-FP8 repack (archive verified at
     /mnt/model-warm/packbuild/qwen3.8-flash-next-fp8): scale-plane
     mapping (publisher BF16 per-128x128-tile vs our F32B128 wire) is
     the one explicitly-verified step; ~10.8G/rank expert traffic halving
     at high batch.
  4. (coordinator, FYI) examples/ firmware description is outside the
     lane write set (unchanged standing request from S5).
  5. (driver lane, FYI) The qwen4_flash hidden boundary is the 4H stream
     vector (10240 bf16) — PP boundaries carry it stage-to-stage; the
     transport shim in the adapter handles capture.

## Commits (this session)

  * ca6eaaf whole-stack TP adapter mode + one-wave pid-file launcher (the
    pre-16-rank pivot; superseded within the session by 9da8bc3's PP4).
  * 9da8bc3 (pushed 9da8bc37) 16-rank TP4xPP4 retarget: bf16 repackage
    packer, bf16 grouped-expert kernel acceptance, HYBRID_TP_PP adapter.
  * 7d35658 B1 exact-32K batch + verifier bf16-expert trace fix + synth
    bf16-experts-only fix (this commit).
  * Ratchets: 192135 -> 192989 -> 193155 (justifications in-line).

## Next (on the coordinator's word)

  1. Reserve all 16 nodes; ONE-WAVE launch via the fleet launcher;
     LAUNCH-STATE.md already documents layout + teardown.
  2. Ready-lines x16; B1 smoke via sparkpipe_model_batch (api stopped per
     the one-client rule); token-stream hash; then the B1 exact-32K cell
     (batch staged: 32640+128 = 32768 = declared max positions).
  3. QUALITY GATE: COMPSEC-17 through the live /v1 before "usable";
     records to qualification/ds4_eval/runs/qwen4flash-<deployment>-<date>/.
  4. Fast-follow: official-FP8 repack + perf comparison.
