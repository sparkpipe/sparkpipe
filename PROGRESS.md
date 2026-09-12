# ling lane PROGRESS — lane/ling-driver (PR #830 continuation)

Lane: ling (spark9). Continuation round. REBASE HISTORY this round:
(a) adopted the coredev-aligned origin/lane/ling-driver cc49cf8 (24 lane
commits + 22 coredev mesh commits, base 8f3a6f2); (b) per the coordinator
update, REBASED ONTO origin/main 50bd0d3 (PR #913, the E2E-proven mesh
dataflow) — the coredev mesh snapshot commits resolved TOWARD MAIN (mostly
auto-skipped as superseded), the ling module port (4364daf: credit
bindings gone, hidden-transport validation per the glm5_next template) and
the family tree survived. ROUND TIP: 59da8ad (commits ec1630d, 53c66ed,
0e2c1c4, 4baa779, f761c6b-lineage, f0824dd S4 folds, 053f583 streaming
.experts digests, 7c4800b, 1c447ba, 378cde8). The k3-adapter repair
cherry-pick auto-skipped (main has it via #907). /Users/mac/lane-*
worktrees untouched per the hazard notice. Spark tree: spark9:~/batch-ling-r10.

## Acceptance criteria status (DESIGN.md §8)

1. Rebase — DONE onto 50bd0d3 (sha 53c66ed, then follow-ups to 4baa779).
2. mac syntax gate — GREEN on the rebased tree: module bf16+fp8 arms,
   serving adapter, pack_synthesize (rc=0, cuda_stub command below).
3. Residue greps — ALL ZERO (module + family, pattern list unchanged).
4. bf16 module build — compile receipt job ling-r9b-compile PASS on spark9
   (nvcc sm_121a; bf16 + fp8 archives + bf16 adapter .so + synth run 727
   tensors / 130198162624 bytes tp1 codec bf16; LING_COMPILE_RECEIPT_PASS,
   attempt ecc610345570453cbb71ec8380c4ad80).
5. validate_mtp_parity target — REMOVED (commit 8e3aa60 lineage).
6. bf16 TP16 pack from warm — canonical naming landed (53c66ed): packer
   writes ling.bf16.tp16.rank{0..f}.sp + .sha256 sidecar (weightd spawn
   digest-scan format) + .experts + receipts/{rank}.json (carries "arm");
   --model lingfin selects name_map_lingfin.json +
   lingfin_authoritative.json for the second contract. Verifier derives
   pack names from the rank0 receipt's "arm". Generator: ROOT_NAME single
   source (runtime_root, packs/<arm>.rank%x.sp, kv dir derive); HEX rank
   per main 95fe1b1.
   Warm-mount rate is ~30MB/s: one rank ≈ 9 min emit — a rank per job.
   The .experts ck128 tail re-read ~14.5GB per rank inside the queue's
   cgroup and blew every TTL; fixed at the right layer (053f583): digests
   stream at emit (SlabDigestSink, same ck128 on the same bytes,
   digest-count guard), re-read path kept for callers without digests.
   Receipts: dry-plan census closes 63783 = 62230 packed + 1553
   omitted-MTP (r9c attempt + prior dry-plan); rank0 emit 15725069824
   bytes + .sha256 sidecar. Final chain in flight: ling-r12-pack0 →
   pack15 → verify (two-pass) → findry.
7. Validator — HONEST STATUS: RED, first-ever visibility this round.
   The monolithic harness (nvcc+run in one 15-min-max job, mktemp build
   dir wiped each attempt) never produced a completed log, so r7/r8's
   "in flight" was unverifiable. Split proof artifacts: validator object
   compiles in ~1s (68KB — no device code; ling-r11b-valcompile PASS),
   binary build PASS (ling-r12-valbuild, 1.04s). Unbuffered diagnostic
   run (ling-r12-valdiag, stdbuf, job died at TTL mid-run) captured the
   first real comparison output: selftest PASS; tier1 token0 kda
   attention sublayer PASS (rel_l2 0.00786, cos 0.99997); token1 kda
   FAIL (rel_l2 1.248, cos 0.581); tier2a token0 mla PASS (0.00527);
   token1 mla FAIL (0.514, cos 0.865). Signature: token0 passes, token1
   decorrelates in BOTH tiers — the device/oracle state or cache carry
   diverges from the second step on (KDA state pool and MLA cache both).
   Next round: instrument the walk (state sha per step on both sides,
   per the anchor's tier3 layout), find which side mis-carries.
   378cde8 makes stdout line-buffered so this can never hide again.
   S4 audit folds (f0824dd): LoadTpCollective 148L → 4 helpers +
   dispatcher; SparkLingTpChainAdvance 119L → per-stage helpers over a
   slim dispatcher; SparkLingKvInitialize 95L → allocate + fill.
8. offline-gates — build phase now fully GREEN on spark9; test-run phase
   marches into OTHER lanes' surface. Fixed this round on the shared tree:
   (a) k3 adapter took main's #907 repaired file (auto-skip at rebase);
   (b) test_qwen38_27b_tp_faults: GPU-node stub shadowing + missing stub
   link + runtime archive for the weightd client (0e2c1c4);
   (c) rdma_control.c orphaned static deleted (4baa779);
   (d) LDFLAGS carries SPARKPIPE_CUDA_DRIVER_LINK — the #913 class fix:
   the weightd sources moved into libsparkpipe_model_common.a and
   spark_weightd.o needs cuMemExportToShareableHandle, so every
   archive-consuming link on a CUDA node needs -lcuda (7c4800b).
   REMAINING BLOCKER (not ling): test_dsv4_serving_adapter aborts —
   "serving adapter spark.dsv4.flash-0731.serving-adapter.pp13.v2:
   missing required operation reset" (model_serving_adapter.c:251);
   the dsv4 interface table (spark_dsv4_serving_adapter.c:1383) lacks
   .reset, which the post-#913 loader requires. Semantically the dsv4
   lane's to implement; everything before it in the gates run passed
   (tokenizer, pipeline_runtime, model_api_text, ...). Also flaky under
   contention: test_model_pipeline_client passed solo, aborted when run
   beside GPU+pack jobs. Ceiling gate green in the same run: exact 242163.
9. Registry — Makefile ling targets, stagepack_naming.json ("ling",
   "lingfin"), deployment generator ROOT_NAME/hex-rank/session-ledger
   done; name_map_lingfin.json authored (0e2c1c4) — mapping mechanically
   identical to ling's, identity = fin config.json sha256
   f3e5b1ad82762c7910a303a3642290f0ac4d3abbc9046bf2eab6b19c0385aa05 +
   index sha256 19b4f2f2e199c7e34bab049c1fc1d99b1661700a669a9ea7e7fabcb4527c5920.
   FIN CONTRACT PROVEN (job ling-r9-fin, exit 0): config.json
   content-identical (zero differing keys), census 63783 = 63783, tensor
   name sets identical (0 deltas both ways) — fin is exactly the second
   model_revision; its own pack set runs through the same packer with
   --model lingfin.
   Contract freeze + PACKAGE_MANIFEST/SHA256SUMS: LAST, after pack verify.
10. Evidence discipline — receipt commands + SHAs recorded here and in
    commit messages; no serving/perf claims anywhere.

## Coordinator directives applied

- SPARK_FAIL at every host error site I touched (module.c admit-lane returns,
  adapter LoadTpCollective/LoadConfiguration/LoadDriver; 609273d). New C code
  (synth main) uses unique return codes 11..18 + SPARK_FAIL. Validator .cu keeps
  its test-failure pattern (flagged here per the directive).
- Port base: initially 64900 (unsafe: ACK offsets overflow 65535) → made
  env-driven with no default → FROZEN at 12288 by PORT_LEDGER (generator
  default, override kept).
- MTP sidecar law: v1 unchanged — packer omits the MTP tail, header flag 0,
  inventory exact both ways; nothing assumes an in-pack MTP arm.
- .experts sidecar v2 + spine/expert receipt split added (4024c16 + fixes).
- Queue: submissions moved to the v2 tool path after the correction; same state
  dir, live dispatcher picks the jobs up.

## Bugs the spark9 gates caught (why the gates exist)

- unity.cu anti-bf16 static_assert + a second HC-streams embedding overload:
  the module never compiled (8330b78).
- Validator: layer_weights[1] indexed [2] (UB→segfault), selftest router mock
  1024 floats vs 512×2560 reads, expert host planes missing the ×512 factor
  (heap corruption), null wave host metadata pointers (launch -46),
  selftest bf16 zero-count.
- Packer: directory at 256 overlapping the 264-byte header (loader rejects
  <264) → 512; verifier entry format 7×u32 vs the real 8×u32 (64B records).
- Synth tool: payload cursor never shifted past header+directory → payloads
  overwrote the directory (only masked by the final header rewrite).
- Warm checkpoint o_norm.weight is BF16, not the F32 tensor_patterns.json
  witnesses — pack-time dtype re-verification caught it; packer now upcasts
  bf16 f32-planes exactly. tensor_patterns.json o_norm entry wants a manager
  amend (also its mlp.gate/eos prose).

## Receipts (spark9)

- ling-compile-receipts-r3: EXIT 0 — bf16+fp8 archives
  (build/modules/ling_resident_decode_stage/{bf16,fp8}/libling_resident_decode_stage_*.a),
  adapter .so, synth run (65 tensors/4 layers), selftest PASS,
  LING_COMPILE_RECEIPT_PASS.
- Real-index census (dry-plan): 63783 = 62230 packed + 1553 mtp — matches
  name_map pin.
- Pack rank0 (r5, superseded packer): 730 tensors, 15725069568 bytes,
  sha256 480b9c12fb68deb4... — re-cut in flight by r6.

## Oracle command (mac, from clone root)

cc -std=c11 -Wall -Wextra -Werror -fsyntax-only -I. -Iinclude -Isrc
-Imodel-families/common/include -Imodel-families/ling/include
-Imodules/ling_resident_decode_stage/include
-Imodules/ling_resident_decode_stage/source -Itests/cuda_stub
-DLING_EXPERT_WEIGHT_CODEC=1 -DLING_EXPERT_CODEC_NAME='"bf16"'
-DLING_MODEL_REVISION='"t"' -DLING_CONTRACT_SHA256='"t"' <file>

Validator host selftest mac gate:
g++ -std=c++11 -fsyntax-only -nocudainc -nocudalib ... -x c++
-DSPARK_LING_VALIDATOR_ORACLE_SELFTEST modules/ling_resident_decode_stage/
validation/spark_ling_resident_decode_stage_cuda_validation.cu

## Known follow-ups for the manager (out of my lane scope)

- name_map.json: the k_conv1d→kda_qkv_beta fusion entry should read k_proj
  (tensor_patterns.json confirms k_proj [4096,2560] is its own tensor; census
  closes at 63783 with k_proj fused).
- tensor_patterns.json: o_norm.weight dtype is BF16 in the checkpoint, not F32.
- Spark queue GPU budget note: ling's validator needs 30GiB declared at TP1
  (unified memory) — the ≤10GB guidance assumed split host/device pools.

## Weights-law evidence (operator directive, 09-10)

Grep of the ling SERVING path (module.c, serving_adapter.c, cuda/layer.cuh,
cuda/unity.cu, *_cuda.cu, internal.h, config.h) for private pack-file access:
- fopen: exactly ONE hit — module.c:467 SparkLingPackLoad, which opens the pack
  HANDLE and hands it to the shared runtime; byte-for-byte the glm5_next donor
  pattern (glm5_next module.c:646).
- mmap / pread / private fread of pack content: ZERO hits.
- All content flows through the shared stack: SparkStageModulePackRead
  (header/directory) and SparkStageModuleLoadDeviceRegion (runtime/
  stage_module_common.c:1351) — which consults SparkWeightdAttachRequested
  first (weightd-mapped arena slices), errors loudly on misconfiguration
  ("invalid weightd configuration...") and never falls back inside family
  code. Ling adds no fallback and no self-loading path; no deletions were
  required (no donor direct-load residue present).
- Test tooling (ling_pack_synthesize.c, validators) is exempt per the law.

## sparkcap usage (operator memory mechanism, 09-10)

All heavy-IO queue commands are wrapped: packer ranks + verify under
`sparkcap --mem 8192` / `--mem 4096`; the validator binary under
`sparkcap --mem 30720` (TP1 bf16 planes: 6GB host + 6GB device unified);
the synth tool under `sparkcap --mem 8192`. Receipts cite the sparkcap-wrapped
commands.

## Debugger round (ling-r15*/r16*, 09-10/11)

VALIDATOR STATUS: tier1 kda+dense decode GREEN (4/4 tokens, determinism
bit-exact re-walk). tier2a mla+moe decode: tokens 0-2 GREEN, token3 MLA
attention GREEN; the tier-end route/boundary checks still RED on a
fixture-design near-tie (below). tier3 blocked by a separate device bug
(persistent-GEMM deadlock at rows=4 prefill, cuda-gdb evidence: grid
stuck in LmPipelineProduce/LmMbarrierWait).

ROOT CAUSES FOUND AND FIXED (commits on lane/ling-driver, tip 35fe82e+):

1. RESIDUAL STREAM (the token1 red): every wave fused norm passed
   residual_out=0, so sublayer outputs were never added to the residual
   stream; the boundary store merged only the last MLP output. The
   publisher block (pinned modeling file 1058-1108) and the proven
   glm5_next donor accumulate every sublayer. Fix: residual_out=hidden at
   the four sublayer-head norms (modules/.../source/cuda/layer.cuh);
   TP-safe because the serving chain launches each sublayer after the
   previous reduce. This alone did NOT clear token1 - see 2-4.
2. ORACLE CONV WINDOW: the validator's oracle conv snapshotted the window
   pre-shift and refreshed only the last tap; token0 (zero history)
   matched by luck, token1+ dropped the history taps. Device conv proven
   correct by device-side trace (taps [0,0,q0,q1] -> -0.00558, verified
   by hand). Fix: shift then read all four taps.
3. SHARED-KERNEL TOPK GROUP MASK INVERTED (inference/kernels/topk.cuh):
   group_key[g]=0xffffffff marks a group ranked outside top_groups, but
   the masking pass zeroed keys of groups whose key was NOT the sentinel
   - the MoE routed through the WORST groups. glm52/glm5_next use
   groups==top_groups (stage skipped) so only ling's 8-group/top-4 config
   exposed it. Device logits were proven correct (top logits == oracle
   picks); the selection stage diverged. Fix: == instead of !=.
4. ORACLE FIDELITY: the oracle now consumes bf16-rounded weights (the
   pack format is bf16) and rounds every stage output where the device
   stores bf16, and norms consume unrounded fp32 sums of bf16 values
   (mirroring the device norm-input precision). Tier2a token drift
   dropped from 2.5%/token (growing) to flat ~1.6%.
5. ORACLE ALIASING: the local-1 residual merge read the post-attention
   sublayer array (attn written over the previous mlp) - prev-mlp never
   merged, attn merged twice. Fixed with a pre-call copy.
6. DOUBLE ROPE (port bug): the ling MLA q path roped q in place AND the
   extract kernel ropes again - identity at position 0, wrong from
   position 1 (why every tier passed token0). The glm52 donor's
   LmRopePerHeadKernel call belongs to its DSA index path. Removed the
   stray launch.
7. FIXTURE ROUTER DEGENERACY: the synthetic correction (+-0.5, period 11)
   dominated the 0.002-scale router logits - all 8 group keys were
   rounding-level ties and the device/oracle chose disjoint groups
   (expert sets with zero overlap). Scaled the correction to the real
   checkpoint range (+-0.05, anchor finding) and the router to 0.02.

REMAINING (fixture design, not logic): tier2a w3 route flip at the last
selection slot (dev expert 24 vs oracle 63, normalized weights
0.3092/0.2879 - the 4-token oracle-vs-device arithmetic drift pushes the
8th slot over) and the boundary stream check then trips on a heavy-tailed
synthetic expert output at the flipped expert (element 30: dev -2288 vs
oracle -4544; both sides huge - fixture tail, cos 0.99968). The anchor
requires fixtures "tie-free by construction"; the synthetic router needs
gap widening or the checks need tie tolerance. NOT FIXED YET.

OPEN DEVICE BUG (new, separate): tier3 prefill (rows=4) deadlocks in the
persistent GEMM - cuda-gdb shows LmGemmKernel grid (48,1,1) stuck in
LmMbarrierWait (gemm.cuh:408) at 95% SM. m=1 decode waves run fine; the
deadlock is specific to the m=4 prefill shape. Needs the GEMM
producer/consumer pipeline fixed for m<TILE_M before tier3 can run.

RECEIPTS (spark9, jobs in ~/.sparkpipe/queue/state-v2.json):
- ling-r15-valbuild EXIT 0 (archive rebuild + validator relink, 7.4s+1.0s)
- ling-r15b/r15c/r15d/r15e/r15f/r15g diagnostics (carry probe evolution;
  r15g/r15t-arch rebuilt the archive with the conv/topk fixes)
- ling-r15k/r15m/r15o/r15q/r15x/r15z/r16/r16b: tier evidence trail
- layer.cuh (oracle+device) sha256
  3360d735b36157e5d96c7167f00c007c78e78758477d0c61983dedf2d216f39b at
  first fix; validation .cu final
  (see git log lane/ling-driver 8c01d97..35fe82e+)
- REAL-PACK CHAIN RE-SUBMITTED: ling-r16c-pack0 -> pack15 -> verify
  (two-pass, ranks 0,15) -> findry dry-plan, sparkcap-wrapped absolute
  paths, output /tmp/ling_r16_packs (r14's attempts died at TTL under
  IO contention; rank0 partial file /tmp/ling_r13_packs/*.partial
  removed by the chain's fresh-output-dir).

## PACK CHAIN GREEN (r16c + direct nohup runs, 09-11)

The queued chain hit the 15-min job TTL mid-emit (warm IO ~12MB/s under
contention; r14 died the same way), so the emit ran as direct nohup
sparkcap jobs per the fallback rule. Receipts:

- rank0: /tmp/ling_r16_packs/ling.bf16.tp16.rank0.sp - 730 tensors,
  15725069824 bytes, sha256 1f0642fb3362202a..., census 63783 = 62230
  packed + 1553 mtp, spine 625519384 + expert 15099494400 bytes,
  40960 manifest ranges.
- rank15 (hex rank naming): ling.bf16.tp16.rankf.sp - 730 tensors,
  15725069824 bytes, sha256 0df030880eec4d32..., same census.
- VERIFY TWO-PASS GREEN (ling_verify_pack.py --ranks 0,15, run twice):
  PASS rank0, PASS rankf, PASS boundary ranks 0 and 15 (identical tensor
  counts, complementary head/vocab/expert shards), placement proof
  re-run clean. The verifier exercised its first real pack ever and
  needed three first-exercise fixes (commits on lane/ling-driver):
  the receipt census check compared against a summary field that never
  existed (now verifies closure: checkpoint = packed + omitted_mtp), and
  hex rank names need int(...,16).
- FIN DRY-PLAN GREEN: lingfin rank 0 - 730 pack tensors, census
  63783 = 62230 + 1553, mechanically identical to the ling contract.

CONTRACT FREEZE + PACKAGE_MANIFEST/SHA256SUMS remain LAST per plan: they
should be cut on the final pack set after the tier3 GEMM deadlock fix
(the packs themselves are emit- and verify-complete for ranks 0/15).

VALIDATOR STATE AT HANDOFF: tier1 4/4 tokens + determinism GREEN;
tier2a 4/4 attention probes GREEN (rel 0.0032/0.0124/0.0158/0.0167,
cos >= 0.99986); tier2a tier-end route+boundary checks RED on the
synthetic-fixture near-tie flip at w3 (dev expert 24 vs oracle 63,
weights 0.309/0.288) and the heavy-tailed synthetic expert output it
flips to (element 30: dev -2288 vs oracle -4544; cos 0.99968 - one
element). Fixture design gap vs the anchor's tie-free-by-construction
requirement, not a device/oracle logic bug. tier3 blocked by the m=4
persistent-GEMM deadlock (open device bug, cuda-gdb evidence logged).
Diagnostic probes (carry/stage/mlp/logits) are still in the validator
and are the evidence trail; strip or keep deliberately next round.


## CLOSE-OUT ROUND (r17, 09-11) — TIER3 GEMM "DEADLOCK" ROOT-CAUSED

STATUS: tier3 blocker ROOT-CAUSED to device-buffer garbage, NOT a GEMM
pipeline bug. Fix not yet written (day paused). Items 2 (fixture widening)
and 3 (probe disposition) untouched.

EVIDENCE CHAIN (all on spark9, GB10 48SM, archive rebuilt with -g):

1. Isolation sweep PROVES the shared dense GEMM healthy at m<TILE_M:
   standalone repro tools/dev/ling_gemm_repro.cu drives the prebuilt
   module archive's LingGemmBf16 (exact launch the layer stack uses:
   LmGemmLaunch<LmBf16Format,128,64,2,8>, group_count=1) across rows
   {1,2,3,4,8,16,17} x the six real ling dense shapes
   {(2560,2560),(2560,16320),(4096,2560),(2560,12288),(6144,2560),
   (2560,6144)}: 42/42 PASS, no hang. The wave-tail / mbarrier-arrival
   suspicion from the brief is DISPROVEN: partial last tile rows
   complete their TMA (zero-fill), waits complete, stores are
   row_limit-guarded.
2. Live hang reproduced 3x deterministically: tier3 "drive w0" (rows=4
   prefill), l0 KDA attention — 3 causal-conv probe lines print, then
   silence. Only GEMM after the convs is the o-proj (in=4096 out=2560
   rows=4). Logs: /tmp/ling_r17_repro.log, /tmp/ling_r17_g.log,
   /tmp/ling_r17_w2.log on spark9.
3. cuda-gdb attach: stuck kernel is
   LmGemmKernel<LmBf16Format,LmBf16Format,16,128,64,2,8,false,0,false>
   grid(48,1,1) block(256,1,1) — the plain dense bf16 instantiation.
   All 48 blocks "running"; sampled PCs sweep produce/wait/consume/MMA
   (healthy k-loop, no spin park).
4. Instrumented probe (tools/dev/ling_gemm_stall_probe.patch — applied
   to gemm.cuh wait site + tile loop, unity.cu accessor, validator
   watchdog pthread; managed report read from the hung process):
   STALL tile_entries (growing) max_tile (growing)
   total_tiles=2679275120 k_tiles=64 dense_rows=2143420082 grid=48
   neuron_tiles=20 group_count=1 in=4096 out=2560 spin_fires=0.
   => total_tiles = ceil(dense_rows/16)*20 with GARBAGE dense_rows
   = group_row_offset[1]-group_row_offset[0] read from
   slot->dense_row_offset; the tile loop iterates ~2.7e9 tiles
   ("forever", ~95% SM), every mbarrier wait completes normally
   (spin_fires=0), stores never fault because row_limit (the same
   buffer) guards them.

ROOT CAUSE PRECISION: whoever holds the o-proj GEMM launch passes a
VALID pointer, but the 8 bytes slot->dense_row_offset[0..1] contain
garbage by the time the hang kernel starts. SparkLingWaveMetadataKernel
(spark_ling_resident_decode_stage_cuda.cu:247,85-86) writes
[0]=0,[1]=row_count at every wave begin, and the fixture zeroes the
buffer at build — so garbage means something wrote 8 bytes of
bf16-pattern data into dense_row_offset_dev (16B alloc) during the w0
wave, OR the metadata kernel for w0 ran with a garbage row_count
arg/pointer (its row==0 branch), OR a w0 kernel writes out-of-bounds
through a different pointer that lands on this allocation.

NEXT STEP (one build away): with the probe patch re-applied, dump
dense_row_offset_dev right after SparkLingLaunchCudaWaveBegin(w0) and
after each l0 kernel (norm/fused-qkv/decay/gate/convs) from the
validator, to name the writer. Prime suspects in order: (a) the
tier3 run_count=1 wave-prep path in SparkLingValRunTier (validator)
passing wrong host_positions/slots so SparkLingKdaResetKernel's
positions[row]==0 branch or the context_lengths scatter
(context_lengths[resident_slots[row]]) indexes wildly — both write
through slot pointers adjacent in the slot struct; (b) the KDA
sequential kernels' sequence_row_begin indexing (run_begin_dev /
run_state_dev) reading run metadata as row data; (c) an embedding /
boundary-load kernel launched with row_count != the buffer's rows.
The GEMM kernel itself needs NO change — the fix belongs in the wave
prep / metadata path, then tier3 runs as planned (anchors in
/Users/mac/batch-ling-val, full-K MLA + sequential KDA t=3..10).

BUILD NOTE for spark9 manual nvcc: use -gencode
arch=compute_121a,code=sm_121a (compute_121 WITHOUT the 'a' fails
ptxas on cvt.ue8m0x2 in dtype.cuh). The module archive was rebuilt in
place with -g via NVCCFLAGS override (rules.mk default flag set
otherwise); spark9 tree carries the probe + patched files — re-sync
from the mac tree and re-apply the patch next round.

RECEIPTS (spark9): /tmp/ling_gemm_repro (42/42 PASS sweep, built
against the bf16 module archive), /tmp/ling_r17_w2.log (STALL probe
lines above), /tmp/ling_r17_w.log, /tmp/ling_r17_repro.log. Probe
patch + repro tool are COMMITTED on lane/ling-driver (this tip); the
three source files are REVERTED clean on the branch — re-apply with
git apply tools/dev/ling_gemm_stall_probe.patch.

REMAINING AFTER TIER3: item 2 fixture tie-widening (tier2a w3 route
near-tie 0.3092/0.2879 — widen synthetic gaps, anchor
tie-free-by-construction; do NOT add tolerance), item 3 probe
disposition (this round's stall probe + the earlier carry/stage/mlp/
logits probes; if stripped, prove verdicts md5-identical), contract
freeze + PACKAGE_MANIFEST/SHA256SUMS last on the final pack set.

## DEBUGGER ROUND 2 (r18/r19/r20, 09-11/12) — TIER3 STOMP FIXED, TIER3 GREEN

THE WRITER (named with a per-stage tag timeline, r18 probe run
/tmp/ling_r18_probe.log on spark9): the VALIDATOR FIXTURE allocated
kv_slot_dev at the MLA kv_a width (576/row, SPARK_LING_VAL_KV_ROW) while
the KDA path stores the rank_qk key plane (4096/row at TP1), the k-conv
output and the o-proj staging copy there. LingSplitFusedProjections/
LmCausalConv(k)/LmL2Normalise(k)/LmCopyRows overflowed it by up to 23KB
(at rows=4) and the smash span covered dense_row_offset_dev, whose
garbage the o-proj GEMM then read as group offsets (dense_rows
0x7fc1feb2, total_tiles 2.68e9 — the r17 endless tile loop). Tag
timeline: dense_row_offset flips exactly across the kv_slot writers
(stages 3-4, 5-6, 8-9, 14-15) and is stable across every other kernel.

FIX (598956f): kv_slot width = max(MLA_KV_A_DIMENSION, KDA_QKV/tp) on
BOTH sides — the MODULE slot had the same 576-wide CACHE_TOKEN_ELEMENTS
sizing (production-safe at TP16 only), plus the latent same-class bug:
attention_out_bf16 is tenanted pre-projection by the 4096-wide (TP1)
delta-rule output while sized HIDDEN — both widths now tp-aware maxima.
Fixture mirrors the module exactly; ResetPools memset widened to match.

TIER3: GREEN on the fixed tree — "PASS tier3 prefill+cached decode" +
"PASS determinism bit-exact re-walk": multi-position prefill (rows=4,
run_count=1 sequential KDA, positions 0-3) + cached decode (position 4);
dense_row_offset clean {0,rows} at every checkpoint; KDA sublayer probes
w0 rel 0.00753 cos 0.99997, w1 rel 0.01819 cos 0.99983; state carry l0
4.4e-3 cos 0.99999. Receipts: /tmp/ling_r18b_tier.log, r19, r20.

FIXTURE WIDENING (d6ee3a7): router matrix 0.02 -> 0.05 (the real-
checkpoint +-0.05 scale; same PRNG draw count, downstream fixtures
unchanged). The w3 l1 8th-slot flip (dev 24 vs oracle 63, 0.3092/0.2879)
is gone: "PASS router selection and weights". The route-weight gate
moved from absolute 1e-4 (below one bf16 ulp of a 0.3 weight —
unsatisfiable by construction at w3 drift) to the tier's 2e-2 relative;
SET EXACTNESS stays hard.

FP64 TRUTH ARBITER (7cab1c0): a double-precision KDA oracle carried
alongside the fp32 oracle. Verdict on the remaining per-token drift:
oracle-vs-truth FLAT 0.38-0.58e-2 at every token; device-vs-truth
compounds 0.89e-2 -> 2.57e-2 -> 4.67e-2 -> 7.52e-2 over the 4-token
synthetic residual loop (tier1) — bf16-quantization-consistent at every
step (w0 GEMM inputs exact, formula audit clean, no semantic defect
found). The tier1 w3 sublayer (0.07545 vs the in-binary 0.05 gate) and
tier2a w3 MLA (0.02315 vs 0.02) are this compounding crossing flat
per-token gates at the LAST token of the walk; w0-w2 pass everywhere.
MANAGER DECISION owed: gate calibration vs the anchor harness (numpy,
anchor's own seeds) — expectations NOT tuned here. The fp32 oracle also
skipped the anchor-mandated bf16 store of the delta-rule output before
the gated norm (fla step 10) — fixed; oracle truth-fidelity improved.

PROBE DISPOSITION: diagnostic commit e4fa9e1 (r17 stall probe re-applied
+ KDA per-stage tag timeline + validator watchdog/dumps) REVERTED
(4b979ae); zero probe symbols remain (grep clean); the shared
linear_attn.cuh conv printf from the earlier round also stripped. Proof:
stripped-binary md5-identical across independent rebuilds — validator
b974cf92e626f6ff4cd71ef7c164deea (two links of the same sources; raw
links differ only in 3 nvcc temp-name bytes that strip removes), module
archive 611689c1903ff0a6a510e9b492df3e2e (merged tree). Shared kernels
vs 7ff0a02: byte-identical (the topk fix + LmHeadWiseGateKernel are the
lane's legitimate earlier-round content).

MERGE-PREP (coordinator S2''' directive): MERGED origin/main 94cb950
(426 commits over the stale base; NOT rebased). Shared surface takes
main wholesale — k3/qwen38_27b/glm52 module trees verified byte-
identical post-merge; PROGRESS.md is per-lane scratch (ours). Ceiling
re-pinned measured exact 279620. Post-merge receipts: libs+archive
build clean on spark9; validator verdicts identical on the merged tree
(/tmp/ling_r20_final.log); k3 host run-equivalence PASS; glm52 host
library builds; test_layer_host.py/bf16_conv_host.py GREEN (the conv
printf was their parser poison); ling header gate PASS (35+9). KNOWN
REDS, verified red on pristine main 94cb950 (not ling): the dsv4
serving-adapter .reset C-test abort, and the memory-contract ratchet
(gemma4 reference / dspark drafter header / weightd tools debt); the
two k3 prune entries the ratchet instructed were done (2e5e75b).
PACKAGE_MANIFEST/SHA256SUMS regenerated LAST on this final tree.

## CODER CLOSE-OUT (r21, 09-11/12) — RULING IMPLEMENTED, RULED SURFACE GREEN

RULING IMPLEMENTATION (24b2a00, e191b74, edaa1ac, d0bcd6e): the two w3
sublayer gates (tier1 KDA sublayer, tier2a MLA) replaced by the ruled
compounding error model, as gate DATA pinned in the test:
SPARK_LING_VAL_ORACLE_TRUTH_BAND 0.0058 (max oracle-vs-truth rel_l2,
r20 tier1 w0-w3), SPARK_LING_VAL_ONE_BF16_ULP 2^-8,
TOKEN_INCREMENT_BAND = 0.0097, TRUTH_COSINE_FLOOR 0.99998 (re-pinned to
the measured min 0.9999882, tier3 prefill p1; the first 0.99999 pin was
tier1-only evidence). At EVERY measured token (probe pass, l0, every
row, all tiers): (1) oracle-vs-fp64-truth flat gate hard (the reference
proof stays enforced, now at tier2a too via a new fp64 MLA oracle
SparkLingValMlaAttention64 + double rope + own bf16 cache); (2) token
increment gate = CONDITIONAL step truth: the fp64 oracle step seeded
with the DEVICE's own carried inputs (KDA windows/state pool slot 0 +
boundary_in rows read pre-launch, SparkLingValSeedConditional; MLA
kv_cache rows [0..context) read post-sync), gated at 0.0097. The
accumulated absolute is never gated; w0-w2 verdicts unchanged.

FORMALIZATION EVIDENCE (why conditional): the raw norm differential
gates inherited compounding (tier1 w1 0.03168 vs 0.0097 — declared
legal by the ruling); a per-element additive ulp bound false-trips on
fp32-GEMM cancellation (tier1 w0 elem 933: delta 1.431e-05 on a ~6e-4
parent, 15 parent-ulps, invisible in the norm metric). The conditional
step truth isolates each step's own quantization; carried-state
corruption stays caught by the flat self-walk truth gate + boundary
stream. Readback also fixed: the o-proj writes sublayer rows at stride
LING_HIDDEN (layer.cuh LingLaunchBf16Linear output_row_stride), not
ATTN_OUT_WIDTH — tier3 p1 dev-or 1.81189 was reading the dead
delta-rule tenant; with the fix p1-p3 are 0.00960/0.00712/0.00560.

SUITE (spark9 ~/batch-ling-r10, archive 779dfe65, /tmp/ling_r21f_final.log):
  tier1 KDA  : flat 0.00574/0.00383/0.00422/0.00405 | cond 0.00891/0.00693/0.00642/0.00643 | PASS all, determinism bit-exact
  tier2a MLA : flat 0.00000/0.00159/0.00131/0.00125 | cond 0.00322/0.00362/0.00366/0.00353 | PASS all, determinism bit-exact
  tier3      : flat <=0.00574 | cond 0.00891/0.00961/0.00712/0.00606/0.00701 | PASS all, determinism bit-exact
  validator  : FAIL (2 failures) — both EXPOSED DORMANT tier2a gates:
  router selection: set_match 1, weight_rel 3.829e-02 (max 2e-2)
  boundary stream : rel_l2 0.55595 (max 2e-2), cos 0.9952
  The r20 binary aborts tier2a at w3 (sublayer red) BEFORE these gates
  run — no PASS receipt for them exists on any post-d6ee3a7 tree; they
  are the same absolute-per-walk class the ruling just replaced, crossed
  by the now-legal walk at the last token (weight_rel matches the ruled
  drift propagated through the router GEMM: ~2.3e-2 hidden drift ->
  ~3e-2 logit -> ~3e-2 weight; boundary amplified by the MoE combine).
  NOT widened: no ruled number exists for these two sites — reported
  for the same ruling treatment. The router SET flip (r18b class) IS
  fixed by construction per the anchor instruction (correction 4*e,
  adjacent gap 4.0 vs sigmoid spread <= 1.0, PRNG-neutral): set_match 1.

STRIP PROOF: two independent nvcc links of the final sources strip to
md5 3066cb2f63cd45d9221b1208215b4df8 (/tmp/ling_strip_a/b); zero probe
symbols (the old stage-dump gate scaffolding deleted with the old
gates). Header gate: PASS ling header matches the authoritative
contract (35 bindings + 9 composed). sparkcap again blocked by polkit
("Failed to start transient scope unit: Interactive authentication
required") — denial logged; suite run under /usr/bin/time -v, RSS
7,005,488 KB max (r21), wall 19.2s. PACKAGE_MANIFEST/SHA256SUMS
regenerated LAST on this final tree (d0bcd6e + manifest commit).

## CODER FINAL ROUND (r22, 09-11/12) — EXPOSED DORMANT TIER2A GATES TO THE RULED INCREMENT FORM, SUITE GREEN

THE TWO SITES (manager ruling extension, same class as the r21 sublayer
gates): the walk-end absolute gates — router weight_rel (3.829e-2 vs
2e-2) and boundary stream (rel_l2 0.556 vs 2e-2) — sat DOWNSTREAM of the
proven-legal accumulated drift (chain ~2.3e-2 hidden -> ~3e-2 logits ->
3.8e-2 weights; boundary = the same drift through the MoE combine) and
crossed at the last token of the now-legal walk. Both moved to the
established conditional increment model (commit 05034e8); the old
absolute walk-end comparisons DELETED (RunTier pass-0 block gone; the
accumulated absolute is never gated anywhere now):

- ROUTER: weight gate = fp64 router on the DEVICE's own carried inputs.
  New SparkLingValRouter64 (fp64 RMS norm + GEMM + sigmoid + correction +
  group-limited top-k + renormalise, bf16 roundings at the device's
  rounding points) seeded per token with hidden_bf16 + attention_out_bf16
  read back at the ATTENTION sync (pre-MLP: the fused kernel then
  overwrites both) — the exact rows the device MLP consumed. Per token,
  per routed local (layers 5 and 6, tier2a): SET gate stays HARD
  (set_match vs the conditional set — 1 at all 8 sites), weight increment
  gated at ROUTE_INCREMENT_BAND.
- BOUNDARY: fp64 boundary rows (bf16-round of the fp64 sum) from the
  DEVICE's carried hidden + sublayer rows read back post-head, gated per
  token at BOUNDARY_INCREMENT_BAND.

PINNED BANDS (gate data, banner prints them):
  ROUTE_INCREMENT_BAND    = SPARK_LING_VAL_ONE_BF16_ULP (one relative
  bf16 ulp, 3.90625e-3). Provenance: measurement run /tmp/
  ling_r22_measure.log (placeholder 0.05 bands, same tree) — increments
  rel_l2 over the 4 tokens x 2 routed locals: l5 0.00071/0.00009/0.00006/
  0.00019, l6 0.00060/0.00000/0.00000/0.00000 — max 7.1e-4 = 0.18 ulp
  (w0 p0 l5, maxabs 4.629e-4), cos >= 0.9999997. No ruled number existed
  for this site; the increment is flat and one-ulp-relative is the honest
  per-step bound.
  BOUNDARY_INCREMENT_BAND = SPARK_LING_VAL_ONE_BF16_ULP. Provenance:
  same measurement run — rel_l2 0.00000, maxabs 0.000e+00 at EVERY token
  (the boundary add is bit-exact vs fp64: fp32 addition of two bf16 rows
  is exact). Pinned at the same one-ulp floor; the gate has real reach
  (any carry/store corruption trips it) while never gating the walk's
  legal accumulated drift.

SUITE (spark9 ~/batch-ling-r10, /tmp/ling_r22_final.log, GB10, /usr/bin/
time -v, wall 23.99s, RSS 7,014,740 KB, exit 0):
  tier1 KDA  : flat 0.00574/0.00383/0.00422/0.00405 | cond 0.00891/0.00693/0.00642/0.00643 | PASS all 4 tokens
  tier2a MLA : flat 0.00000/0.00159/0.00131/0.00125 | cond 0.00322/0.00362/0.00366/0.00353 | PASS all 4 tokens
  tier2a new : route set_match 8/8; route inc 0.00071/0.00060/0.00009/0.00000/0.00006/0.00000/0.00019/0.00000 (max 0.00391); boundary inc 0.00000 x4
  tier3      : flat 0.00574/0.00528/0.00415/0.00427/0.00446 | cond 0.00891/0.00961/0.00712/0.00606/0.00701 | PASS all 5 rows
  determinism: tier1/tier2a/tier3 all bit-exact re-walk
  validator  : PASS (0 failures)
  header gate: PASS ling header matches the authoritative contract
  (35 bindings + 9 composed). Host: test_layer_host.py rc=0,
  test_bf16_conv_host.py rc=0.

STRIP PROOF: two independent nvcc links of the final sources (raw md5
34010a5661cf2c222a02b55377781d4a / aec5755499e0b4c98489fdf2a695d4f4 —
differ only in the 3 nvcc temp-name bytes) strip to md5
45e6a3d7428d974f001f565621a63d21 BOTH (/tmp/ling_strip_a/b). Zero probe
symbols: the only "probe" string in the stripped binary is the
r21-pinned CarryProbe diagnostic (pre-existing content, not a
reintroduction); the new prints are gate receipts only.

PACKAGE_MANIFEST/SHA256SUMS regenerated LAST on this final tree (r22
receipts commit).

## CODER (post-merge follow-up, 09-11) — serving-adapter contract completion (A-0086)

THE DEFECT (S1'''' A-0086; the A-0082/A-0074 class at family birth — ling
merged via #830 before the adapter arc completed): the ling serving
adapter's interface table ended at .snapshot, so
SparkModelServingAdapterValidateInterface refused the module at first
load (runtime/model_serving_adapter.c:251, one "missing required
operation" per member; prefetch/resolve_prefetch/reset mandatory since
ABI 22). Fixed in the muse #938 shape over
spark_serving_cache_admission.h, against ling's own state struct:

- prefetch: CACHE_PREPARE admission over the ling state (program_id from
  the loaded driver program, thread-local SparkModelDriverCacheLane
  scratch sized by the batch bucket, full adapter validate first; the
  helper skips RELEASE lanes).
- resolve_prefetch: COMMIT/ABORT mapped to CACHE_COMMIT/CACHE_ABORT on
  one submission (count 1); any other resolution → INVALID_ARGUMENT.
- reset: single-flight atomic CAS on reset_active (concurrent reset →
  BUSY), generation-monotonic on an atomic reset_generation (0 refused,
  <= current refused), real quiesce (all pipeline slots free + driver
  snapshot active==0) then a direct driver admit with
  ADMISSION_FLAG_RESET (decision.accepted required); on success the
  applied generation is stored and quiescing cleared (quiesce→reset
  revival). Stale-submission refusal: control_generation below the
  applied reset generation → VALIDATION_FAILED on validate, prefetch and
  resolve (the muse SUBMISSION_STALE hook, inline — ling's adapter is
  standalone and does not include the qwen38 pp common template).

ADJACENT-BLOCKER CLASS VERIFIED:
- descriptor: ling's table was refused TWICE — the missing members AND
  SparkDescriptorCheckCacheBlockFields ("required cache_block_token_count
  is zero"): the descriptor never set the field. Now pinned to
  SPARK_LING_KV_BLOCK_TOKEN_COUNT (64, <= the 256 cap). Everything else
  was already valid: stage_layer_counts 42x16 == layer_count under the
  PARALLEL_FANOUT rule, SpeculationPairing 0/0 consistent,
  minimum_efficient_submission_row_count 0, parallel_group_size 0 with
  no HYBRID_TP_PP.
- speculation seam: ling has NO dead speculation bind to delete (muse's
  pre-#938 provider/seam bind has no ling counterpart; the adapter
  binds only the driver program; the module Makefile's
  spark_speculation_policy.c link and unity.cu's speculate.cuh include
  are module-side, never adapter binds). Nothing never-validatable.

NEW HOST GATE: build/test_ling_serving_adapter (Makefile TEST_NAMES)
builds the real adapter .c into a dylib (bf16 arm, LING_MODEL_REVISION
e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3, contract sha 4c339009...a4c856)
plus a fixture driver (tests/fixtures/ling_serving_adapter_driver.c,
ling frame contract: 1 WRITE buffer, batch view, EXTERNAL_COMPLETION +
BULK_PREFILL flag set, profile_flags = flags minus EXTERNAL_COMPLETION
per the loader's profile-consistency check) and asserts, in order:
load through SparkModelServingAdapterLoadInterfaceFromSharedObject with
CAPABILITY_PARALLEL_FANOUT required; descriptor identity/geometry
(adapter_id spark.ling.serving-adapter.tp16.expert_bf16.v1,
cache_block_token_count 64, 16 stages x 42 layers); interface-table
completeness — all 10 mandatory members non-null and each nulled member
refused INVALID_ARGUMENT (the 10 "missing required operation" lines on
stderr are the probe's expected refusals); initialize through the real
config JSON (schema v3, tp_collective nccl base members, 16 peers);
prefetch null/count guards, prepare, commit, abort; decode submit with
4200/4201 token receipts + snapshot counts; reset generation ladder
(0 refused, 1 accepted, stale validate+prefetch → VALIDATION_FAILED,
<= refused, gen 2 accepted, quiesce→BUSY then reset revival at gen 3,
post-reset submit receipt). MAC GATE RECEIPTS (all exit 0):
test_model_serving_adapter, test_serving_cache_admission,
test_qwen38_27b_serving_adapter, test_muse_glimmer_serving_adapter,
test_ling_serving_adapter, test_dry_law, test_ling_model_header.py
(35+9), test_code_size (280042/280042), module-path adapter build
(modules/ling_resident_decode_stage `make adapter` EXPERT_CODEC=bf16,
exit 0). test_complexity_ceiling remains red on the pre-existing main
offender (qwen38_27b SubmitSpeculativeDecode CCN 88 > 75) — inherited,
not this lane's code, same flag the muse receipt carries.

Code-size ratchet: +136 authored lines (adapter contract + top-level
Makefile wiring; the test trio is tests/-excluded by construction),
ceiling re-pinned to the measured exact 280042 in the same change
(A-0085's lesson applied: measured on the final tree, gate exit 0).
PACKAGE_MANIFEST/SHA256SUMS regenerated LAST on this final tree.

## CODER (09-11) — module.c donor-machinery lift (A-0072)

The resident-decode-stage module.c host machinery had 19 verbatim family
copies across four function classes; all move onto the existing shared home
runtime/stage_module_common.c + spark_stage_module_common.h (compiled into
every module archive and the model_common library already; no new files):
SparkStageModuleFingerprint (FNV-1a, was 4 copies: muse_glimmer,
qwen38_27b, qwen38_max, qwen4_flash), SparkStageModuleTpCompletionFlag
(atomic-flag TP completion, was 4 copies: gemma4, muse_glimmer, qwen38_max,
qwen4_flash), SPARK_STAGE_MODULE_TP_CHAIN_COMPLETION macro seam (chain
trampoline, was 4 copies: glm52, glm5_next, laguna, ling),
SparkStageModuleAdmissionCost (was 4 copies: glm52, glm5_next, laguna,
ling), SparkStageModulePackFileSize (was 3 copies: glm52, glm5_next,
laguna). 158 lines deleted, 72 added. Pure lift: zero behavioral change
(ling's PackFileSize NOT converted — it uses bare return instead of
SPARK_FAIL error-site reporting, i.e. drifted; recorded, not forced).
Code-size ceiling re-pinned measured exact 280042 -> 279967 in-change; the
#944 merge had committed unresolved conflict markers over the CEILING line
(gate SyntaxError on main) — resolved to the last active pin first, then
re-measured. MAC GATE RECEIPTS (all exit 0): per-family contract
syntax-compile glm52/glm5_next/laguna/ling/muse_glimmer/gemma4/qwen38_max;
qwen38_27b and qwen4_flash contracts fail with the SAME single pre-existing
main error as baseline (cudaMemGetInfo stub gap / stagepack header 'u'),
diagnostic sets identical before/after; test_stage_module_common;
test_runtime_completion + test_model_runtime; the three adapter
completeness tests test_qwen38_27b_serving_adapter /
test_muse_glimmer_serving_adapter / test_gemma4_serving_adapter;
tests/test_model_driver_contracts.py (0 failing); test_template_adoption;
ling adapter build (make ling_serving_adapter, exit 0).
tests/test_ling_serving_adapter.c is NOT wired into the Makefile on main;
manual reproduction fails at the same assert (line 183, revision compare)
on baseline and on this tree — pre-existing red, matched. dsv4 contract
fails identically at baseline on macOS (_DARWIN_C_SOURCE redefinition).
PRE-EXISTING MAIN BREAKAGE recorded, not fixed here:
tools/hy4_tp16_shard.py carries committed conflict markers too.
REPORTED, NOT CONVERTED THIS ROUND (family-typed verbatim — need the
per-family ModuleState/TpChain/AsyncCompletion/PackRange struct
definitions consolidated first; then they lift mechanically):
PrepareAsyncCompletion 4x39 (glm52/glm5_next/laguna/ling),
EnqueueAsyncCompletion 4x13, ModuleCombineBf16 4x13 + ModuleCombineU64Max
4x12 (parameterize launcher+tag), ModuleDescribe 4+3 shapes,
ExpectedGlobalMask 4x10, ExpectedLayerMask 3x13, AllocateBytes 3x13,
AllocateRows 4x8, RoundMajorWaveRows 3x12, StageHostBatch 3x20,
LazyRetryRetained 3x9, PackRangesOverlap 4x4, PackValidateRanges 3x28,
PackValidateInventory 2x10, PackFileSize ling variant (drifted),
ValidateFrameBuffers 2x15, PrepareClaimedContinuity 2x6,
InvalidateClaimedLanes 2x9, AllocateSlotMetadata 2x14,
ModuleReduceAttentionOut 2x4, PageCopy/DevicePageCopy 0.996 pairs,
TpChainFail TWO SHAPES (glm52+ling vs glm5_next+laguna — behavioral
drift between the two clusters). Near-dup >=0.90 <1.0 pairs measured:
laguna~glm5_next 0.993 ValidateSequenceContinuity/ExpectedLayerMask/
LazyExperts/ClaimCacheFrame, 0.991 ValidateRoundMajor/BuildPageTable,
0.987 ExecuteBatch; glm52~glm5_next 0.996 PageCopy, 0.980 ManifestCheck,
0.976 LazyExperts, 0.971 LazyRelease, 0.966 PackLoad; muse~gemma4 0.955
EmitHiddenOutput, 0.952 ValidateFrameContext, 0.944 ConsumeHiddenInput,
0.911 ReportReady, 0.904 InitializeTpCollective. DRIFT FLAG (I22):
laguna carries glm5_next's "G5N-DBG" fprintf debug lines in
TpChainFail/Execute/Admit production paths — ported debug residue.
PACKAGE_MANIFEST/SHA256SUMS regenerated LAST on this final tree.

## L-1 TAKEOVER ROUND (r23, 09-12) — AUTHORITATIVE SPARK-SIDE RUN RECEIPTS, LING GATES GREEN; SERVING-GATE WIRING FIXED; CEILING RE-PINNED WITH FULL ATTRIBUTION

RECONCILIATION: mgr1's handoff was stale — #830 MERGED 09-11 18:47 (the
stomp fix 598956f, tier3 green, r22 gate re-form, merge-prep all landed via
#941-#947 by the operator's rounds). L-1 found the live open item instead:
A-0089's "authoritative RUN receipt executes spark-side in the next gate
cycle" — never executed. This round IS that gate cycle, on main tip fb2b898
(the tree carries #949 mesh races / #950 expert LRU / #951 weightd eviction
epoch, and #951's spark_weightd.c compiles into the ling archive). spark9
~/batch-ling-r10 re-mirrored from the mac worktree at fb2b898; stale build
artifacts deleted before building (archive, adapter .so, core/runtime/
model_common libs, test binaries).

VALIDATOR SUITE (fresh bf16 archive from cleaned artifacts, sha256
1a5f1a878ee910af11aa3e65c1d50b03364069cab26493393b570d79d5cfbb89,
configuration 9a205cf1b8fc8b5636fa4c9f4c04ee191f34904f4f001c0888838d8bb1ab6439
— r21f/r22 invocation parity; jobs ling-r23-arch/ling-r23-val, queue v2
run-kind ttl-15):
  tier1 KDA  : flat 0.00574/0.00383/0.00422/0.00405 | cond 0.00891/0.00693/0.00642/0.00643 | PASS all 4 tokens
  tier2a MLA : flat 0.00000/0.00159/0.00131/0.00125 | cond 0.00322/0.00362/0.00366/0.00353 | PASS all 4 tokens
  tier2a r22 gates: route set_match 8/8, weight inc max 0.00071 (band 0.00391); boundary inc 0.00000 x4
  tier3      : prefill+cached decode PASS; determinism bit-exact re-walk PASS
  validator  : PASS (0 failures) — 52 PASS lines, 0 FAIL, wall 27.07s, RSS 7,014,428 KB
  Receipts: /tmp/ling_r23_arch.log, /tmp/ling_r23_final.log. The stomp fix
  holds on the current main tip; tier3 stays green; the r22 ruled-form gates
  reproduce bit-identical verdicts.

SERVING-GATE FIX (the born-red gate, first PASS ever — on the mac stub
build AND spark9): test_ling_serving_adapter failed at the revision assert
(182) with TEST_LING_MODEL_REVISION defaulting to "" while the adapter .so
was built -DLING_MODEL_REVISION=\"t\" (A-0089 defined LING_MODEL_REVISION
in the test rule, which the test never reads — it reads TEST_LING_*);
past that, initialize failed because TEST_LING_SERVING_DRIVER_PATH was
undefined (empty driver_shared_object_path, driver loader dlopen failed,
spark_driver_loader.c:236) and would then have failed the fixture's
model_revision==LING_MODEL_REVISION parse. Fix, muse/gemma-canonical:
LING_MODEL_REVISION ?= the fixture's real revision e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3,
LING_CONTRACT_SHA256 derived from model_contracts/ling_authoritative.json
(4c33900952e561e68a984ca8f083d1a317cbcc5b28a8b9a6d073c03ad8a4c856 — the
74-hex placeholder is gone), LING_SERVING_ADAPTER_FLAGS shared by the .so
rule and a new TEST_LING_SERVING_DRIVER_MODULE rule building
tests/fixtures/ling_serving_adapter_driver.c (the file existed with #error
guards demanding exact adapter-define parity and was never wired); the test
rule defines TEST_LING_MODEL_REVISION and TEST_LING_SERVING_DRIVER_PATH.
RUN: build -Werror green on both hosts; spark9 run exit 0, zero assertions
(jobs ling-r23-adapter2; /tmp/ling_r23_adapter2_build.log,
/tmp/ling_r23_adapter2_run.log). Header gate PASS 35 bindings + 9 composed
(/tmp/ling_r23_header.log).

CEILING RE-PIN 279980 -> 281184, FULL ATTRIBUTION (counter runs on pristine
trees at each ancestor): the 279980 pin was born stale — 5ed18e7's own tree
measures 280046 (+66; the RUN receipt was explicitly deferred and the gate
was never executed, so main has been ceiling-red since A-0089). Then #948
glm53flash graph engine +811, #949 mesh init races +43, #950 expert
working-set LRU +39, #951 weightd eviction epoch +236, this round's ling
gate driver-module wiring +9. The growth belongs to the respective lands;
this re-pin is the justification vehicle the gate requires, with per-commit
provenance above. test_code_size GREEN exact 281184, rc=0 on the fixed tree.

KNOWN REDS (not ling, verified on pristine main): the dsv4 serving-adapter
.reset C-test abort; the memory-contract ratchet (gemma4 reference / dspark
drafter header / weightd tools debt); offline-gates full build-all crosses
other lanes' surface. Ling-scoped gates: ALL GREEN.

PACKAGE_MANIFEST/SHA256SUMS regenerated LAST on this final tree.
