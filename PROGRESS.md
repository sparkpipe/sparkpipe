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
