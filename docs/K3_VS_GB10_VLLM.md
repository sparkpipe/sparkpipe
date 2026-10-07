# K3 vs gb10-vllm — the honest comparison and the >30 tok/s path

Reference: github.com/ciprianveg/gb10-vllm `kimi-k3/v5` (v5-prd image), cloned
at /tmp/gb10-vllm for the 2026-09-21 audit (cf5645a). Our side: the SparkPipe
K3 resident-decode stack. Their side carries file:line into their repository;
our side names files and symbols. SparkPipe's own K3 numbers come from
[K3_PERF.md](K3_PERF.md). Every number is graded MEASURED (a receipt in the
tree), DERIVED or ARITHMETIC (computed from a measured or analytical input),
or ANNOUNCED (attributed claim without a receipt in our tree).

## The numbers

| Stack | Point | Grading |
|---|---|---|
| gb10-vllm v5-prd, TP16+DCP8, nst6 spec | C1 29.81 tok/s, C8 87.12 agg (136 peak) — v5/README.md:68-79 | ANNOUNCED (their published table; no artifacts in our tree) |
| gb10-vllm v5-prd, DSpark nst6 on (llama-bench tg2048, coherent corpus, lower acceptance) | 23.59 t/s @d4000, 20.03 @d200k — v5/README.md:81-91 | ANNOUNCED |
| gb10-vllm, the 2026-09-19 announcement by @ciprianveg | 29 tok/s K3 decode on 16 Sparks | ANNOUNCED (the same stack as the 29.81 row, not SparkPipe) |
| SparkPipe K3, one TP4 rank of stage 0 on one Spark | 55.5 ms warm B1 stage step — K3_PERF.md "Numbers" | MEASURED (one stage, 2026-08-16) |
| SparkPipe K3, TP4xPP4 | 18.0 tok/s pipelined, ~4.5 tok/s for one B1 sequence through four stages — K3_PERF.md | ARITHMETIC from the 55.5 ms step |
| SparkPipe K3, TP16 PP1 | 20.2 tok/s at 15 us per all-reduce; ~12.9 tok/s at GLM's measured 167 us — K3_PERF.md | DERIVED / ARITHMETIC; not deployable (no TP16 descriptor) |
| SparkPipe transport, 16-rank 8 KiB all-reduce | 167 us p50 (GLM5_NEXT_ROOFLINE.md, fleet 2026-09-25) | MEASURED |
| SparkPipe transport, warm serving chain | allreduce 561-700 us/round, pure-mesh component 136 us/round (docs/archive/HILLCLIMB_20260920.md, an archived session log; hillclimb/allreduce branch tip) | MEASURED (2026-09-20) |

SparkPipe's K3 fleet numbers are in K3_PERF.md (2026-10-07: TP16 B1 10.3
tok/s, B8 ~34 tok/s aggregate, no speculation, BF16 spine). gb10-vllm
publishes no no-speculation number: its whole v5 benchmark section runs
static nst6 DSpark (README "Benchmarks (v5-prd, TP16 + DCP8, static nst6,
TILE8, kv3G, 16x GB10)"), with load-time MXFP8 for selected dense linears and
FP8 shared experts. Its llama-bench row, 23.59 t/s at depth 4000 (20.03 at
200k), is speculative on a corpus where it reports lower acceptance; the
coding C1 of 29.81 is speculative too. A like-for-like comparison needs our
speculative number or their nst0 run.

## Per-lever verdicts

### L1. One-shot RoCE collectives (their RoCEnante) — verdict: EVEN, ours has more

- Theirs: b12x one-shot all-reduce + all-gather over RoCE for decode-size
  messages, capability vote at init, fail-stop, rank-invariant dispatch,
  single-stream-in-graph requirement — v4plus-build/mods/v4plus-batch1/
  payload_b12x_roce_all_reduce.py:5-27; DCP all-gather for query head-gather +
  LSE reduce legs + KV shard gather — v4plus-build/mods/v4plus-rocante-dcp/
  patch_rocante_dcp.py:3-39. Recipe caps: VLLM_ROCE_ALLREDUCE_MAX_SIZE 1M /
  ALLGATHER 8M with NCCL fallback above the cap — recipes/kimi-k3-dcp-tp16-prd.yaml:65-66.
- Ours: the same protocol class shipped and battle-worn: the one-shot mesh
  relay with publish/doorbell/exact-key waits — ring/transport/
  tp_device_collective.c (`SparkTpDeviceCollectiveRunRound`), publish-ack
  backpressure (#1063), cancel (#1056), the fuzz harness
  (tests/test_tp_allreduce_fuzz.c, 2328-check 200-round green at
  hillclimb/allreduce). MEASURED warm 561-700 us/round, pure-mesh 136 us —
  docs/archive/HILLCLIMB_20260920.md, an archived session log ("Verdict
  (all MEASURED, r3)").
- Where they are ahead: an init-time capability VOTE that disables the
  backend uniformly on any mismatch (payload:88-90) and per-size caps with
  fallback; ours fail at submit with CAPACITY_EXCEEDED
  (tp_device_collective.c) after a create-time topology exchange —
  equivalent fail behavior, less ceremony.
- Where ours is ahead: device-resident in-graph waits are already in the
  transport (the `capture_armed` path in tp_device_collective.c — the S3
  plan EXECUTING), and our S2.5 (device-resident mesh slots) was probed to
  be PHYSICALLY unavailable on GB10 (ibv_reg_mr over VMM device memory =
  EFAULT, dmabuf refused — hillclimb/allreduce commit 79faff3), so the
  100 us route is S3 in-graph, which we have and their README does not claim.
- cf5645a: their DCP all-gather exposed the one op our mesh lacked; the
  ALL_GATHER operation (publish + stripe gather) is implemented and
  CPU-proven in the loopback harness — tp_device_collective.c (gather
  combine), `SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER` in
  include/sparkpipe/spark_tp_device_collective.h, runner binding
  `K3RunnerCombineGatherBf16` in spark_k3_resident_decode_stage_runner.cu, fuzz coverage in
  test_tp_allreduce_fuzz.c (FuzzGatherCheck). MEASURED (CPU loopback, 222
  checks + 20-round fuzz, 0 failures).

### L2. DSpark speculative decoding — verdict: AHEAD on plumbing, EVEN on kernels, the biggest C1 lever

- Theirs: dspark method, nst6 static, fused verify TILE8 kernel
  (VLLM_K3_FUSED_TILE, README.md:97-98,130), fp8 draft path (README.md:99,
  VLLM_K3_FP8_DRAFT README.md:132), adaptive depth opt-in
  (mods/fix-dspark-adaptive-nst/run.sh:24-29 — target =
  floor(mean accepted + 1.5) clamped [1,ceiling], step down direct, step up
  one per window; mods/fix-adaptive-min-depth/run.sh:1-13 — floor default 4).
- Ours: the draft-verify LOOP was already complete and law-pinned —
  K3EngineSubmitDraft (inference/llms/kimi_k3/engine.h), verify planning
  at depth+1 rows per sequence (engine.h K3EnginePlanStep, verify rows root
  + drafts), K3EngineCommitVerify with the EOS truncation law
  (engine.h; tests/test_k3_engine.py K3-009), the KDA state rollback
  fold for accepted prefixes (inference/llms/kimi_k3/slice.cuh
  K3FoldAccepted), the drafter TARGET-TAP extraction on the slice
  (slice.cuh `dspark_aux`), the RedHatAI drafter pack format and
  packer (modules/k3_resident_decode_stage/source/spark_k3_dspark_format.h,
  tools/k3_dspark_stagepack.py) — the same dspark-redhatai class their
  draft_model_path points at (recipes/kimi-k3-dcp-tp16-prd.yaml:17), and the
  fleet keeps the drafter fleet warm (kimi-k3-dflash2-lightseek, dspark-
  redhatai, ...). The qwen drafter precedent pins the drafter forward math
  (tools/qwen36_dspark_reference.py) and the remote-drafter-over-a-socket
  pattern exists on BOTH sides (their mods/v6-remote-dspark/
  patch_remote_dspark.py:1-36 ZMQ RemoteK3DSparkSpeculator; the rtx5090 DFT3
  draft farm, docs/RTX5090_SPECULATION_NODE.md, where only qwen38_27b
  consumes remote drafts today).
- What we lacked and cf5645a landed: the adaptive depth POLICY (their L2
  controller) as a CPU-proven core — inference/llms/kimi_k3/spec_verify.h
  (K3AdaptiveDepth, defaults window 8 floor 4 ceiling 7 = their rule), the
  greedy acceptance resolution from per-row target argmax
  (spec_verify.h K3VerifyResolve, the exact prefix+bonus semantics
  K3EngineCommitVerify consumes), the engine wiring (engine.h
  K3EngineEnableAdaptiveDepth and K3EngineDraftDepth, observe-on-commit in
  K3EngineCommitVerify), and the config/recipe knobs
  (spark_k3_serving_adapter.c K3ServingLoadSpeculation +
  env SPARK_K3_DYNAMIC_DRAFT_DEPTH/WINDOW/MIN_DEPTH mirroring their
  VLLM_DSPARK_* envs). CPU-VALIDATED by
  tests/test_k3_spec_verify.py against the k3 engine (the t1 reference
  oracle stays green).
- Where they are ahead (GPU-gated on our side): the fused verify kernel
  (their TILE8 fuses the 8-row spec verify into the attention kernel; ours
  replays the slice over verify rows, which is correct but pays full per-row
  kernel width) and the fp8 draft KV. Our fused-verify analog for the
  ACCEPTANCE side lands compile-gated: inference/kernels/spec_verify.cuh
  (LmSpecVerifyKernel, instantiated in unity.cu, extern K3SpecVerifyAccept
  in inference/llms/kimi_k3/bind.cu) computes accepted+bonus on device
  so the fold (K3FoldAccepted) and the commit consume without the host scan.

### L3. Marlin MoE nopad (192->256 expert pad skip) — verdict: N/A, our path never pads

- Theirs: KimiMoE pads moe_intermediate 3072->4096 when the TP16 per-partition
  value 192 < 256, inflating MoE weights 89->118.9 GiB/rank; the mod skips it
  (mods/fix-k3-marlin-nopad/patch_marlin_nopad.py:3-13).
- Ours: the K3 MoE path is our own pack+GEMM with explicit tile geometry —
  K3_EXPERT_INTERMEDIATE 3072 (inference/llms/kimi_k3/config.h), TP16
  tile_k 32 with divisibility refusal (K3_PERF.md "TP16 divisibility"; the
  sharder refuses any degree the tile counts do not divide), expert weights packed
  MXFP4 at native width — no 256 pad exists anywhere in the path. Nothing to
  port; their lever is a vLLM-Marlin artifact we never had.

### L4. B12X MLA CuTe attention kernels — verdict: BEHIND on kernel library, EVEN on architecture

- Theirs: B12X_MLA attention backend + Triton MXFP4 MoE unlock on SM12x
  (recipe :90,99, README.md:31,104), tier-1 vLLM kernel PR ports
  (mods/k3-tier1/PORT-REPORT.md: pr53524 router prefetch, pr53525 KDA PDL,
  pr54168 low-M latent MoE tail, pr54697 KDA projection overlap, pr56159
  KDA mixed-batch gather/scatter), the _C rebuild with the MLA cache-kernel
  epilogue #54896 + FP8 CTA swizzle #55180 (README.md:22-24,101,
  mods/perf-pr55356-54896-*, mods/perf-pr55180-*).
- Ours: the full kernel stack is in-tree and owned — persistent GEMM slice
  (inference/llms/kimi_k3/layer.cuh), KDA delta-rule + causal conv +
  bounded decay (unity.cu instantiations), MLA latent attention
  (inference/kernels/attn.cuh LmAttentionDecodeKernel), fused
  residual RMSNorm, AttnRes multi-source mixing, the certified-B1 fp8 head
  (`head_certified_fp8_*` in spark_k3_resident_decode_stage_runner.cu). K3
  graph capture was deleted by a29ea53 (K3_PERF.md "Current code state").
  The gap is PER-KERNEL polish (PDL pipelines, low-M tails, cache epilogues)
  that their fork ports from upstream vLLM; ours are hand-maintained. Our
  measured 55.5 ms stage step sits within about 15% of the 48.6 ms
  bandwidth roofline (K3_PERF.md) while their kernels shave the same class of
  constant.
  No port possible without GPU execution; the compile-gated pieces we CAN
  land were in cf5645a (below).

### L5. Dynamic FP8 quantization at load — verdict: BEHIND, and it is ~1/3 of their gain

- Theirs: at load, dense linears -> MXFP8 and shared experts ->
  fp8_per_block_static, with attention/lm_head/gate excluded
  (README.md:110-124; recipe :116) — attribution: about a third of v4->v5.
- Ours: the K3 pack is MXFP4 experts + BF16 spine (config.h K3_MXFP4_GROUP
  32, K3_KV_BITS 16); the certified fp8 head exists for B1
  (the runner's `head_certified_fp8_*`) but the dense spine streams BF16. A
  spine-quantization wave (MXFP8 dense GEMMs with the same exclusion list) is the single largest
  GPU-gated lever we do NOT have; it multiplies with speculation (bandwidth
  per verify row halves on the dense path).

### L6. DCP8 decode-context-parallel — verdict: BEHIND on deployment, design now unblocked

- Theirs: --decode-context-parallel-size 8 (recipe :94) with the RoCEnante
  all-gather legs (patch_rocante_dcp.py:3-39: head-gather, LSE reduce, KV
  shard gather) — it exists to fit long contexts under a 3 GB KV cap
  (recipe :100).
- Ours: TP16 splits heads (96/16=6, K3_PERF.md "TP16 divisibility"); every rank
  attends the whole context. A context split was never wired.
- cf5645a landed the design as compile-gated code: the online-softmax range
  kernel (inference/kernels/attn.cuh LmAttentionDecodeRangeKernel —
  bounded [begin,end) position scan emitting per-rank m/l/acc partials, the
  same partial structure their LSE-reduce legs carry), the merge kernel
  (attn.cuh LmAttentionRangeMergeKernel — the log-sum-exp merge across
  ranks), the extern entries (bind.cu K3MlaDecodeRange /
  K3MlaRangeMerge), the host partition/coverage/merge math
  (inference/llms/kimi_k3/dcp.h — CPU-TESTED in test_k3_spec_verify.py:
  exact tiling, remainder to leading ranks, empty tails past the context,
  merge == direct softmax), and the transport it rides (the ALL_GATHER op,
  L1). Wiring the MLA call chain to dcp_degree>1 stays GPU-gated: the
  partial buffers, the second collective per MLA layer, and the KV view
  interleave decision need a device to measure.

### L7. Stability guards (their extra) — verdict: EVEN

Their NaN-gumbel, multistream record-stream, grammar fences, draft-noeplb
(README.md:29-31,105-107) are fixes for vLLM-runtime hazards; our equivalent
class is the exact-key chain waits, key re-adopt, QP repair, and the
dispatch gate (hillclimb/allreduce fixes B/C/D/E/F, all deployed+verified).
Different bugs, same discipline, both sides already paid.

## The delta path to >=30 tok/s C1

There is no measured SparkPipe starting point: the first step is an
end-to-end fleet B1 number (K3_PERF.md "Open work").

1. TP16 PP1 (NOT LANDED): the adapter descriptor is TP4xPP4 only, so a TP16
   deployment does not load. Analytically 20.2 tok/s at 15 us per
   all-reduce; ~12.9 tok/s at GLM's measured 167 us (ARITHMETIC,
   K3_PERF.md). Collective latency, not bandwidth, gates this step.
2. Graph capture (REMOVED by a29ea53): exclusive lazy loading makes capture
   ineligible. Restoring it needs pinned experts, as GLM does (78c2c21), or
   relocatable graphs.
3. DSpark nst6 static (cf5645a, plumbing CPU-proven; needs the GPU verify
   run): at their corpus-class acceptance (~3 accepted of 6), the committed
   tokens per step go 1 -> ~4, and verify rows are cheap in the expert
   stream (the TP4xPP4 prefill estimate is 92 tok/s at B8, K3_PERF.md, so an
   8-row step is nowhere near 8x a 1-row step). The 2026-09-21 audit's
   multiplier (1+6)/(1+3) = 1.75 gives 20.2 x 1.75 = ~35 on the analytical
   TP16 base, or 12.9 x 1.75 = ~23 at the measured all-reduce latency
   (ARITHMETIC).
4. Adaptive depth (cf5645a, CPU-proven): protects (3) on corpora with lower
   acceptance; the floor-4 controller keeps the verify pipeline fed through
   dips. (Their 23.59 t/s llama-bench row also runs nst6 speculation, on a
   corpus where they report lower acceptance.)
5. Optional multipliers, GPU-gated: spine MXFP8 (L5, ~1/3-class), S3
   in-graph AR (L1, 136 us measured pure-mesh already), kernel tier-1 class
   polish (L4).

## What cf5645a landed

CPU-PROVEN (executed on the audit host, 2026-09-21):
- spec_verify.h policy core + engine.h adaptive wiring —
  tests/test_k3_spec_verify.py PASS; tests/test_k3_engine.py,
  tests/test_t1_reference_k3.py PASS unchanged (the oracle guards).
- The ALL_GATHER transport op + k3 runner gather combine — CPU loopback
  harness MEASURED: 222 checks 0 failures (basic + FuzzGatherCheck),
  --fuzz 20 0 unrecovered.
- Adapter knobs + recipes (inference/llms/kimi_k3/recipes/*.json,
  example_k3_adapter_config.json) + env mirror of their VLLM_DSPARK knobs.
- docs (this file) + ledger regen.

COMPILE-PROVEN, EXECUTION-GATED (sm_121a gate green on sparkb; never run):
- LmSpecVerifyKernel (fused spec-verify acceptance on device).
- LmAttentionDecodeRangeKernel + LmAttentionRangeMergeKernel (the DCP
  context-split attention halves).
- The dcp.h host math they consume (CPU-tested) + the runner/adapter hooks.

NOT PORTED (with reasons): Marlin nopad (our path never pads, L3), B12X
kernel-library polish (L4 — needs in-situ kernel work on a device), spine
MXFP8 (L5 — a pack-format + kernel wave, GPU-gated), fp8 draft KV (L2 —
needs the drafter pack dual-format + device A/B).
