# qwen36 driver -> hwiface v1 island mapping (qwen27b-dev)

Base: hwiface_v1.md (advisor fold) + hwiface_v1_freeze.md (FROZEN). The freeze
pins the island set E0/L1-L5/F1 and rule R7: v1 primitives and islands are
DSV4-only; generalizing to qwen/glm/k3 is migration step 7, after the DSV4 ROCm
port proves the seams. This document therefore does NOT propose ABI changes -
it prepares the qwen36 port so step 7 is a mechanical application of frozen
decisions, and it flags every place where the qwen36 architecture needs an
island-set EXTENSION (a coordinator/freeze act), not an interpretation.

## 1. Function -> island table

qwen36 = 48 GDN (gated delta net) layers + 16 full-attention layers, dense FFN,
PP-Nx stage slices, MTP chain drafter + DSpark block drafter (device selector).
Launchers: 45 SparkQwen36Launch* externs in module.c against the .cu archive.

| Island (freeze) | qwen36 functions today | File | Notes / class |
|---|---|---|---|
| E0 prologue.embed | LaunchEmbeddingGather via ModuleBeginHidden; stream expand n/a (no hybrid-GDN multi-stream) | module.c | C2 (integer gather) |
| L1 boundary_norm_project | attention_norm RMSNorm + per-layer input projections: GdnLayer qkv/gate/beta/decay linears; AttnLayer query/key/value linears | module.c RunLayer/RunGdnLayer/RunAttnLayer | C3 |
| L2 layer.attention (GDN half) | ConvUpdate, DecayBeta, GdnStep (decode/replay path); ChunkConv+GdnChunk (prefill chunks); GatedNorm; GDN output linear | RunGdnCoreDecode/Replay/Prefill | C3; step-vs-chunk path choice is a LOSSLESSNESS CONTRACT (module.c:1169) - a target may not fuse across it without re-proving bit-identity |
| L2 layer.attention (attn half) | AttnPrepare (rope+q-norm+k/v page write), AttnDecode (paged sparse decode over block table) | RunAttnLayer | C3; paged cache read efficiency is the measured ctx2048 gap (bench analysis #2) |
| L3 layer.cache_transition | KV emission inside AttnPrepare (page-table write); GDN recurrence state_f32 + conv tails in gdn_pool (the recurrent "cache"); snapshot/restore pair | module.c GdnSnapshot, paged_kv.c | split verdict applies: page addresses/emit counters/ring indices C2; payload C3 |
| L4 moe_routed | **n/a** - FFN is dense. Freeze island set has no dense-FFN row: extension needed (see 3) | - | - |
| L5 moe_shared | **n/a** for MoE semantics; the dense FFN maps here structurally (single accumulator leg, fork/join with nothing) or better to a new L4' ffn row | RunFfn (+FfnGateUp fused small-batch path) | C3 |
| F1 head.final | final hcPost fold + HeadShadowQuantize + HeadScreenedArgmax(Score) + MaxLoc pack/unpack; MTP draft argmax (u64 maxloc TP reduce); DSpark selector top-K/gate/lattice walk | EmitHead, RunMtpArgmaxRow, dspark_selector host | token ids/maxloc/feedback integers C2; scores C3. Freeze Q3: markov/confidence heads stay INSIDE F1 - the DSpark selector is exactly that case, no draft.speculate island |

Cross-island machinery (not islands): frame validation (ValidateFrame/
ValidateDecodeView/PrefillView/Speculation), lane continuity
(ValidateLaneSequenceContinuity + Commit/Invalidate), admission
(AdmissionCost/KvPredicate), slot/lane claim (shared SparkStageModule helpers),
TP delta reduce (TpReduceDelta -> collective seam), profiling, bisect dumps
(debug-only, validation-build material per v1 section 7 - candidates for
exclusion from the serving image at port time).

## 2. Primitive usage map (module.c surface -> spark_hw_* family)

The freeze list was written from DSV4's module.c; qwen36 uses the same shape:

- memory: pool/workspace cudaMalloc in AllocatePools/AllocateSlot; pinned host
  staging (host_* arrays in the slot); cudaMemcpyAsync H2D/D2H in UploadRows,
  EmitHead, replay staging; memset on cold-lane state reset.
- queue: one stream per pipeline slot (slot->cuda_stream); TP work uses a
  second stream (state->tp_stream). Queue count 2-5 per process.
- event: none found in module.c hot path (fork/join only inside .cu captures) -
  smaller event surface than DSV4.
- graph: capture/replay exists via the shared stage graph cache contract
  (replay_frame flag names the REPLAY WALK, not graph replay - verify at port;
  the CUDA graphs live in the .cu capture tier).
- collective: SparkQwen36TpState over tp_device_collective (NCCL backend or
  transport fallback) -> exactly the freeze's collective seam row; ordering
  already core-owned (TpReduceDelta call sites), transport target-owned.
- descriptor reads: multiprocessor_count passed into launches; SM121 guard in
  .cu; dynamic-shared budget in kernel attrs - all migrate to SparkHwTarget.

## 3. Extensions the qwen36 port needs from the interface (coordinator acts)

1. Dense-FFN island row. The frozen set only names routed/shared MoE. qwen36
   (and k3?) need either an L5-generalized 'layer.ffn' row or an explicit
   statement that dense FFN rides L5 with an empty fork leg. One sentence in a
   freeze revision; zero ABI impact (islands are named by the core either way).
2. Hybrid-recurrence state channel. The GDN recurrence is a per-lane float
   state + conv tail, not pages: L3's declared handles say "cache pages". The
   port needs the state-pool named alongside cache pages as an L2/L3 state
   channel (it already is one physically; the freeze text just never says so).
3. Drafter packs. MTP weights ride the main pack; the DSpark drafter loads a
   SECOND pack (SPARK_QWEN36_DSPARK_PACK_PATH). Per freeze Q3 both heads stay
   in F1, but the loader story (two logical tensor inventories per stage) is a
   stagepack-format fact the interface fold should acknowledge.

None of these touch spark_hw_* signatures - they are naming/disposition items.

## 4. Sequencing (honest reading of R7)

The freeze is explicit: qwen generalization happens AFTER DSV4 steps 1-6 land.
What this lane can do NOW without violating R7:
- keep this mapping current as the DSV4 S1/S2 splits land (they will name the
  exact spark_hw_iface.h content qwen36 then consumes);
- pre-stage the mechanical part: qwen36's GB10 constants audit (same sweep-list
  method as freeze F4) - done below;
- do NOT fork a qwen-specific spark_hw header early: two neutral headers would
  be a DRY violation and a freeze violation in spirit.

### qwen36 GB10 constant audit (F4 method)

- SPARK_QWEN36_SMALL_BATCH_MAX_ROWS/TILE_N/K_CHUNK (module.c 273-275, mirrored
  in .cu): small-batch GEMM tile selectors -> target-internal, sweepable.
- SM121 capability guard + PTX image trap in .cu -> target fail-closed guard.
- multiprocessor_count runtime query in module.c init -> descriptor field.
- GDN chunk length 64 (SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS) and tap ring
  capacity 2048 are MODEL facts (family header), stay portable-side.
- No other GB10 numeric constant reachable from module.c host control code
  (verified by grep for literal sizes in launch args; kernels take them as
  arguments).

## 5. Declared limits and unsupported features (task 4 list; prefix caching excluded as known-missing, owned elsewhere)

Verified in code 2026-08-22 (file:line):
1. Tensor-core decode attention - NOT implemented (.cu:875 says the wmma tiling
   of the three inner products is the later commit). Biggest measured lever:
   ctx2048 decode loses 21-25% vs ctx512 (BENCH_ANALYSIS_20260822.md #2).
2. Work-control residency layer - kv_client is opened/closed only
   (module.c:2137/2155/3278); the glm52 JIT discipline (lookahead, pressure
   limits, packet-zero priority) is not wired. Mooncake tier works but is not
   driven by work control.
3. Serving context cap 8192 positions (adapter.c:90, owner's KV-limit decision);
   the bench server additionally capped max_model_len 4096, so cells beyond
   4096 remain unmeasured for any implementation.
4. Stale README claims: "Open by design" still lists chunked prefill (landed:
   ChunkConv/GdnChunk) and "MTP weights load and verify today" (landed: the
   adapter drives full draft->verify->replay rounds, plus the DSpark block
   drafter with device-side selector). README needs a refresh, not code.
5. TP>1 requires whole-stack single-stage geometry (module.c Configure: TP only
   when stage_count==1 && slice==whole model) - PP and TP do not compose in
   this driver today.
