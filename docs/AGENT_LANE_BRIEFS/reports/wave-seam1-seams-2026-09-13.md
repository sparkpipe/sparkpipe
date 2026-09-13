# SEAM-1 common-module survey — qwen38max, qwen4_flash, qwen38-27b, glm53full (glm52), glm53flash (glm5_next), hy4 (2026-09-13)

Operator directive: identify every driver-side module that should be COMMON and
PARAMETERIZED. This is the driver-side complement to the DRY-1 packer study
(#976, `docs/DRY_PACKBUILDER_PROPOSAL.md`): DRY-1 owns `tools/*_stagepack.py`,
`*_verify.py`, `*_experts_manifest.c`; this survey owns
`modules/<family>_resident_decode_stage/` and `model-families/<family>/`, plus
duplication DRY-1 could not see from the tools side. Baseline: main `6dbb58e`.
Similarity figures are difflib SequenceMatcher on line sequences after
family-token normalization (all `qwen38max/qwen38_max/Qwen38Max/...` variants
mapped to one token), so "100%" means identical modulo the family name.

## 0. Verdict in one paragraph

The six families are two codebases stamped three times each. The qwen trio
(max / flash / 27b) share one GDN + attention CUDA kernel suite where 22
kernels are identical modulo name (one carries the wrong family's name —
`SPARK_QWEN38_ROUTER_SORT_CAPACITY` verbatim inside qwen4_flash), and the glm
pair (glm52 / glm5_next) share a CUDA tree where `unity.cu` is 87% and `api.h`
93% pure token rename. Cross-superfamily similarity is 3-17%: nothing is shared
in source even where the shape is identical (pack-load/bind, KV slot pools,
validation harnesses, serving adapters). Estimated recoverable: ~15,000-17,000
of ~53,800 driver-side C/H/CU LOC (28-31%) plus ~1,900 of Makefile/script
boilerplate, with a per-driver `llm_defines.h` of roughly 200 keys as the
parameter carrier.

## 1. File inventory

LOC = C/H/CU/CUH only. Purposes: module.c = stage lifecycle + KV frame + pack
bind + TP init + layer loop; cuda.cu = kernel suite + launchers;
serving_adapter.c = JSON deployment config + TP collective + submission
serving; stagepack_format.h = pack wire kinds/geometry; firmware.h = runtime
caps; model.h = model geometry (the natural llm_defines.h carrier).

### qwen38_max (modules 6,124 + model-families 198 = 6,322)

| file | LOC | purpose |
|---|---|---|
| source/spark_qwen38_max_resident_decode_stage_module.c | 2,031 | lifecycle, KV slot pool + 249-line frame prep, MoE, MTP bind, weightd lazy pack |
| source/spark_qwen38_max_resident_decode_stage_cuda.cu | 1,668 | GDN suite + attention + MoE + router sort 512 + argmax kernels |
| source/spark_qwen38_max_serving_adapter.c | 374 | MTP provider + speculation seam bind (thin, uses template) |
| source/spark_qwen38_max_stagepack_format.h | 319 | pack kinds/geometry (extends common header) |
| include/...firmware.h | 293 | runtime caps |
| tools/qwen38_max_pack_synthesize.c | 71 | synthetic pack builder for tests |
| validation/*cuda_validation.cu | 1,363 | GPU numerical gate |
| validation/validate_*.sh + publish wrapper | 93 | harness shell |
| model-families: model.h / work_control.h/.c | 198 | geometry + work control stub |

### qwen4_flash (modules 8,554 + model-families 217 = 8,771)

| file | LOC | purpose |
|---|---|---|
| source/spark_qwen4_flash_resident_decode_stage_module.c | 2,530 | superset of max's module.c + PLE, indexer, HC inject/readout, MTP draft chain |
| source/spark_qwen4_flash_resident_decode_stage_cuda.cu | 2,587 | superset of max's kernel suite + PLE/indexer/HC/MaxLoc/E8m0/tp-u64max |
| source/spark_qwen4_flash_serving_adapter.c | 288 | thinnest adapter, template user |
| source/spark_qwen4_flash_stagepack_format.h | 548 | 5 release arms, extends common header |
| include/...firmware.h | 327 | runtime caps + HC/indexer/PLE knobs |
| tools/qwen4_flash_pack_synthesize.c | 78 | synthetic pack builder |
| validation/*reference.c | 507 | CPU GDN oracle (twin of 27b's) |
| validation/*cuda_validation.cu | 1,684 | GPU numerical gate |
| model-families: model.h / work_control.h/.c | 217 | geometry (66 defines, richest qwen key set) |

### qwen38-27b (modules 11,435 + model-families 248 = 11,683)

| file | LOC | purpose |
|---|---|---|
| source/spark_qwen38_27b_resident_decode_stage_module.c | 3,326 | third-generation module (drifted from qwen siblings, 10%) |
| source/spark_qwen38_27b_resident_decode_stage_cuda.cu | 2,154 | shared GDN/attention suite + RANS decoder + small-batch GEMM + accum/relay |
| source/spark_qwen38_27b_serving_adapter.c | 2,322 | speculative frame server: prefix publish/borrow/cover, dflash2 fold |
| source/spark_qwen38_27b_tp.c/.h | 346 | wraps common SparkTpDeviceCollective + family geometry |
| source/spark_qwen38_27b_dspark_cuda.cuh + dspark_format.h | 487 | draft-model rANS kernels + wire |
| source/spark_qwen38_27b_native_ws.cuh | 266 | native workspace kernels |
| source/spark_qwen38_27b_stagepack_format.h | 476 | pack kinds (extends common header) |
| include/...firmware.h | 304 | runtime caps |
| tools/qwen38_27b_pack_synthesize.c | 62 | synthetic pack builder |
| validation/*reference.c | 507 | CPU GDN oracle (twin of flash's) |
| validation/*cuda_validation.cu | 1,180 | GPU numerical gate |
| model-families: model/runtime_contract/serving_constants/work_control | 248 | geometry + serving knobs (16 defines) |

### glm53full / glm52 tree (modules 8,669 + model-families 704 = 9,373, plus dspark backend 3,077)

| file | LOC | purpose |
|---|---|---|
| source/spark_glm52_resident_decode_stage_module.c | 2,157 | lifecycle, pack assign/load, TP chain, contract hash |
| source/cuda/config.h | 69 | short-token key map onto model.h (38 defines) |
| source/cuda/layer.cuh | 1,326 | MLA/DSA layer template, GEMM tiles, buffers |
| source/cuda/unity.cu | 276 | expert-codec unity build (rename-copy of glm5_next's) |
| source/cuda/api.h + launch_shape.h | 45 | entry decls + launch shape |
| source/spark_glm52_resident_decode_stage_cuda.cu | 581 | host launchers |
| source/spark_glm52_resident_decode_stage_internal.h | 168 | internal state |
| source/spark_glm52_serving_adapter.c | 990 | template-user adapter, draft bridge, reset control |
| source/spark_glm52_stagepack_format.h | 286 | OWN kind enum + payload types (bypasses common header) |
| include/...firmware.h + batch_tuning.h | 212 | caps + batch variants (23 defines) |
| validation/*cuda_validation.cu | 2,554 | GPU numerical gate |
| model-families/glm52: model/kv_geometry/rope/mtp_tree/dspark/chat_template.h + chat_template.c | 704 | geometry (77), kv capacity request, rope tables, MTP tree, tokenizer chat template |
| modules/glm52_dspark_draft_backend/ (companion module) | 3,077 | draft model CUDA backend + dispatch policy |

### glm53flash / glm5_next (modules 15,694 + model-families 229 = 15,923)

| file | LOC | purpose |
|---|---|---|
| source/spark_glm5_next_resident_decode_stage_module.c | 4,010 | superset of glm52 module + CUDA graph capture, KDA recurrent, MTP tree drive, page tables |
| source/cuda/config.h | 84 | short-token key map (55 defines) |
| source/cuda/layer.cuh | 2,513 | glm52 layer.cuh + KDA + HC sinkhorn + indexer-kv |
| source/cuda/unity.cu | 287 | rename-copy of glm52's |
| source/cuda/api.h + launch_shape.h + index_kv.cuh | 66 | entry decls + latent KV index |
| source/spark_glm5_next_resident_decode_stage_cuda.cu | 1,190 | host launchers + MTP draft ops |
| source/spark_glm5_next_resident_decode_stage_internal.h | 290 | internal state |
| source/spark_glm5_next_tap_ring.c/.h | 278 | tap ring sidecar |
| source/spark_glm5_next_serving_adapter.c | 1,409 | BYPASS: hand-rolled TP rails/session-ports JSON (~370 LOC) replacing the template |
| source/spark_glm5_next_stagepack_format.h | 486 | OWN kind enum (bypasses common header) |
| include/...firmware.h + batch_tuning.h | 275 | caps + batch variants (22 defines) |
| tools/glm5_next_pack_synthesize.c | 290 | synthetic pack builder (5x its qwen twins) |
| validation: cuda_validation 1,829 + mtp_parity 1,209 + tap_ring 1,170 + flash_decode_cell 280 | 4,488 | GPU gates (glm52 has none of the last three) |
| model-families/glm5_next: model.h + kv_geometry.h + name_map.json + tensor_patterns.json | 229 + 4,352 data | geometry (101 defines — largest), kv capacity request twin |

### hy4 (modules 1,644 + model-families 65 = 1,709)

| file | LOC | purpose |
|---|---|---|
| source/spark_hy4_resident_decode_stage_module.c | 188 | rung-1 lifecycle skeleton, honest UNSUPPORTED execute |
| source/spark_hy4_resident_decode_stage_cuda.cu | 282 | rope + gate + fused-norm kernels |
| source/spark_hy4_stagepack_format.h | 472 | pack kinds (extends common header) |
| source/spark_hy4_stagepack_format_check.c | 48 | format check |
| source/spark_hy4_fp8_scale_contract.h | 159 | FP8 2D block scale contract |
| include/...firmware.h | 231 | caps |
| tools/hy4_pack_convert.c | 264 | pack converter |
| model-families/hy4: model.h | 65 | geometry (48 defines) |

No serving adapter, no validation .cu, no weightd attach: hy4 is the rung-1
model citizen for seams (uses stage_module_common + tp_device_collective
headers) but has not grown the duplicated plumbing yet.

## 2. Copy-cluster map

Whole-file normalized similarity (difflib) and, for the two big code families,
shared-function identity counts. "Pattern-identical" = same code, different
constants only.

### Cluster A — qwen CUDA kernel suite (cuda.cu trio; 6,409 LOC)

Pairs: max-vs-flash 63.6% whole-file, 22 shared functions of which 22 at >=86%
and 22 exactly 100% at kernel level (GdnStep 89.2, AttnPrepare 96.9,
AttnDecode 98.3, ChunkStep 98.4, ChunkQkDecay/ChunkSolve/EmbeddingGather/
GateScores/GateSelect/ResidualAdd/RopeFrequency/SwiGlu/TpCombineAdd/
WarpReduceMax/HeadArgmax... 100.0). flash-vs-27b 51.0% whole-file, 19 kernels
at 100%. max-vs-27b 54.4%.
Pattern-identical with different constants: every kernel differing only in
dimension/token constants. Family-local arms: flash adds PLE (4 kernels),
Indexer (2), HC (5), HeadMaxLoc (2), E8m0, sharded embedding, tp-u64max
(~700 LOC); 27b adds rANS (3 device fns), small-batch GEMM (3), accum/relay
(4), frame error, TP geometry (~600 LOC).

### Cluster B — GLM CUDA tree (config.h/layer.cuh/unity.cu/api.h; 4,663 LOC pair)

unity.cu 87.4% (pure rename: tiles 128/64/2/8 identical), api.h 92.9%,
batch_tuning.h 90.9%, config.h ~50% key-for-key, internal.h 61.6%,
layer.cuh 51.6% whole-file with the entire glm52 MLA/DSA/GEMM skeleton shared
and glm5_next adding KDA state + HC sinkhorn + indexer (~1,200 LOC).
Makefiles 83.5% similar on top of the already-common rules.mk.

### Cluster C — qwen module.c trio (7,887 LOC)

max-vs-flash 53.8% whole-file; 8 shared functions at 100% including
SparkModuleKvPrepareFrame (249 lines in BOTH, byte-identical modulo names),
KvEvictSlot (39), ModuleDescribe/ExecuteFrame/AllocatePools (96.3)/Configure
(90.1)/UploadRows (91.5). flash extends with PLE/HC/MTP-draft/prefill runs.
27b drifted: 10.7% vs max (third-generation rewrite) but still restates the
same shape: KV slot arrays (lane/logical/sequence/dirty/pinned/free_stack),
pack validate/load/assign, allocate-slot family, environment toggle blocks.

### Cluster D — GLM module.c pair (6,167 LOC)

glm52 vs glm5_next 42.9% whole-file; 26 shared functions, 17 at 100%
(PackValidateRanges 28, PrepareAsyncCompletion 39, ValidateFrame 44,
AllocateSlotHead, BuildHeadShadow, CombineBf16/U64Max, RoundMajorWaveRows,
StageHostBatch...), the rest 47-92%. glm52 is effectively a strict subset;
glm5_next adds graph capture (SparkGraph* x8), MTP (x4), KDA recurrent (x4),
device page copy, worker completion.

### Cluster E — serving adapters (5,383 LOC across 5 files)

glm52-vs-glm5_next 48.4% whole-file, isomorphic 27-handler skeleton
(LoadTpCollective, LoadConfiguration, InitializeSpeculationSeam, ValidateRow
Order, ReservePending, OrphanDriverCompletion, DriverCompletion, DriverWake,
AvailableSubmissionCount, Destroy, LoadDriver, ValidateConfiguration,
Initialize, ValidateBoundaries, ValidateSubmission, BuildFrame, Admit,
CacheContext, Prefetch, ResolvePrefetch, Submit, Progress, Quiesce, Snapshot,
ResetControl, Reset, GetInterface). glm5_next hand-rolls what the template
gives glm52 (see section 4). qwen38-27b's 2,322-LOC adapter is a parallel
third-generation implementation of the same machinery plus speculative
prefix/publish/cover and dflash2 fold — pattern-identical structure (load
config JSON, validate row order, pending slots, build/run frame), 15-17%
textual. qwen38_max (374) and qwen4_flash (288) are the thin template users.

### Cluster F — stagepack format headers (2,587 LOC, 6 files)

qwen trio 42-63% similar, all extending the COMMON
include/sparkpipe/spark_stagepack_format.h; hy4 also extends the common
header. glm52/glm5_next bypass it with their own enums and payload types
(DRY-1 already proved the wire's directory entry is cross-family identical).
The kinds map is re-stated twice more by the verifiers (DRY-1 finding).

### Cluster G — validation + harness (11,500 LOC + 6 shell scripts)

reference.c pair: 507 = 507 lines, 83.2% (rename-only differences) — a CPU
GDN oracle copied verbatim. val_cu qwen trio 31.7-66.6%; glm52-vs-glm5_next
14.5% (drifted hard). validate_*.sh sextuplets 37-86% (q27-vs-flash 86.4%).
Makefiles 60-71% within qwen trio, 83.5% glm pair (rules.mk already common;
the residue is duplicated target blocks).

### Cluster H — deployment/JSON + env plumbing (cross-family pattern)

Identical JSON member sets with per-family forks:
schema_version, model_revision, stage_pack_path, max_sequence_positions,
tp_degree, tp_rank, draft_bridge_host/port (+ glm52-only
execution_row_capacity, decode_split_context_threshold, expert_weight_codec;
glm5_next adds the tp_collective superset: algorithms, backends, rails,
session_ports, session_ports_hc, split_ring_min_payload_bytes,
counter_rotating_split_ring, direct_all_to_all_max_payload_bytes).
Environment flags restated per family with family prefixes for identical
semantics: STAGE_KV_STORE, DEBUG_SKIP_MOE, DEBUG_SKIP_GDN, DEBUG_DUMP_HIDDEN,
TP_STANDALONE (50 getenv calls across the six module.c/adapters).

### Cluster I — small constants restated (the audit seeds, confirmed)

- `SPARK_QWEN38_ROUTER_SORT_CAPACITY 512u` defined identically in qwen38_max
  cuda.cu:1341 AND qwen4_flash cuda.cu:2063 — under the WRONG family's name.
- `MAX_ACTIVE_SEQUENCE_COUNT 512u` restated in all three qwen firmware
  headers (runtime cap posing as a family constant).
- Rope frequency formula `exp2f(-(2*pair/dim)*log2f(theta))` copied 5x
  (qwen38_max, qwen4_flash, qwen38_27b x2, glm layer.cuh variant).
- KV page/slot multipliers: GLM pair's kv_geometry.h twins (78.2% normalized)
  with a FillCapacityRequest differing in exactly 7 constants, both using
  #ifndef guards; KV_BLOCK_TOKEN_COUNT 64 in both.
- GLM launch tiles 128/64/2/8 and HEAD_TILE 1024 restated in both trees.

### Tools (referenced, not re-measured)

DRY-1 (#976, docs/DRY_PACKBUILDER_PROPOSAL.md) already measured and proposed:
stagepack_core + descriptor-verified kind map + one experts-manifest utility;
13,748 tool LOC -> ~4,850. My families hold 6,247 packer LOC
(qwen38 1,062; q4f 1,398; q27 900; glm52 599+1,051; glm5_next 994; hy4 392),
~890 verifier LOC, and 5 experts-manifest copies (qwen38max 305, glm52 195,
glm5_next 194, hy4 103 + v2 332). One driver-side overlap DRY-1 could not
see: tools/qwen38_pack_verify.py imports the 27b packer's tables — the
driver-side format cluster (F) is what makes that cross-import necessary.

## 3. Parameter surface per proposed common module

The per-driver include-path `llm_defines.h` holds ALL values; today they are
scattered across model.h (47-101 defines), firmware.h (16-48), config.h
(38-55), batch_tuning.h (22-23), serving_constants.h (16), format headers
(10-25) — a per-family total of 81-211 keys over an estimated union of ~200.

1. `common_gdn_stage_kernels` (Cluster A; seed qwen38_max, absorb flash/27b).
   Varies: hidden, layers, vocab, attn heads/kv heads/head dim, rope dim/
   theta, GDN key/value heads, GDN head key/value dims, conv kernel, chunk
   tokens, attention period + full-attn phase, MoE routed/shared counts,
   experts-per-token, expert/shared intermediate, mxfp4 group, fp8 block,
   swiglu limit, rms epsilon, TP degree/rank passthrough, launch shapes
   (router sort capacity, block sizes), context length. ~45 keys.
   Family-local (stays driver-side): flash PLE/indexer/HC/MaxLoc/E8m0 arms,
   27b rANS + small-batch GEMM + accum relay.
2. `common_kv_frame` (Cluster C subset; seed the two 249-line
   KvPrepareFrame twins). Varies: kv block count, page slots, slot bytes,
   lane count, cache layer/block strides, max active sequences, pipeline
   slots. ~10 keys.
3. `common_stagepack_format_ext` (Cluster F; one kind map + 64 B directory
   entry + geometry table + header variants). Varies: magic, format version,
   alignment (256 shared), revision/sha byte widths, flags (MTP), payload
   types, tensor kinds, header sizes (120/128/v3), per-family geometry rows.
   ~25 keys.
4. `common_serving_tp_config` (Cluster H; template call + optional rails/
   session-ports policy). Varies: schema version, model id/revision, codec
   name, execution row capacity, decode split threshold, draft bridge,
   tp collective block (backend kind/path, identifier, listen/peer/control
   ports, algorithms, thresholds, timeouts, topology, step rails, rail
   hosts, session ports, max payload bytes). ~22 keys.
5. `common_serving_frame` (Cluster E; pending/completion/wake/validate/
   submit/progress/quiesce/snapshot/reset skeleton). Varies: lane count,
   pending capacity, boundary policy flags, cache admission policy,
   speculative draft count policy, screen/head output policy. ~12 keys of
   which ~4 behavioral flags.
6. `common_glm_cuda_tree` (Cluster B; one config.h key space + unity.cu +
   layer template). Varies: full glm model key set (77 vs 101 defines:
   MLA/DSA dims, latent dims, KDA layer counts, HC sinkhorn iterations,
   indexer dims), expert codec name, launch tiles. ~60 keys; KDA + HC +
   indexer-kv stay glm5_next-local behind the same interface.
7. `common_validation_oracle` (Cluster G; one reference.c + one harness
   shell). Varies: geometry subset, tolerance table, layer mix. ~12 keys.
8. `common_kv_geometry` (Cluster I; FillCapacityRequest twin). Varies: 7
   keys (layout, layers, compressed dim, position dim, bytes/scalar, fp8
   block, index-key triple).
9. `llm_defines.h` carrier: one unprefixed key space per driver include
   path; the per-family headers above become views over it (they keep
   working as `#define SPARK_<FAM>_X LLM_X` shims during migration).

## 4. The already-common seams — used, bypassed

| seam | used by | bypassed by | the violation |
|---|---|---|---|
| modules/resident_decode_stage_rules.mk | all 6 | none | glm pair still carries 83.5%-similar duplicate target blocks on top |
| include/sparkpipe/spark_stage_module_common.h (ledger, admission, env, pack read, timing, fork/read-ahead) | all 6 | none | qwen trio restate env-flag sets instead of one policy table |
| include/sparkpipe/spark_serving_adapter_template.h | qwen38_max, qwen4_flash, qwen38-27b, glm52 | glm5_next | glm5_next hand-rolls ~370 LOC of TP collective/rails/session-ports JSON the template's LoadTpCollective + policy already does |
| include/sparkpipe/spark_stagepack_format.h (common kinds + 64 B entry) | qwen trio, hy4 (extend it) | glm52, glm5_next (own enums) | DRY-1 proved the wire identical cross-family; the glm fork is the reason the verifier kind map must exist twice |
| include/sparkpipe/spark_tp_device_collective.h + ring/transport | q27 (wraps), glm pair, qwen modules | none found | glm5_next's adapter-side parsing bypass is the seam violation, not the transport |
| weightd client seam (attach/lazy_pack/lease/map) | qwen38_max, glm52, glm5_next | qwen4_flash, qwen38-27b (direct pack read), hy4 (n/a) | two pack-acquisition paths coexist for identical formats |
| runtime/model_resident_client + serving adapter registry (SparkModelServingAdapterGetInterface) | uniform on the module side | hy4 has no adapter yet | — |
| tools/spark_pack_common.py | 14 tools (DRY-1) | — | per DRY-1 |

## 5. Unique math worth extracting even at one driver

- rANS GPU entropy decoder (qwen38-27b dspark: RansBuildTable /
  RansStageChunk / RansDecodeTileHalf + LaunchSmallBatchLinearRans) — generic
  symmetric-entropy decode, one implementation exists.
- Sinkhorn-normalized routing (glm5_next HcSplitSinkhornKernel + HC
  low-rank mix) — generic constrained-assignment kernel.
- KDA chunked recurrent replay (glm5_next SparkRecurrent*/KdaReplay + the
  kda_* weight plumbing; spec'd in docs/GLM_KDA_REFERENCE.md and
  GLM_KDA_STATE_LAYOUT.md) — generic linear-attention state machinery.
- Certified-FP8 screened argmax head (qwen trio HeadScreenedArgmax family;
  glm5_next's SparkLmHostLaunchHeadCertifiedFp8B1WithScore already
  name-identical to a glm52 twin) — generic quantized-head decoding.
- hy4 FP8 scale contract (spark_hy4_fp8_scale_contract.h, 2D block scale
  propagation e4m3/e8m0) — the fleet's only written-down scale contract;
  generic for any FP8 driver.
- glm52 interleaved RoPE table builder (spark_glm52_rope.h static inline +
  the 5x copied frequency formula) — one rope module, parameterized by
  (dim, theta, interleave).
- glm52 chat template renderer (spark_glm52_chat_template.c) — family
  strings vary, engine shape is generic tokenizer-binding material.

## 6. Recommended conversion order (cheapest identity proofs first)

1. glm pair unity.cu + api.h + batch_tuning.h (87-93% rename-copies) —
   byte-identity provable in one build.
2. The two 249-line KvPrepareFrame twins into common_kv_frame.
3. reference.c oracle twin into common_validation_oracle.
4. glm5_next serving adapter onto the serving template (deletes ~370 LOC and
   the second TP JSON parser in one move).
5. qwen38_max + qwen4_flash module.c pair merge (53.8%), then the kernel
   suite (Cluster A) — largest LOC, needs the llm_defines key space first.
6. Stagepack format ext (with DRY-1's descriptor-verify landing).
7. qwen38-27b reabsorption last: it is three generations drifted and needs
   the speculative-frame common first.

## 7. Surprises

1. qwen4_flash ships qwen38's name: `SPARK_QWEN38_ROUTER_SORT_CAPACITY` at
   qwen4_flash cuda.cu:2063 — the copy is literal enough to carry the wrong
   family's identifier.
2. SparkModuleKvPrepareFrame is 249 lines, twice, byte-identical modulo
   names — nobody ever diffed the two module.c files.
3. glm5_next bypassed a common seam that glm52 uses in the same repo: two
   TP-collective JSON parsers coexist, one template-backed, one hand-rolled.
4. The glm pair's Makefiles are 83.5% identical DESPITE the shared rules.mk —
   the rules were extracted and then the targets re-grew per family.
5. qwen38-27b is the drift warning shot: 10.7% similarity to its own siblings
   proves that without a common trunk, three generations of the same driver
   can coexist invisibly.
6. hy4 (smallest, newest) is what the operator wants: a 188-LOC lifecycle
   shell over the common seams with honest UNSUPPORTED — it has simply not
   yet had time to grow its 5,000 LOC of copies.
