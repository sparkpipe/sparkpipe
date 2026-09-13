# SEAM-2 common-module survey — k3, dsv5, ling, gemma4, laguna, muse, minimax-h3 (2026-09-13)

Operator directive: identify every driver-side module that should be COMMON and
PARAMETERIZED. The drivers are copy-paste islands; this report maps the islands,
measures the duplication, and lists what varies per driver so each cluster can
become one common module plus one include-path parameter file (`llm_defines.h`
style). Baseline: main `6dbb58e`. Minimax tree read from `origin/lane/minimax-driver`
(fetched as FETCH_HEAD in this survey; read-only).

Companion machine-readable artifact:
`docs/AGENT_LANE_BRIEFS/reports/wave-seam2-seams-2026-09-13.seams.json`.

Packer/verifier TOOLS are out of scope here: DRY-1 (`docs/DRY_PACKBUILDER_PROPOSAL.md`,
#976) already measured the pack-builder corpus and proposed the universal packer.
This report covers the DRIVER side (`modules/*_resident_decode_stage/`,
`model-families/*/*`) plus duplication DRY-1 did not analyze (host glue, CUDA
cells, serving adapters, validation rigs, firmware headers, model-defines headers,
per-family Makefiles and deployment generators).

## 1. File inventory (LOC, purpose)

### k3 (kimi) — `modules/k3_resident_decode_stage` + `model-families/k3`

| file | LOC | purpose |
|---|---|---|
| source/spark_k3_resident_decode_stage_runner.cu | 1292 | stage orchestration: TP4 island combine/reduce, weightd lazy acquire/release, manifest check, head exchange, slice launch |
| source/spark_k3_serving_adapter.c | 656 | serving adapter over `spark_serving_adapter_template.h`; speculation seam, device-collective plumbing |
| source/spark_k3_resident_decode_stage_cuda.cu | 427 | hand-rolled KDA scratch/arena carve + pool sizing |
| source/spark_k3_pack_load.c | 257 | private pack open/validate/load |
| source/spark_k3_dspark_format.h | 161 | speculative-draft pack format |
| include/.../spark_k3_batch_tuning.h | 152 | hand-rolled bucket ladder (b1..b1024) |
| include/.../spark_k3_resident_decode_stage_runner.h | 133 | runner interface |
| source/spark_k3_bind.c | 121 | per-layer weight bind |
| include/.../spark_k3_resident_decode_stage_cuda.h | 82 | cuda side interface |
| include/.../spark_k3_pack_load.h | 72 | pack loader interface |
| source/spark_k3_resident_decode_stage_module.c | 58 | module init/destroy, slice derivation |
| include/.../spark_k3_pool_sizing.h | 50 | slice pool sizing math (MLA+KDA) |
| include/.../spark_k3_bind.h | 38 | bind interface |
| include/.../spark_k3_resident_decode_stage_module.h | 37 | module state |
| include/.../spark_k3_serving_adapter.h | 16 | adapter vtable decl |
| include/.../spark_k3_weightd_include.h | 14 | weightd client include seam |
| model-families/k3/.../spark_k3_model.h | 128 | model constants (93 layers: 24 MLA + 69 KDA) |
| model-families/k3/.../spark_k3_runtime_contract.h | 48 | runtime contract pins |
| model-families/k3/.../spark_k3_kv_geometry.h | 48 | kv capacity-request filler (MLA latent + KDA slot bytes) |

k3 total: 3,411 module + 224 family. k3 is the odd generation: it consumes
`inference/llms/kimi_k3/layer.cuh` (the older per-model include tree) and
hand-rolls scratch, pack load, and bucket tuning; it does NOT use
`inference/kernels/*`, `runtime/gemm.cuh`, `spark_stage_module_common`,
`spark_pack_load_common`, or `spark_batch_variant_tuning_common`.

### dsv5 / dsv41flash — `modules/dsv4_resident_decode_stage` + `model-families/dsv4`

| file | LOC | purpose |
|---|---|---|
| source/spark_dsv4_resident_decode_stage_module.c | 6198 | monolith: configure, kv tiers, PLE, MTP draft chain, HC readout, graph capture/instantiate, MoE, GDN core decode |
| source/spark_dsv4_resident_decode_stage_cuda.cu | 3553 | MLA + sparse-attention + dspark kernels |
| validation/..._cuda_validation.cu | 1604 | cuda validation gate |
| source/spark_dsv4_serving_adapter.c | 1415 | serving adapter over template |
| source/spark_dsv4_stage_runner.c | 514 | stage runner (host) |
| source/spark_dsv4_stagepack_format.h | 407 | stagepack structs + tensor kinds |
| source/spark_dsv4_paged_cache.c | 406 | paged kv cache |
| source/spark_dsv4_dspark_kernels.cuh | 392 | draft-model kernels |
| include/.../spark_dsv4_resident_decode_stage_firmware.h | 264 | ABI constants |
| source/spark_dsv4_jit_kv.c | 231 | jit kv path |
| source/spark_dsv4_pool_layout.h | 226 | pool layout |
| include/.../spark_dsv4_batch_tuning.h | 172 | bucket ladder (own copy) |
| include/.../spark_dsv4_resident_decode_stage_runner.h | 147 | runner interface |
| source/spark_dsv4_dspark_pro_kernels.cuh + pro_chain.cuh | 289 | pro draft chain |
| Makefile + Makefile.pro | 188 | builds (two variants) |
| source/spark_dsv4_jit_kv.h + paged_cache.h + lane_continuity.h + sparse_attention_split.h + hc_splitk.h + serving_adapter.h | 336 | interfaces |
| model-families/dsv4/src/spark_dsv4_cache_plan.c | 1037 | kv plan/arena planner |
| model-families/dsv4/src/spark_dsv4_parallel_shape.c | 153 | parallel shape math |
| model-families/dsv4/src/spark_dsv4_cache_arena.c | 102 | arena |
| model-families/dsv4 headers (model, pro_model, aliases, runtime_contract, cache_plan, cache_arena, parallel_shape) | 627 | constants + interfaces |

dsv4 total: 15,268 module + 1,919 family. Largest module; internal diversity is
genuine (MLA + MTP + PLE + HC + graph capture), but its bucket tuning,
stagepack format, and pack-load layers still duplicate the fleet pattern.

### ling — `modules/ling_resident_decode_stage` + `model-families/ling`

| file | LOC | purpose |
|---|---|---|
| validation/..._cuda_validation.cu | 2847 | validation gate |
| source/spark_ling_resident_decode_stage_module.c | 1959 | module incl. hand-rolled pack validate/load (~450 LOC) + KDA/KV/slot lifecycle |
| source/spark_ling_serving_adapter.c | 1247 | hand-rolled serving adapter incl. TP rail/session-port deployment config |
| source/cuda/layer.cuh | 1228 | per-layer cell over shared inference/kernels (KDA conv/delta-rule/decay, MLA latent attention, MoE) |
| source/spark_ling_resident_decode_stage_cuda.cu | 585 | buffer sizing, upload, launch entry |
| source/spark_ling_stagepack_format.h | 422 | stagepack structs + tensor kinds |
| tools/ling_pack_synthesize.c | 297 | synthetic pack builder for validation |
| source/cuda/unity.cu | 247 | gemm binding + expert weight codec selection |
| source/spark_ling_resident_decode_stage_internal.h | 170 | internal buffers |
| include/.../spark_ling_resident_decode_stage_firmware.h | 141 | ABI constants |
| Makefile | 118 | build |
| source/cuda/config.h | 72 | model defines fold |
| include/.../spark_ling_batch_tuning.h | 57 | bucket ladder via common |
| source/cuda/api.h | 42 | gemm/codec extern decls |
| source/cuda/launch_shape.h | 3 | thread counts |
| model-families/ling/name_map_lingfin.json | 478 | weight name map |
| model-families/ling/.../spark_ling_model.h | 147 | model constants (78 layers, MLA+KDA hybrid) |
| model-families/ling/name_map.json + tensor_patterns.json + kv_geometry.h | 165 | pack naming, patterns, kv filler |

ling total: 8,809 module + 815 family.

### gemma4 (31b + 26b) — `modules/gemma4_resident_decode_stage` + `model-families/gemma4`

| file | LOC | purpose |
|---|---|---|
| validation/..._cuda_validation.cu | 1904 | validation gate |
| source/spark_gemma4_resident_decode_stage_module.c | 1349 | module incl. pack load via `spark_pack_load_common.h` (token-pasted) |
| source/spark_gemma4_stagepack_format.h | 482 | stagepack structs + tensor kinds |
| source/spark_gemma4_resident_decode_stage_cuda.cu | 423 | sliding/full attention launches over shared kernels |
| source/spark_gemma4_serving_adapter.c | 307 | serving adapter over template |
| tools/gemma4_pack_synthesize.c | 244 | synthetic pack builder |
| include/.../spark_gemma4_resident_decode_stage_firmware.h | 224 | ABI constants |
| Makefile.moe / Makefile | 133 | 26b-a4b (defines `SPARK_GEMMA4_MOE_BUILD`) / 31b builds |
| validation/validate_..._cuda.sh | 53 | gate script |
| model-families/gemma4/.../spark_gemma4_model.h | 72 | 31b constants |
| model-families/gemma4/.../spark_gemma4_moe_model.h | 61 | 26b-a4b constants |
| model-families/gemma4/.../spark_gemma4_moe_model_aliases.h | 59 | alias layer mapping generic `SPARK_GEMMA4_MODEL_*` to variant |

gemma4 total: 5,102 module + 192 family. It is the fleet's PROOF OF CONCEPT for
one driver source tree serving two model sizes: one source, two Makefiles, two
contract SHAs, an alias header re-pointing the generic key namespace.

### laguna — `modules/laguna_resident_decode_stage` + `model-families/laguna`

| file | LOC | purpose |
|---|---|---|
| source/spark_laguna_resident_decode_stage_module.c | 2313 | module incl. hand-rolled pack load |
| source/spark_laguna_serving_adapter.c | 1225 | hand-rolled serving adapter (ling's twin) |
| validation/laguna_layer7_realpack.cu | 1107 | realpack layer-7 equivalence rig |
| source/cuda/layer.cuh | 928 | per-layer cell (GQA full/sliding + YaRN rope + MoE) |
| validation/laguna_layer7_reference.py | 709 | python reference for layer 7 |
| source/spark_laguna_resident_decode_stage_cuda.cu | 622 | buffers, YaRN inv-freq upload, launches |
| source/spark_laguna_stagepack_format.h | 346 | stagepack structs + tensor kinds |
| tools/laguna_pack_synthesize.c | 325 | synthetic pack builder |
| source/cuda/unity.cu | 264 | gemm binding + codec selection |
| validation/spark_..._cuda_validation.cu + run_laguna_layer7.sh + validate.sh | 425 | gates |
| source/spark_laguna_resident_decode_stage_internal.h | 146 | internal buffers |
| include/.../spark_laguna_resident_decode_stage_firmware.h | 136 | ABI constants |
| Makefile | 118 | build |
| source/cuda/config.h | 70 | model defines fold |
| include/.../spark_laguna_batch_tuning.h | 53 | bucket ladder via common |
| source/cuda/api.h + launch_shape.h | 45 | gemm decls, thread counts |
| model-families/laguna/name_map.json + tensor_patterns.json + model.h + kv_geometry.h | 763 | pack naming, patterns, constants (48 layers, GQA sliding/full), kv filler |

laguna total: 8,709 module + 763 family.

### muse — `modules/muse_glimmer_resident_decode_stage` + `model-families/muse_glimmer`

| file | LOC | purpose |
|---|---|---|
| source/spark_muse_glimmer_resident_decode_stage_module.c | 1682 | module incl. hand-rolled pack load |
| validation/..._cuda_validation.cu | 620 | validation gate |
| source/spark_muse_glimmer_resident_decode_stage_cuda.cu | 321 | flat attention/moe launch file |
| source/spark_muse_glimmer_serving_adapter.c | 293 | serving adapter over template |
| tools/muse_glimmer_pack_synthesize.c | 243 | synthetic pack builder |
| source/spark_muse_glimmer_stagepack_format.h | 222 | stagepack structs + tensor kinds |
| include/.../spark_muse_glimmer_resident_decode_stage_firmware.h | 183 | ABI constants |
| validation/validate_..._cuda.sh + publish_validator_wrapper.sh | 84 | gates |
| Makefile | 67 | build |
| model-families/muse_glimmer/tensor_patterns.json + work_control.h/.c + model.h + name_map.json | 395 | patterns, work control kv plan, constants, naming |

muse total: 3,610 module + 395 family.

### minimax-h3 — NO module on main; tree on `origin/lane/minimax-driver`

Branch finding first: `lane/minimax-driver-rebase6` DOES NOT EXIST on origin
(fetch fails: couldn't find remote ref). `lane/minimax-driver` exists and
carries the whole h3 tree. No other branch carries a second h3 tree
(checked `git ls-remote --heads` for minimax; only this one).

| file (at FETCH_HEAD) | LOC | purpose |
|---|---|---|
| modules/minimax_h3_resident_media_stage/source/..._cuda.cu | 591 | diffusion-transformer media stage kernels: adaLN modulation, attention with private block reductions, ffn; audio/video VAE gates live in validation |
| source/..._media_stage_module.c | 46 | revision/contract pins, stage block counts |
| source/spark_minimax_h3_stagepack_format.h | ~120 | media stagepack format |
| include/..._firmware.h + serving_adapter.h | ~230 | ABI + media job/receipt format |
| tools/minimax_h3_pack_synthesize.c | ~200 | synthetic media pack builder |
| validation/ (spark_minimax_h3_reference.c, _cuda_validation.cu, v3/v4/v5 gates, fixtures ~100 files) | ~4,500 incl fixtures | three-gate rig: v3 DiT block, v4 audio VAE, v4 video VAE |
| model-families/minimax_h3/.../spark_minimax_h3_model.h | 187 | constants: text tokenizer/encoder dims + media dims |
| model-families/minimax_h3/.../spark_minimax_h3_kv_geometry.h | 22 | encoder kv page math |
| model-families/minimax_h3/src/spark_minimax_h3_scheduler.c + header | 79 | flow-matching sigma scheduler (shift-sigma build, timesteps, euler step) |
| tools/minimax_h3_stagepack.py (477), gen_deployment.py (164), extract_tensors.py, npz_to_raw.py, stagepack_routing_test.py, tp16_boundary_check.py | ~1,300 | pack/deploy tooling |
| deploy/minimax_h3/ (manifest + 16 stage jsons) | — | TP16 media deployment |

h3 is a MULTIMODAL pipeline (text encoder + media diffusion stage), so its
driver is `resident_media_stage`, not `resident_decode_stage` — but it repeats
the fleet patterns: stagepack format + pack synthesize + firmware ABI header +
serving adapter header + gen_deployment tool + per-stage deploy jsons.

## 2. Copy-cluster map

Method: line diffs plus identifier-normalized diffs (family tokens folded,
literals masked) plus role-level function-clone matching (function names
compared after stripping the family token, bodies compared by token Jaccard).
Numbers below are measured on this baseline, not estimated.

### C1. Serving-adapter deployment-config + TP rail plumbing — the worst cluster

ling and laguna each hand-rolled a ~1,200-LOC serving adapter instead of using
`include/sparkpipe/spark_serving_adapter_template.h` (which k3, dsv4, gemma4,
muse all consume at 293-656 LOC). Role-clone measurements, ling vs laguna:

| role function | ling stmts | similarity |
|---|---|---|
| ServingValidateTpCollectiveMembers | 7 | 95% |
| ServingLoadSessionPorts | 18 | 91% |
| ServingLoadTpRailHosts | 22 | 91% |
| ServingLoadTpStepRails | 15 | 89% |
| ServingLoadTpAlgorithms | 17 | 91% |
| ServingDriverCompletion | 38 | 80% |
| ServingSnapshot | 25 | 85% |
| ServingProgress | 2 | 89% |
| PrepareAsyncCompletion | 28 | 89% |
| ServingValidateRowOrder | 24 | 84% |
| ServingLoadTpCollective | 14 | 27% (diverged) |

Whole-file: 68.6% raw / 32.3% normalized divergence (ling 1247 vs laguna 1225).
The remaining four families' adapters (dsv4 1415, k3 656, gemma4 307, muse 293)
share the control-path roles at 72-89% (CacheContext, ResetControl,
ResolvePrefetch, Snapshot, Destroy, Quiesce).
Wasted: roughly 1,900 LOC reducible to a common TP/deployment-config loader.

### C2. Pack validate/load/bind machinery

`model-families/common/include/sparkpipe/spark_pack_load_common.h` is a
token-pasted template implementing ValidateEntryPlacement, LoadEntry,
VerifyCoverage, BuildOrdinals, FillLinearView, and the Bind{Layer,Global,Mtp}
dispatch. Adoption: gemma4 YES (plus the qwen families, outside this lane).
Bypass: ling (PackValidateHeader/EntryGeometry/Ranges/AssignLayer/LoadEntry/
Expected{Layer,Global}Mask/ValidateInventory — ~450 LOC), laguna (same roles,
~350), muse (~200), k3 (private pack_load.c 257 + bind.c 121), dsv4 (own format
+ load inside the monolith). ling PackValidateRanges vs laguna: 78% similarity.
Wasted: ~1,200 LOC. This is duplication of a seam that ALREADY EXISTS — the
migration is mechanical.

### C3. stagepack_format.h quintuplets

ling 422, gemma4 482, laguna 346, muse 222, dsv4 407 (plus minimax's media
variant on the branch). All encode the same `.spstage` contract (257B header
with revision + digest slots, 64B entries, 256B align — DRY-1's finding on the
pack side). Pairwise normalized divergence ling~laguna 36.5%, ling~muse 68%,
gemma4~muse 53.7%. Differences are the tensor-kind enum range, scale-block
geometry, and per-kind slice modes. Wasted: ~1,300 LOC (one common
struct/validator + a per-family tensor-kind table of ~150 LOC).

### C4. Model-defines headers (the llm_defines.h candidate)

Three generations of the same idea coexist:

1. `model-families/<fam>/include/sparkpipe/spark_<fam>_model.h` — raw constants
   (k3 128, ling 147, gemma4 72+61+59 alias layer, laguna 158, muse 83, dsv4
   131+130 alias layer, minimax 187).
2. `modules/<fam>/source/cuda/config.h` — folds the family constants into
   short driver keys (ling 72 vs laguna 70: SAME key skeleton, 8.8% normalized
   divergence on shared keys; ling adds KDA keys, laguna adds sliding/GQA keys).
3. k3 folds via `inference/llms/kimi_k3/config.h` instead; dsv4 via
   pool_layout.h + firmware.h.

The gemma4 alias header (`SPARK_GEMMA4_MODEL_* -> SPARK_GEMMA4_MOE_*`) is the
exact mechanism a fleet-wide `spark_driver_defines.h` needs: driver code reads
only generic keys; each family supplies one constants header + one alias fold.

### C5. KDA / GDN / sliding state machinery — same shape, three implementations

- k3: hand-rolled KDA in cuda.cu/runner (state slots
  heads x key_dim x value_dim x 4B per layer, conv windows kernel=4, decay
  gates); slot math duplicated in `spark_k3_kv_geometry.h` AND inside the .cu.
- ling: KDA over shared `inference/kernels/linear_attn.cuh` (LmCausalConvKernel
  with LM_CONV_SWISH, LmDeltaRuleKernel, LmBoundedDecayKernel) + kda_state_pool
  with per-slot bytes pinned by static_assert.
- gemma4: "GDN" naming ALIASED onto sliding-window attention
  (`#define gdn_ordinal_by_layer sliding_ordinal_by_layer`); hybrid map is
  period/phase (`LAYER_IS_FULL = index%6==5`).
- laguna: sliding-window positions kernel (LmBuildSlidingWindowPositionsKernel)
  + GQA decode; same ordinal-by-layer hybrid map (`LAYER_IS_SLIDING`).
- dsv4 monolith also carries RunGdnLayer/RunGdnCoreDecode roles.

Shared shape: hybrid layer-map predicate, gdn/attn ordinal-by-layer arrays,
per-layer state/window slot bytes, state pool + index. The pack_load_common
template even parameterizes on `SPARK_PACK_LOAD_LAYER_IS_GDN` — the fleet
already standardized the PREDICATE name, not the machinery.
Parameter surface: layer map (predicate/period/phase), state element bytes,
conv kernel width, decay gate bounds, window tokens, heads/key/value dims.

### C6. Rope tables + sliding-window geometry

- laguna: `LagunaBuildYarnInvFrequency` host builder + H2D upload (dual rotary
  domains: full vs sliding, separate thetas + attention factor).
- gemma4: LaunchSlidingRope/LaunchFullRope over shared LmRopePerHeadKernel
  (theta 10000 sliding vs full table + base 1e6 + qk scale).
- ling: latent-attention rope sections (64-dim rope per head, nope+rope layout
  static_asserts).
- k3: rope dim 64 via kimi config; minimax media: cos/sin tables (v3 13x96,
  v4 video 33x48 fixtures).
Same shape every time: build inv_freq (or cos/sin) table of dimension/2,
upload, launch shared kernel with per-domain theta. Candidate common module:
`spark_rope_plan.h` (~9 keys: mode theta|yarn|table, rotary dims per attention
class, thetas, yarn factor, table elements, qk scale).

### C7. firmware.h ABI-constant headers

k3-in-runner.h 133, ling 141, gemma4 224, laguna 136, muse 183, dsv4 264:
each declares the SAME view ABI set (node context, frame context, prefill
frame view, kv block table, linear view, decode batch view versions) + limits
+ module id/target strings. ling~laguna 22.3% normalized; ling~muse 67.2%.
Candidate: one `spark_resident_decode_stage_firmware_common.h` with ~10 keys
per family.

### C8. Batch-tuning bucket ladder

`spark_batch_variant_tuning_common.h` EXISTS and ling/laguna delegate to it
(57/53 LOC shims). k3 (152) and dsv4 (172) hand-roll the identical
b1..b1024 bucket ladder + module-id strings. Wasted: ~250 LOC; migration is
converting k3/dsv4 to the shim shape (6 keys: prefix, suffix, codec name,
top-k, expert count, per-bucket tiles).

### C9. pack_synthesize.c per module

ling 297 vs laguna 325: 20.6% normalized divergence — near-copy. gemma4 244,
muse 243, minimax ~200. These build synthetic validation packs; DRY-1
explicitly excluded the in-module C synthesizers from its packer corpus, so
this duplication is MISSED by DRY-1. Wasted: ~700 LOC toward a parameterized
synthesizer (family tensor-kind enum + dims + entry list, ~10 keys).

### C10. Validation rigs

validate_*.sh: 4 near-copies (53-71 LOC; ling~laguna 37.7%).
spark_*_cuda_validation.cu: ling 2847, gemma4 1904, muse 620, laguna 193(+ the
distinct 1,816-LOC layer7 realpack rig), dsv4 1604 — same gate structure
(fixture load, per-stage kernel compare, digest print) re-expressed per family.
Candidate: one common harness + per-family stage tables (~8 keys). Wasted:
~2,000+ LOC, the largest single cluster by mass, but lowest risk-adjusted
value (validation drift is annoying, not dangerous — the dangerous clusters
are C1/C2/C3).

### C11. Makefiles

`modules/resident_decode_stage_rules.mk` EXISTS, yet ling/laguna carry two
near-identical 118-line Makefiles (29.7% normalized divergence), gemma4 65+68,
muse 67, dsv4 93+95, k3 none (gate scripts instead). Variable part is ~7 keys:
MODULE_FAMILY/IDENTIFIER/TARGET, MODEL_HEADER, contract SHA, source lists,
variant defines. Wasted: ~250 LOC.

### C12. gen_deployment tools (DRY-1-adjacent, host side)

ling_gen_deployment.py 182, laguna 212, muse 121 (~50% pairwise normalized
divergence), k3 uses .sh generators, minimax branch adds a 164-LOC one. Each
emits deployment_manifest.json + stage.N.json + model_resident.json. Wasted:
~400 LOC to one generator with ~12 keys (ranks, TP degrees, ports, pack paths,
stage layer counts).

### C13. kv_geometry.h capacity fillers

k3 48, ling 46, laguna 44 (+ minimax encoder-kv variant 22): all fill
`SparkKvCacheCapacityRequest` from family constants. ling's is the most
complete (abi_version + fp8 scale block; k3's omits abi_version — a latent
contract drift already visible). Candidate: one filler + ~12 geometry keys.

### Non-clusters (genuinely unique, do NOT force into common)

- dsv4 module.c monolith content: MLA + PLE + MTP chain + HC readout + graph
  capture (the fleet's ONLY cudaGraph use) — unique per-model control flow.
- k3 runner.cu TP4 island orchestration with weightd lazy acquire — unique
  (PP+TP hybrid over weightd leases).
- laguna layer7 realpack rig — the fleet's only numeric equivalence harness
  vs a python reference; worth KEEPING unique, maybe generalizing later.
- minimax media diffusion stage (adaLN, VAE gates, sigma scheduler) — different
  math domain; only its pack/firmware/deploy wrappers duplicate the fleet.

## 3. Parameter surfaces per proposed common module

The operator's target shape: ONE driver include-path file holding ALL values
per family (llm_defines.h style). gemma4's alias-header pattern is the working
in-repo precedent for two variants (31b/26b) sharing one driver tree.

Common module 1: `spark_driver_defines.h` (merge of model.h + cuda/config.h +
firmware module-id strings) — keys:
identity: MODEL_ID, MODULE_ID, MODULE_TARGET, DRIVER_MODEL_ID, DRIVER_REVISION,
CONTRACT_SHA256 (6)
topology: HIDDEN_DIMENSION, LAYER_COUNT, VOCAB_COUNT, OUTPUT_VOCAB_COUNT,
MAXIMUM_CONTEXT_TOKENS, FIRST_ROUTED_LAYER, WEIGHT_LAYER_COUNT (7)
attention core: ATTENTION_KIND (mla|gqa|latent_linear), HEAD_DIMENSION,
Q_HEADS, KV_HEADS, ATTENTION_SCALE, WINDOW_TOKENS, FULL_LAYER_PERIOD,
FULL_LAYER_PHASE, LAYER_MAP_PREDICATE (9)
MLA block (when mla|latent_linear): LATENT_DIMENSION, QK_ROPE_HEAD_DIMENSION,
QK_NOPE_HEAD_DIMENSION, VALUE_HEAD_DIMENSION, ROPE_THETA (5)
KDA block (when latent_linear): KDA_HEADS, KDA_KEY_DIM, KDA_VALUE_DIM,
KDA_QK_DIM, KDA_CONV_KERNEL, KDA_GATE_LOWER_BOUND, KDA_STATE_ELEMENT_BYTES,
KDA_CONV_WINDOW_BYTES_PER_LAYER (8)
rope plan: ROPE_MODE (theta|yarn|table), ROPE_FULL_ROTARY_DIMENSION,
ROPE_SLIDING_ROTARY_DIMENSION, ROPE_FULL_THETA, ROPE_SLIDING_THETA,
ROPE_ATTENTION_FACTOR, ROPE_TABLE_ELEMENTS, QK_SCALE (8)
moe: ROUTED_EXPERT_COUNT, EXPERTS_PER_TOKEN, EXPERT_INTERMEDIATE_DIMENSION,
DENSE_INTERMEDIATE_DIMENSION, ROUTED_SCALE, W1_COMPONENTS, ROUTER_GROUPS,
ROUTER_TOP_GROUPS, TOP_K (9)
norm/precision: RMS_NORM_EPSILON, FINAL_LOGIT_SOFTCAP, EMBED_SCALE,
FP8_SCALE_BLOCK, KV_BITS, BF16_ELEMENT_BYTES (6)
kv: KV_PAGE_SLOTS, KV_SLOT_BYTES, KV_BLOCK_TOKENS (3)
special tokens: BOS/EOS/EOS_ALT/EOS_ALT2/PAD (5)
TOTAL ~65 keys, of which ~40 shared and ~25 attention-family-conditional.

Common module 2: pack load/bind (extend existing spark_pack_load_common.h
adoption) — keys: family token, LAYER_MAP_PREDICATE, SEEN type/width/format,
TENSOR_KIND range, STAGEPACK_GLOBAL/MTP_LAYER, BYTES_MATCH geometry,
EXPECTED_GEOMETRY, REGION_HOOK, MODULE_TAG (~12).

Common module 3: stagepack format common — keys: MAGIC, REVISION, header
bytes/entry bytes/align (fixed by contract), tensor-kind enum table,
scale-block geometry, per-kind slice mode (~10).

Common module 4: serving TP/deployment config (fold ling/laguna hand-rolls
into spark_serving_adapter_template.h) — keys: TP_DEGREE, rail/step-rail host
lists, algorithm names, session ports, hidden-transport members, collective
peers, quiesce/reset policy flags (~22).

Common module 5: cuda cell glue (config.h/api.h/launch_shape.h/unity.cu) —
keys: LAYER_THREADS, ATTN_THREADS, UNITY_TILE_N/K, UNITY_STAGES, UNITY_WARPS,
EXPERT_WEIGHT_CODEC, GEMM grouped flag (~9; ling/laguna currently identical
values for most).

Common module 6: hybrid state (KDA/GDN/sliding) — keys: from defines block +
STATE_POOL_STRATEGY, WINDOW_POSITIONS_MODE, CONV_ACTIVATION (swish fixed for
ling/k3) (~6 beyond defines).

Common module 7: firmware ABI common — keys: 6 ABI versions, limits
(MAX_ACTIVE_SEQUENCES, MAX_ROWS), tag strings (~10).

Common module 8: batch tuning shim for k3/dsv4 — keys: MODULE_ID_PREFIX/SUFFIX,
EXPERT_CODEC_NAME, TOP_K, EXPERT_COUNT (~5 + common's per-bucket table).

Common module 9: pack synthesizer — keys: tensor-kind table, dims, entry
recipes, fixture paths (~10).

Common module 10: validation harness — keys: fixture dir, stage list,
tolerances, digest expectations (~8).

Common module 11: Makefile via rules.mk — keys: 7 listed in C11.

Common module 12: flow scheduler (from minimax, generic even at one user) —
keys: SHIFT, SIGMA_POINT_COUNT, step rule selector (4).

Common module 13: kv geometry filler — the ~12 geometry keys from defines +
abi_version pin (fix k3's omission).

## 4. The already-common: use vs bypass

| seam | k3 | dsv5 | ling | gemma4 | laguna | muse | minimax (branch) |
|---|---|---|---|---|---|---|---|
| spark_serving_adapter_template.h | USE | USE | BYPASS | USE | BYPASS | USE | adapter header only |
| spark_pack_load_common.h | BYPASS | BYPASS | BYPASS | USE | BYPASS | BYPASS | n/a (media format) |
| spark_batch_variant_tuning_common.h | BYPASS | BYPASS | USE | n/a (no moe buckets) | USE | n/a | n/a |
| spark_stage_module_common (lifecycle/ledger/paged cache) | BYPASS | USE | USE | USE | USE | USE | n/a |
| tp_device_collective | USE | USE | USE | USE | USE | USE | boundary-check tool only |
| inference/kernels/* + runtime/gemm.cuh | BYPASS (llms/kimi_k3) | USE | USE | USE | USE | USE | partial (spark_lm_kernels only) |
| spark_lm_kernels.cuh (model-families/common) | BYPASS | USE | USE | USE | USE | USE | USE |
| weightd client seam | USE (lazy leases) | USE | BYPASS | BYPASS | USE | BYPASS | n/a |
| resident_decode_stage_rules.mk | BYPASS | partial | BYPASS (own Makefile) | partial | BYPASS (own Makefile) | partial | n/a |
| model_resident_client | n/a (runtime-side) | n/a | n/a | n/a | n/a | n/a | n/a |

Reading: ling and laguna are the youngest modules and they BYPASS the serving
template (the two oldest hand-rolls got copied forward instead of the seam);
everything except gemma4 bypasses pack_load_common; k3 is the systematic
bypasser (5 of 8 seams) because it was built on the older
inference/llms/<model> pattern.

## 5. Unique math worth extracting (generic even at one user)

1. minimax flow-matching sigma scheduler (`spark_minimax_h3_scheduler.c`):
   shift-sigma curve, sigma->timesteps, euler step. Clean, dependency-free,
   branch-only today. Extract to model-families/common as `spark_flow_scheduler.h`.
2. laguna YaRN inv-frequency builder (`LagunaBuildYarnInvFrequency`): dual-domain
   rotary table math; becomes the rope-plan's yarn mode.
3. KDA slot sizing algebra (state bytes = heads*key*value*element_bytes;
   conv window = qk_dim*components*kernel*element_bytes): currently asserted
   independently in k3 kv_geometry, ling config.h static_asserts, ling model.h
   macros — one inline header would end the triple-maintenance.
4. laguna `RowsPerExpert` (ceil(tokens*top_k/experts)) — generic MoE row
   estimator, one-liner, belongs next to the MoE keys.
5. dsv4 `spark_dsv4_cache_plan.c` (1,037 LOC): page-tier/arena planning math;
   single user today but it is the only planner in the fleet — candidate for
   model-families/common if a second KV-tier family appears.
6. minimax media block reductions (warp-shuffle max/sum in the media .cu):
   DELETE against shared reduction helpers in inference/kernels rather than
   extract — a bypass, not unique math.

## 6. Top-10 copy clusters by wasted LOC

| # | cluster | members | wasted LOC (approx) |
|---|---|---|---|
| 1 | validation harness re-writes | ling 2847, gemma4 1904, dsv4 1604, muse 620 | ~2,500 |
| 2 | serving-adapter hand-rolls | ling 1247, laguna 1225 vs template ~300 | ~1,900 |
| 3 | pack validate/load/bind bypass | ling, laguna, muse, k3, dsv4 | ~1,200 |
| 4 | stagepack_format.h quintuplets | 1,879 total | ~1,300 |
| 5 | cuda cell glue (layer/unity/config/api/launch_shape) | ling 1,592, laguna 1,313 | ~1,200 |
| 6 | module.c pack/lifecycle boilerplate in monoliths | dsv4 module.c non-compute share | ~800 |
| 7 | pack_synthesize.c near-copies | 1,109 total + minimax | ~700 |
| 8 | gen_deployment generators | 679 total + k3 sh + minimax | ~400 |
| 9 | batch_tuning hand-rolls | k3 152, dsv4 172 | ~250 |
| 10 | Makefiles beside rules.mk | 436 total | ~250 |

Total identified waste: ~10,500 LOC across my seven families, against roughly
15,700 LOC of genuinely family-specific kernels/math that must stay.

## 7. Surprises

1. `lane/minimax-driver-rebase6` does not exist on origin. Only
   `lane/minimax-driver` carries h3 (model-families + a resident MEDIA stage
   with a three-gate validation rig and TP16 deploy). If a rebase6 lane was
   expected, it was never pushed or was deleted.
2. The common seams ALREADY EXIST for the worst duplication
   (serving template, pack_load_common, batch tuning common, rules.mk).
   ling/laguna bypassing the serving template is the single most expensive
   avoidable decision in the fleet; gemma4 is the model citizen.
3. gemma4 already solves the operator's two-variant problem (31b/26b: one
   driver tree, alias header, two Makefiles, two contract SHAs). Its alias
   header is the template for the fleet-wide llm_defines.h migration.
4. k3 depends on `inference/llms/kimi_k3/` — the OLD per-model include tree —
   while every younger family consumes `inference/kernels/*`. k3 is a
   different generation, not a copy-paste island of the same one; its survey
   numbers (low role overlap) reflect that.
5. ling vs laguna api.h/launch_shape.h/unity.cu are literally
   rename-only copies (values identical), i.e. someone cloned the ling cell
   and renamed — yet their serving adapters, supposedly the same authorship
   era, diverged 32%: the hand-rolls drift, the renamed glue did not.
   Copying is not the sin; hand-rolling near-identical config plumbing is.
6. minimax hand-rolls warp-shuffle block reductions that exist in the shared
   kernel tree — a fresh bypass being born on an active branch, worth catching
   before the merge.
7. k3's kv_geometry filler omits abi_version/descriptor_bytes that ling's sets
   — exactly the contract-drift class the common filler would kill.
