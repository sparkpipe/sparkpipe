# DESIGN — gemma-4 family driver (one family, two contracts)

Lane: gemma4 · branch `lane/gemma4-driver` @ origin/main 8f3a6f2 · UNTRACKED deliverable.
Author: designer stage, 2026-09-09. HF facts refetched today from
`https://huggingface.co/google/{gemma-4-31B-it,gemma-4-26B-A4B-it}` (config.json +
model.safetensors.index.json) and transformers `main` (`modeling_gemma4.py`,
`configuration_gemma4.py`, `modeling_rope_utils.py`). Semantics below cite those
sources; nothing is guessed.

**Problem in one sentence:** bring google/gemma-4-31B-it (dense) and
google/gemma-4-26B-A4B-it (MoE) onto the resident-decode driver stack as ONE
family module with TWO contracts, consuming only shared kernels that already
exist or are already approved to land on sibling lanes.

**The reframe:** gemma-4 is the first pure-GQA family in the repo, and the
engine has been waiting for one: the frozen `inference/kernels/gqa.cuh` triple
(sliding-window positions, KV store, GQA decode) plus the mimo/qwen llms layers
already exercise exactly this shape. The design is therefore ~90% deletion from
the qwen4_flash donor module (GDN, chunk, hyper-connections, indexer, PLE,
head-screening all go) plus rewiring to `gqa.cuh`. The two models differ only in
numbers and in one compile-time FFN block; the dsv4 flash/pro alias-header
mechanism covers that with zero new machinery. No new shared kernel is required.

---

## 1. Per-model geometry (HF-verified 2026-09-09)

Both repos: `Gemma4ForConditionalGeneration` (multimodal wrapper; text stack =
`model.language_model.*`, vision tower out of scope by contract — same scope
note pattern as qwen4_flash). Text `model_type: gemma4_text`. `dtype: bfloat16`.

| Field | gemma-4-31B-it | gemma-4-26B-A4B-it | Source |
|---|---|---|---|
| model_id | `google/gemma-4-31B-it` | `google/gemma-4-26B-A4B-it` | config.json |
| hidden_size | 5376 | 2816 | text_config |
| num_hidden_layers | 60 | 30 | text_config |
| vocab_size | 262144 | 262144 | text_config |
| layer_types | 50 sliding : 10 full | 25 sliding : 5 full | `layer_types` list; pattern `5 sliding : 1 full`, last layer full |
| sliding window | 1024 | 1024 | `sliding_window` |
| q heads (sliding) | 32 × head_dim 256 | 16 × 256 | `num_attention_heads`, `head_dim` |
| kv heads (sliding) | 16 | 8 | `num_key_value_heads` |
| q heads (full) | 32 × head_dim **512** | 16 × **512** | `global_head_dim` via `per_layer_config` override |
| kv heads (full) | **4** | **2** | `num_global_key_value_heads` (gated by `attention_k_eq_v: true`) |
| attention_k_eq_v | true (full layers have **no v_proj tensor** — index shows v_proj ×50/×60 and ×25/×30) | true | config + index tensor census |
| MLP (dense branch) | intermediate 21504, `gelu_pytorch_tanh`, every layer | intermediate 2112, every layer (`mlp.*` ×30) | config + index |
| MoE | none (`enable_moe_block: false`, `num_experts: null`) | 128 experts, top-8, `moe_intermediate_size` 704, **parallel with dense MLP in all 30 layers** | config + index (`experts.gate_up_proj`/`down_proj` 3D, ×30) |
| router | — | `router.norm` (RMS, no weight) → `× router.scale × H**-0.5` → `proj` H×128 → softmax fp32 → top-8 → renormalise sum-to-1 → `× per_expert_scale[selected]` | `Gemma4TextRouter.forward` |
| norms per layer | input, post_attention, pre_feedforward, post_feedforward (4) | those 4 + pre_feedforward_2, post_feedforward_1, post_feedforward_2 (7) | index census |
| norm semantics | **plain weighted RMSNorm** (`normed * weight`; NOT gemma3's `(1+w)`), eps 1e-6; q_norm/k_norm per-head **with** weight, v_norm per-head **no weight** (`with_scale=False`, no tensor in index) | same | `Gemma4RMSNorm` |
| per-layer `layer_scalar` | ×60, checkpoint all-ones | ×30, all-ones | index + `DecoderLayer.forward` (`hidden_states *= self.layer_scalar`) |
| per-layer-input (PLE) | **dead**: `hidden_size_per_layer_input: 0`, no tensors | dead, same | config + `if self.hidden_size_per_layer_input:` |
| kv sharing | dead: `num_kv_shared_layers: 0` — every layer stores its own KV | dead, same | config + `Gemma4TextAttention.__init__` |
| rope (sliding) | default, theta 1e4, all 128 pairs of 256 dims | same | `rope_parameters.sliding_attention` |
| rope (full) | **proportional**: `partial_rotary_factor` 0.25 → `rope_angles = int(0.25·512/2) = 64` rotated pairs; inv_freq[i] = `1e6 ** (-i/256)` for i < **64**, then **0** for i in [64,256); angle = pos·inv_freq → upper 3/4 of each 512-dim head is identity-rotation (ANCHORS finding 1: publisher wins, design's earlier 128-pairs claim was wrong) | same (dims identical) | `_compute_proportional_rope_parameters` (rope_utils.py:246-263) |
| attention scaling | **1.0** (`self.scaling = 1.0`; qk-norm makes q RMS 1; `attention_scaling` 1.0 from rope init) | same | `Gemma4TextAttention.__init__` |
| final_logit_softcapping | 30.0 (logits only: ÷30, tanh, ×30; monotone) | 30.0 | config + `Gemma4ForCausalLM.forward` |
| attention softcap | none (text; softcap exists only in the audio stack) | none | `eager_attention_forward` default |
| tie_word_embeddings | true (no `lm_head` tensor in index; head = `embed_tokens`) | true | config + index |
| embed scale | bf16(sqrt(5376)) = **73.5** | bf16(sqrt(2816)) = **53.0** | `Gemma4TextScaledWordEmbedding` (fp32 buffer cast to weight dtype at multiply — the bf16 rounding is load-bearing, PR 29402 note) |
| bos / eos / pad | 2 / [1, 106] / 0 | same | config |
| max_position_embeddings | 262144 | 262144 | config |
| text params (derived, contract-recorded) | ≈30.15B (embed 1.409B + 60×0.4789B) | ≈25.16B total, ≈3.75B active/token (attn 34.6M + dense 17.8M + 8×3×704×2816=47.5M per layer) | arithmetic from config |
| weights on warm | NOT YET (`/mnt/model-warm/gemma-4-31b-it` downloading) | NOT YET (`/mnt/model-warm/gemma-4-26b-a4b-it`) | manager brief |

Exact forward (decode, one layer), from `Gemma4TextDecoderLayer.forward`:

```
residual = h
h  = RMS(h, input_layernorm)
h  = attn(h)                        # see below
h  = residual + RMS(h, post_attention_layernorm)
residual = h
h  = MLP( RMS(h, pre_feedforward_layernorm) )                 # dense branch
# 26B only:
b1 = RMS(mlp_out, post_feedforward_layernorm_1)
r  = router(residual)                # router reads the RESIDUAL, not the normed input
b2 = RMS→experts→RMS (pre_ffn_2 / post_ffn_2) of residual
h  = b1 + b2
h  = residual + RMS(h, post_feedforward_layernorm)            # both arms
h *= layer_scalar                  # pack-asserted all-ones → identity
```

Attention (per layer kind), from `Gemma4TextAttention.forward`:

```
q = q_proj(h);  q = q_norm(q);  q = rope(q)                    # per-head norm, then rope
if sliding:
    kv = kv_proj_fused(h)            # pack fuses k_proj|v_proj rows: 16|16 blocks of 256 (31B)
    k = k_norm(kv.k);  k = rope(k, theta 1e4)
    v = v_norm(kv.v)                 # NO weight
else:                                # full, k_eq_v
    kraw = k_proj(h)                 # 4×512 (31B) / 2×512 (26B); no v weight exists
    v = v_norm(kraw)                 # raw k, no rope, no weight
    k = k_norm(kraw); k = rope(k, inv_freq_table)
store(k, v)                          # post-norm, post-rope values are what the cache holds
out = softmax(q·kᵀ · 1.0) v         # window = last 1024 positions (sliding) or full context
attn = o_proj(out) + TP all-reduce
```

## 2. Architecture mapping — every op → existing kernel or new code

No gemma kernel goes into `inference/kernels/` in v1. Family-local CUDA lives in
the module's `spark_gemma4_resident_decode_stage_cuda.cu` (namespace
`SparkGemma4*`), per the per-family rule; promotion to shared is a flagged
follow-up if a second family needs it.

| Op | Kernel / mechanism | Location | Status |
|---|---|---|---|
| Layer + final RMS norms (weighted, plain) | `LmFusedResidualRmsNormKernel` (residual+norm fused; final norm = residual_bf16=0 form) | inference/kernels/norm.cuh:52 | exists on main |
| q_norm / k_norm (per-head, weighted); v_norm (per-head, no weight); router norm (whole-row, no weight = head_count 1) | `LmHeadRmsNormKernel` (per-head RMS, `weight_or_null`, fp32 multiply epilogue) | inference/kernels/norm.cuh (muse lane) | **approved landing, lane/muse-driver — consume by frozen contract** |
| All projections (q, fused-kv, o, dense gate_up/down, router proj, head) | `LmGemmLaunch` bf16 path (mimo_2_5 idiom) | inference/kernels/gemm.cuh | exists |
| rope (sliding layers + q/k of full layers) | `LmRopePerHeadKernel` theta 1e4, rope_dim 256 | inference/kernels/attn.cuh:50 | exists |
| rope (full layers, proportional) | same kernel with the approved trailing params `(inv_freq_table, attention_scale=1.0f, rope_offset)`; table = 256 fp32 entries `1e6**(-i/256)` i<128 else 0, pack-emitted global | attn.cuh:50 + laguna signature | **approved landing, lane/laguna-driver — converge on exactly this signature** |
| KV store | `LmGqaKvStoreKernel<Geometry,THREADS,KV_HEADS,256,256 / 1×512,512>` per layer kind | inference/kernels/gqa.cuh:56 | exists, frozen |
| sliding window selection (1024) | `LmBuildSlidingWindowPositionsKernel` | gqa.cuh:13 | exists, frozen |
| attention decode | `LmGqaAttentionDecodeKernel`, runtime `heads` (local), `qk_scale = 1.0f` | gqa.cuh:77 | exists, frozen |
| dense FFN activation | gelu_pytorch_tanh(gate)·up | **family-local `SparkGemma4GatedGeluKernel`** (~20 lines, module cuda file) | new, family-local (silu variant `LmSiluMulKernel` norm.cuh:129 stays untouched; promote if a 2nd gelu family lands) |
| router: softmax over 128 | `LmHeadSoftmaxKernel` rows=tokens, width=128, temperature 1.0 | inference/kernels/head.cuh:100 | exists (generic fp32 row softmax; naming is historical) |
| router: top-8 + renormalise | `LmTopkSmallKernel<THREADS,8,true,1,1,LM_TOPK_SCORE_IDENTITY>` mixture_scale=1.0 on fp32 probs — renormalised top-k of softmax probs ≡ HF's `topk(softmax)/sum` exactly | inference/kernels/topk.cuh:43 | exists |
| router scale fold | `router.scale × H**-0.5` folded into `router.proj` columns at pack time (exact linear re-association; HF rounds after the multiply, we round after the GEMM — within tolerance, stated) | packer | pack-time |
| per-expert scale | `per_expert_scale[e]` folded into expert `down_proj[e]` rows at pack time (scalar × output row); `route_weight` stays pure renormalised prob | packer | pack-time |
| MoE dispatch/grouped GEMM/finalize | `LmRouteBuild` + grouped `LmGemmLaunch` (w1 gate_up fused, w2 down) + `LmMoeFinalizeKernel` (fp32 route_weight × bf16 expert out) | inference/kernels/route.cuh, norm.cuh:295; mimo_2_5/layer.cuh:163-217 idiom | exists |
| tied-embed argmax head | `LmHeadCandidateKernel`/`LmHeadCommitKernel` over embed shard + qwen4_flash `TpCombineU64Max` combine pattern | inference/kernels/head.cuh; module donor | exists |
| embedding gather + scale | donor `SparkQwen4FlashEmbeddingGatherShardedKernel` + one scale multiply → **family-local `SparkGemma4EmbeddingGatherShardedScaledKernel`** | module cuda file | new, 2-line delta on donor kernel |
| softcap 30 | **not implemented** — ÷30, tanh, ×30 is strictly monotone → greedy argmax invariant (operator ruling). Logits/probabilities deferred; contract records the deviation | — | ruled out of v1 |
| `layer_scalar` | identity; packer asserts every buffer is exactly 1.0 and fails closed otherwise | packer | pack-time |

**Consumption reference for the whole decode idiom:**
`inference/llms/mimo_2_5/layer.cuh:106-160` (GQA store/window/decode with runtime
qk_scale) and `:163-249` (MoE + dense MLP). The gemma module's attention and FFN
sections are that idiom with the geometry/activations above.

## 3. Two-contract family design (dsv4 pattern)

One module, one entry prefix `SparkGemma4ResidentDecodeStage`, one source tree;
two build arms selected by a define, exactly like dsv4 flash/pro:

- `model-families/gemma4/include/sparkpipe/spark_gemma4_model.h` — the generic
  `SPARK_GEMMA4_MODEL_*` namespace. Top of file:
  `#if defined(SPARK_GEMMA4_MOE_BUILD)` → `#include "sparkpipe/spark_gemma4_moe_model_aliases.h"`
  `#else` → dense defines. (dsv4 precedent: `spark_dsv4_model.h` + `spark_dsv4_pro_model_aliases.h`.)
- `spark_gemma4_moe_model.h` — 26B defines; alias header maps
  `SPARK_GEMMA4_MODEL_*` → `SPARK_GEMMA4_MOE_*`.
- MoE-only tensors and code paths sit behind `#if SPARK_GEMMA4_MODEL_MOE_BLOCK`
  (compile-time; no runtime codec/arch branch — module hard boundary 3). The
  dense arm never compiles router/expert code; the code-size ratchet sees it.

| Aspect | 31B contract | 26B-A4B contract |
|---|---|---|
| build flag | (none) | `-DSPARK_GEMMA4_MOE_BUILD -DSPARK_GEMMA4_MODEL_MOE_BLOCK=1` |
| module identifier | `spark.gemma4.31b.resident_decode_stage.bf16.linear_bf16.kv_bf16.h5376.l60.v1` | `spark.gemma4.26b-a4b.resident_decode_stage.bf16.linear_bf16.kv_bf16.h2816.l30.e128k8.v1` |
| contract JSON | `model_contracts/gemma4_31b_authoritative.json` | `model_contracts/gemma4_26b_a4b_authoritative.json` |
| header binding test | one `tests/test_gemma4_model_header.py` with a per-arm BINDINGS table (qwen4_flash pattern), parameterised over the two contracts | same test, second table |
| stagepack magic | `0x50534734` ("4GSP" family tag), FORMAT_VERSION 1; enum covers BOTH arms — MoE kinds simply absent from 31b packs and the pack loader rejects them there (fail closed) | same enum, MoE kinds present |
| layer dispatch | table-driven: layer index `% 6 == 5` → full (generated from `LAYER_COUNT`), sliding otherwise; 60 vs 30 is data | same |
| firmware description | `examples/model_descriptions/gemma4_31b_resident_decode_stage_firmware.json` | `.../gemma4_26b_a4b_..._firmware.json` |
| serving adapter ids | `spark.gemma4.serving-adapter.tp4pp4.v1`, model_id `google/gemma-4-31B-it`, TP degree 4 | `spark.gemma4.serving-adapter.tp4pp2.v1`, model_id `google/gemma-4-26B-A4B-it`, TP degree 4, PP 2 |
| serving model revision pin | `GEMMA4_MODEL_REVISION` define (empty until warm download pins the snapshot; adapter `#error`s if unset at build — same provenance pin as qwen4_flash Makefile:19-30) | same |

Shared by both arms (identical code): attention path, norm sandwich, dense MLP,
head/embedding, KV machinery, adapter template wiring (`session_ports` parse,
tree topology, credit 8), stagepack loader, pack synthesize, validation harness.

## 4. Rope / attention convergence verdict (frozen gqa.cuh + laguna yarn)

**Verdict: full convergence, zero kernel extensions to frozen files.**

- `LmGqaKvStoreKernel` (gqa.cuh:56) and `LmGqaAttentionDecodeKernel` (gqa.cuh:77)
  fit gemma as-is: `static_assert` slot layouts hold for all four
  (model × layer-kind) geometries (§6 table); runtime `heads` + `qk_scale` cover
  the 32/16, 32/4, 16/8, 16/2(→1 local) head splits; `heads % KV_HEADS == 0`
  holds on every rank; qk_scale is the literal `1.0f` (HF `scaling = 1.0`).
  **No flagged request against gqa.cuh.**
- Sliding window is `LmBuildSlidingWindowPositionsKernel` with `window = 1024`
  — the [available−1024, available) selection matches HF's local-attention
  convention including self-position.
- **Rope: gemma CONVERGES on the laguna signature.** Sliding layers use the
  existing 6-arg call (theta 1e4). Full layers need the three trailing
  defaulted params exactly as frozen by lane/laguna-driver
  (`inv_freq_table` = 256-entry fp32 global with 128 proportional freqs + 128
  zeros; `attention_scale` = 1.0f; `rope_offset` default). Proportional rope is
  therefore a DATA table, not code. If laguna's landing drifts, gemma's full-
  layer rope is the only call site to rebase (one line).
- **Norm: gemma consumes muse's `LmHeadRmsNormKernel` and needs the
  `weight_or_null` half of that contract.** Important correction to manager
  memory: gemma-4's norms are PLAIN weighted RMS (transformers `Gemma4RMSNorm`
  has no `(1+w)`), so gemma does NOT need `LmCenteredRmsNormKernel` — the
  existing `LmFusedResidualRmsNormKernel` already matches gemma's
  residual-norm semantics. Gemma's dependency on muse is per-head q/k/v norm
  only (head_dim 256/512 rows, weight and no-weight variants), which is exactly
  the approved `LmHeadRmsNormKernel`.
- No yarn math, no attention-scale multiply, no softcap kernel, no
  kv-shared-layer path, no PLE: all dead branches in this checkpoint family
  (§1) are simply absent.

## 5. File plan + port_family.py substitution sketch

Route: `python3 /Users/mac/batch-ling/tools/dev/port_family.py` (argparse
rename porter, longest-first substitution, prints per-file counts; silent
zero-count ports cannot pass). Donor = `modules/qwen4_flash_resident_decode_stage`
(closest architecture: GQA attention + 512-expert MoE + TP + sharded embedding +
synthesizable stagepack). Deletions are a separate surgical pass after the port
(ling precedent: cut by function, grep-verify clean), because a rename porter
cannot delete.

```
model-families/gemma4/
  include/sparkpipe/spark_gemma4_model.h            (new; dense defines + MOE alias hook)
  include/sparkpipe/spark_gemma4_moe_model.h        (new)
  include/sparkpipe/spark_gemma4_moe_model_aliases.h(new)
  name_map.json                                     (new; layer_classes: sliding_layers[], full_layers[])
  tensor_patterns.json                              (new; §1 tensor census)
modules/gemma4_resident_decode_stage/
  Makefile                                          (31B arm; qwen4_flash Makefile skeleton)
  Makefile.moe                                      (26B arm; dsv4 Makefile.pro pattern)
  include/sparkpipe/spark_gemma4_resident_decode_stage_firmware.h
  include/sparkpipe/spark_gemma4_serving_adapter.h
  source/spark_gemma4_resident_decode_stage_module.c
  source/spark_gemma4_resident_decode_stage_cuda.cu
  source/spark_gemma4_serving_adapter.c
  source/spark_gemma4_stagepack_format.h
  source/spark_gemma4_internal.h
  tools/gemma4_pack_synthesize.c
  validation/spark_gemma4_reference.c               (independent oracle, §7)
  validation/spark_gemma4_resident_decode_stage_cuda_validation.cu
  validation/validate_gemma4_resident_decode_stage_cuda.sh
examples/model_descriptions/gemma4_31b_resident_decode_stage_firmware.json
examples/model_descriptions/gemma4_26b_a4b_resident_decode_stage_firmware.json
tests/test_gemma4_model_header.py
model_contracts/gemma4_31b_authoritative.json
model_contracts/gemma4_26b_a4b_authoritative.json  (+ references/gemma4/ pin: config.json ×2, modeling_gemma4.py, modeling_rope_utils.py, index ×2)
```

Substitution map (core entries): `qwen4_flash`→`gemma4`,
`Qwen4Flash`→`Gemma4`, `QWEN4_FLASH`→`GEMMA4`, `Qwen38`→`Gemma4` (adapter macro
layer), plus identifier strings per §3.

**DELETED from donor** (grep-verified clean before any build):
`SparkQwen4FlashConvUpdate/DecayBeta/GdnStep/GatedNorm/Chunk*` kernels and GDN
state pool (cuda.cu ~lines 23-186, 718-1216), `Hc*` hyper-connection kernels
(1449-1560), `Indexer*` DSA kernels (1562-1825), `Ple*` PLE kernels (1825+),
`HeadScreenedArgmax`/`HeadShadowQuantize` speculative screening (keep plain
`HeadArgmax` path via `LmHeadCandidate/Commit`), MTP stage-pack kinds and
`MTP_LAYER_COUNT` plumbing (gemma has no MTP), GDN/HC/indexer/PLE stagepack
tensor kinds and their shape-table entries, `qwen4_flash_work_control.c` (no
work-control header in this family) — with their MODULE_ADDITIONAL_HOST_SOURCES
entries.

**ADDED:** shared `gqa.cuh` consumption (§2), `SparkGemma4GatedGeluKernel`,
`SparkGemma4EmbeddingGatherShardedScaledKernel`, two-pool KV plumbing (§6),
router chain (rms→gemm→softmax→topk→route), 26B parallel-FFN block behind
`SPARK_GEMMA4_MODEL_MOE_BLOCK`, stagepack kinds for the 7-norm/MoE arm, full-layer
inv_freq table buffer.

## 6. Pack plan — bf16 both models; nvfp4 seam; fleet/TP arithmetic

**Codec: bf16 everywhere (weights, KV, activations), both arms.** No quantized
arm is built, ever, by this lane (operator no-self-quantize law). The nvfp4
arm (`/mnt/model-warm/gemma-4-31b-it-nvfp4`, LATER) is a coexistence smoketest
target only. **Seam design (machinery explicitly NOT built now):** the
stagepack format carries a per-tensor `weight_format` (donor field, donor
`SPARK_SYNTH_SELECT_FORMAT`/`PayloadBytes` plumbing) and the module identifier
already carries codec fragments — the nvfp4 arm is a third Makefile +
identifier/target fragments + a packer codec selection, reusing the generic
codec ABI. Nothing in the v1 enum or dispatch is shaped so nvfp4 needs a
driver-source change.

Packer: `tools/gemma4_stagepack.py` (family packer; source law = official
google releases off warm ceph only; repackage-only). Pack-time transforms,
each stated in the contract: fuse `k_proj|v_proj` rows (sliding), fold
`router.scale × H**-0.5` into `router.proj` columns, fold `per_expert_scale[e]`
into expert `down_proj[e]` rows, emit full-layer `inv_freq` table as a global
tensor, assert `layer_scalar == 1.0` (else fail closed), record
`embed_scale ∈ {73.5, 53.0}`, record softcap 30 as inert.

**TP shape (derived from head divisibility; per-rank kv heads must be integral
or duplicated-by-packer):**

31B: q 32; sliding kv 16; full kv 4.
- TP16: full kv 4/16 < 1 — fails. TP8: 4/8 < 1 — fails. **TP4: 32/4=8 q, 16/4=4
  sliding kv, 4/4=1 full kv — exact. → TP4.**

26B: q 16; sliding kv 8; full kv 2.
- TP8: 2/8 fails. **TP4: q 4/rank, sliding kv 2/rank; full kv 2 heads on 4
  ranks → packer duplicates (ranks 0,1 → global kv head 0; ranks 2,3 → head 1;
  per-rank template KV_HEADS=1 on full layers; each rank's 4 q heads sit inside
  one kv group since groups are 8 wide).** Kernel arithmetic is unchanged —
  `kv_head = head / (heads / KV_HEADS)` degenerates to 0 on the local rank,
  which is exactly the one stored head. TP2 (2/2, no duplication) is the
  fallback if duplication is rejected; TP4 proposed for fleet uniformity.
  Precedent for replicated KV under TP: k3's latent cache is fully replicated
  at TP16.

Per-rank KV slot bytes (16 B-aligned, both pass the gqa static_asserts):

| pool | 31B TP4 | 26B TP4 |
|---|---|---|
| sliding slot | 4·(256+256)·2 = **4096 B** | 2·(256+256)·2 = **2048 B** |
| full slot | 1·(512+512)·2 = **2048 B** | 1·(512+512)·2 = **2048 B** (duplicated source) |

Two KV pools per rank (sliding layers, full layers), each its own
`LmKvGeometry<SLOT_BYTES,PAGE_SLOTS,true>` + `LmKvView`, sharing ONE page table
(same sequence/position mapping; views differ only in pool pointer + geometry).
Pool sizing is deployment config (KV_BLOCK_COUNT knob), not contract.

Weights per rank: 31B TP4 ≈ 7.5 GB + stage slice (PP4: 15 layers ≈ 1.8 GB);
26B TP4 ≈ 12.6 GB with duplication, PP2: 15 layers ≈ 6.1 GB. Both trivially
inside the 110 GiB node ceiling.

**Fleet:** 31B = TP4×PP4 on 16 sparks (15 layers/stage). 26B = TP4×PP2 on 8
sparks (15 layers/stage). Port base **65100** — ledger
(61500/62500/63500/64500/64630/64700/64800 laguna/64900 ling/65000 minimax)
has no collision (checked modules/, model-families/, runtime/, tools/*.py,
ROADMAP_TP16_FLEET.md). `SPARK_GEMMA4_STAGE_TP_SESSION_PORTS` = base +
src·4 + sink, degree² uint16 row-major, diag 0; 31b root offset +0, 26b root
offset +100 within the family allocation (the ~200/model convention).

## 7. Validation plan

**Independent anchor:** `validation/spark_gemma4_reference.c` — a from-scratch
C oracle implementing §1-§2 math (bf16 rounding, plain RMS, per-head RMS,
gelu_pytorch_tanh, proportional rope, softmax router, 1024-window GQA) with its
own constants and **zero driver imports** (donor pattern:
`spark_qwen4_flash_reference.c`, 507 lines). The CUDA validator runs the stage
on a synthesized pack (`tools/gemma4_pack_synthesize.c`, template-driven like
the donor's) and compares against the oracle.

- Tolerances: bf16 exactness is NOT claimed for fp32-accumulated GEMM/softmax
  paths; compare with the driver-family standard — argmax token equality over
  scripted prompts plus bounded fp32 relative error on hidden/logit tensors
  (same tiers as qwen4_flash validator: degree-1 stage-0 tier + whole-stack
  TP4-standalone tier via `spark_resident_decode_stage_cuda_validation_common.sh`).
- Boundary cases the validator must include: context < 1024 on a sliding layer;
  context crossing the 1024 boundary; position 1023/1024 window edge; full
  layer after window has scrolled; 26B full-layer kv duplication mapping
  (rank r reads head r/2); layer index 5/11 (first full layers) and last layer
  (L-1 ≡ 5 mod 6); all-zero router probabilities guard (renorm +1e-20 path).
- Hand-computed constants asserted (not fitted): embed scale 73.5 / 53.0
  (bf16-rounded), qk_scale exactly 1.0, proportional table `1e6**(-i/256)`
  i<**64** else 0 (ANCHORS finding 1: 64 rotated pairs, publisher-derived),
  sliding theta 1e4, eps 1e-6, eos {1,106}.
- **CPU-class gates on sparka (no GPU):** module host objects compile with
  `cc -std=c11 -Wall -Wextra -Werror` + cuda_stub (campaign technique);
  `tests/test_gemma4_model_header.py` BINDINGS pass against both contracts;
  stagepack format header compiles with shape assertions (Write-to-/tmp + cc);
  dry-law gate (`tests/test_dry_law.py`) and code-size ratchet
  (`tests/test_code_size.py`) green; contract `--check` byte-exact;
  `Makefile`/`Makefile.moe` Darwin-legible.
- **BLOCKED-ON-DOWNLOAD (marked, not blocking the build):** real-weight pack
  validation, numerical comparison vs HF reference on a spark, fleet bring-up,
  and the revision pin (`GEMMA4_MODEL_REVISION` = HF commit sha) — all gated on
  `/mnt/model-warm/gemma-4-{31b-it,26b-a4b-it}` landing. Design is complete
  without them; contracts carry `"source_revision": "pending-warm-download"`
  until pinned.
- GPU receipts for the module archive itself (synthesized weights) are NOT
  blocked: they run at `make publish` on any spark once code lands (lane law:
  no GPU at design stage; build once on a spark later).

## 8. Risks + the 3 hardest problems

1. **Dual head-dim / dual-KV-geometry dispatch (hardest).** One module,
   two `LmKvGeometry` instantiations and two attention template sets selected
   per layer kind; wrong-kind page-table or pool wiring is the likeliest bug
   class. Mitigation: the validator's boundary matrix (§7) exercises both kinds
   in one process; the shared page table means a wiring error cannot silently
   "work" on one kind.
2. **k_eq_v dataflow.** v derives from the raw k projection with a scale-free
   per-head norm while k is weighted-normed + roped from the same buffer;
   ordering and aliasing must be explicit (three launches: v_norm, k_norm, rope
   — v_norm reads kraw before k_norm overwrites). The oracle computes v from
   the un-roped, un-k-normed projection independently, so a swapped order fails
   numerically, not silently.
3. **26B parallel-FFN + router-on-residual semantics.** Five norms, two
   branches, router reading the pre-norm residual, and renormalise-then-
   per-expert-scale weighting. Any norm-ordering mistake is exactly the
   "wrong tokens" bug class the glm53 lane paid for. Mitigation: oracle asserts
   branch sums, and the pack-time folds (scale→proj columns, per-expert→down
   rows) are each covered by a dedicated packer unit assertion against the
   unfused math.

Other risks, smaller: muse/laguna merge drift (single call-site rebase each,
§4); 26B kv-head duplication rejected by review → TP2 fallback (§6 arithmetic
stands either way); vocab 262144 embed/head shard attention at TP4 (65k rows/
rank — the qwen4 sharded-gather + u64-max combine machinery already operates
at this scale); 60-layer PP4 stage boundaries land on full-attention layers
(index 14/15 boundary) — stage layer lists are explicit data, not derived.

## 9. Acceptance criteria (ordered, testable)

1. `tools/dev/port_family.py` port of §5 file set into
   `modules/gemma4_resident_decode_stage/` + `model-families/gemma4/` with
   non-zero substitution counts for every file; deletion pass grep-verified
   (zero references to Gdn|Chunk|Hc|Indexer|Ple|Screened outside comments-free
   code — and code carries no comments).
2. Both arms compile host-side on Darwin with cuda_stub, `-Werror`, no
   comments, C only: `Makefile` (31B) and `Makefile.moe` (26B) produce distinct
   identifiers and module archives; `#error` fires if `SPARK_GEMMA4_MOE_BUILD`
   tensors appear in the dense pack loader or vice versa (fail-closed enum
   checks).
3. `tests/test_gemma4_model_header.py` green for both contracts (BINDINGS
   cover §1 table; contract `source_revision` present-or-pending).
4. Stagepack format header + `gemma4_pack_synthesize.c` compile and synthesize
   both arm shapes; expected-tensor-count and geometry assertions pass for
   L=60 and L=30 pack headers.
5. `spark_gemma4_reference.c` oracle implements §2 and self-checks its
   constants (73.5/53.0, qk_scale 1.0, inv_freq table, window 1024) — runs on
   sparka (CPU), zero driver imports.
6. CUDA validation tier: validator builds and passes the synthesized-weight
   comparison vs oracle on a spark (queue task), including the §7 boundary
   matrix, for BOTH arms (31B shapes at minimum; 26B MoE path same harness).
7. Offline gates green: dry-law, code-size ratchet, contract --check,
   package manifest regenerated LAST (driver-porting ledger rule).
8. BLOCKED-ON-DOWNLOAD gates (explicitly deferred, listed in the PR report):
   real-weight pack verification vs warm checkpoint, HF-reference numerical
   comparison, revision pin, fleet TP4×PP4 / TP4×PP2 bring-up with telemetry
   receipt.

## Shared-code flags (for the manager)

- **gemma consumes two approved sibling landings and nothing else:**
  `LmHeadRmsNormKernel` (muse, needs `weight_or_null` — v_norm/router-norm have
  no weight) and the laguna `LmRopePerHeadKernel` +3 trailing params (full-
  layer proportional rope as an inv_freq table). No new shared kernels, no
  edits to `gqa.cuh`/`norm.cuh`/`attn.cuh`/`topk.cuh`/`head.cuh` are requested.
- **Correction to manager memory:** gemma-4 norms are PLAIN weighted RMSNorm
  (not `(1+w)`), so gemma does NOT need muse's `LmCenteredRmsNormKernel`.
- Family-local new CUDA (module file only): gated-gelu-tanh kernel, scaled
  sharded embedding gather. Promotion to shared deferred until a second family
  needs them, per the DRY ruling.
- Softcap: none in v1 (monotone under greedy argmax — ruling confirmed against
  `Gemma4ForCausalLM.forward`: logits-only, never inside attention).

## Open questions

1. **26B TP4-with-duplication vs TP2-no-duplication** — design proposes TP4
   (§6); packer duplication is the only extra work. Manager/operator call if
   the fleet wants TP2 for the 8-spark cell instead.
2. Merge order for muse norm.cuh / laguna attn.cuh: gemma codes against the
   frozen signatures and rebases two call sites at merge; confirm neither lane
   plans post-approval signature changes.
3. Warm download ETA for the revision pin — contracts ship
   `pending-warm-download` and the adapter `#error`s until pinned; acceptable?
