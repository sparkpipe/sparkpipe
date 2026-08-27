// GLM 5.3 Flash (family glm5_next) geometry for the host tiers.
//
// One geometry source for the family: every literal below is held against
// model_contracts/glm53_flash_authoritative.json by
// tests/test_glm5_next_geometry.py in lockstep, and every shape claim was
// censused from the checkpoint's safetensors headers (76,108 tensors; the
// evidence lives in the contract's checkpoint_tensor_shapes sections).
//
// The family is an ASSEMBLY of three donors:
//   glm52 - MLA projections + DSA indexer + MoE/router + FP8 [128,128] spine
//   k3    - KDA linear attention (kimi delta rule, chunk 64 + recurrent)
//   dsv4  - hyper-connections (mHC) + the kpool compressor mechanism
// The three real deltas are called out where they occur: rope-0 MLA, the
// checkpoint->pack name mapping (model-families/glm5_next/name_map.json),
// and the hybrid 34 KDA / 11 DSA dispatch.
//
// Reference semantics: transformers models/glm5_next/modeling_glm5_next.py
// (text stack only; the vision tower is out of scope for this lane).
#ifndef SPARKPIPE_SPARK_GLM5_NEXT_MODEL_H
#define SPARKPIPE_SPARK_GLM5_NEXT_MODEL_H

#include <stdint.h>

/* -- model ------------------------------------------------------------------ */
#define SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION 4096u
#define SPARK_GLM5_NEXT_MODEL_LAYER_COUNT 45u          /* weight layers 0..44; MTP is 45 */
#define SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX 45u
#define SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT 154880u
#define SPARK_GLM5_NEXT_MODEL_MAXIMUM_CONTEXT_TOKENS 1048576u
#define SPARK_GLM5_NEXT_MODEL_RMS_NORM_EPSILON 1e-05f
#define SPARK_GLM5_NEXT_MODEL_SWIGLU_LIMIT 10.0f
#define SPARK_GLM5_NEXT_MODEL_END_OF_TEXT_TOKEN_ID 154820u
#define SPARK_GLM5_NEXT_MODEL_USER_TOKEN_ID 154827u
#define SPARK_GLM5_NEXT_MODEL_OBSERVATION_TOKEN_ID 154829u
#define SPARK_GLM5_NEXT_MODEL_PAD_TOKEN_ID 154820u

/* Deployment policy (not model constants). */
#define SPARK_GLM5_NEXT_MODEL_KV_POOL_TOKENS 4194304u
#define SPARK_GLM5_NEXT_MODEL_RESTRICTED_VOCAB_COUNT 256u
#define SPARK_GLM5_NEXT_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH 256u

/* -- hybrid dispatch (DELTA 3) -----------------------------------------------
 * 45 weight layers: 34 KDA linear attention + 11 DSA at layers 3, 7, ..., 43
 * (a 3-KDA head, then every 4th). The checkpoint's kda_layers /
 * full_attn_layers lists agree with this closed form; the layer-45 MTP
 * layer is a DSA layer WITHOUT hyper-connections and is dispatched by the
 * speculative path, not this macro. */
#define SPARK_GLM5_NEXT_MODEL_ATTENTION_PERIOD 4u
#define SPARK_GLM5_NEXT_MODEL_GLOBAL_ATTENTION_PHASE 3u
#define SPARK_GLM5_NEXT_MODEL_KDA_LAYER_COUNT 34u
#define SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT 11u
#define SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer_index) \
	(((layer_index) % SPARK_GLM5_NEXT_MODEL_ATTENTION_PERIOD) != \
	 SPARK_GLM5_NEXT_MODEL_GLOBAL_ATTENTION_PHASE)

/* -- KDA linear attention (k3 donor) ----------------------------------------
 * kimi delta rule: 64 heads x 128 key/value, qk L2-normalized in kernel,
 * beta per head through sigmoid, decay per head-PER-CHANNEL (full rank),
 * state fp32 heads x key x value. DELTA vs the released K3 checkpoint:
 * the output gate is the LOW-RANK two-stage g_a[128]->g_b[8192] form with a
 * "safe" sigmoid forget gate; K3 shipped a full-rank g_proj. The pack
 * therefore carries kda_decay_gate_down_weight (fused f_a|g_a rows,
 * replicated) + kda_gate_up_weight instead of k3_gate_weight. */
#define SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT 64u
#define SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION 128u
#define SPARK_GLM5_NEXT_MODEL_KDA_HEAD_VALUE_DIMENSION 128u
#define SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION \
	(SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_KDA_CONV_KERNEL 4u
#define SPARK_GLM5_NEXT_MODEL_KDA_CHUNK_TOKENS 64u
#define SPARK_GLM5_NEXT_MODEL_KDA_QK_L2NORM 1u
#define SPARK_GLM5_NEXT_MODEL_KDA_FULL_RANK_GATE 1u     /* per-head-per-channel decay (dt_bias [8192]) */
#define SPARK_GLM5_NEXT_MODEL_KDA_GATE_LOWER_BOUND -5.0f
#define SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK 128u
#define SPARK_GLM5_NEXT_MODEL_KDA_STATE_ELEMENT_BYTES 4u
#define SPARK_GLM5_NEXT_MODEL_KDA_STATE_ELEMENTS_PER_HEAD \
	(SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION * SPARK_GLM5_NEXT_MODEL_KDA_HEAD_VALUE_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER \
	(SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT * \
	 SPARK_GLM5_NEXT_MODEL_KDA_STATE_ELEMENTS_PER_HEAD * \
	 SPARK_GLM5_NEXT_MODEL_KDA_STATE_ELEMENT_BYTES)
/* Forget gate: -5.0 * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)).
 * A_log is one f32 per head (64; no k3-style source-head slice). */
#define SPARK_GLM5_NEXT_MODEL_KDA_A_LOG_HEAD_COUNT SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT

/* -- MLA on DSA layers (glm52 donor) ----------------------------------------
 * DELTA 1: qk_rope_head_dim = 0. The query is nope-only (256/head), the KV
 * latent is the pure 512 lora (glm52: 576 = 512 + 64 rope), and there is NO
 * rope anywhere in the text stack (the reference passes
 * position_embeddings=None; the indexer never rotates either). Absorbed
 * scoring against the 512-wide latent with scale 256**-0.5; the kernel
 * wants a MLA_ROPE_DIM=0 instantiation of glm52's latent attention.
 * PORTING NOTE (coordinator correction 30e87ec): strip rope from BOTH the
 * MLA scoring AND the dsv4/glm52-donor indexer - config's
 * indexer_rope_interleave is NOT a porting dependency for this family; no
 * rope tables, no rope kernels, no positions input to attention at all. */
#define SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT 64u
#define SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION 1536u
#define SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION 512u
#define SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION 256u
#define SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION 0u
#define SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION 256u
#define SPARK_GLM5_NEXT_MODEL_MLA_USE_NOPE 1u
#define SPARK_GLM5_NEXT_MODEL_MLA_QK_SCALE 0.0625f        /* 256 ** -0.5 */
#define SPARK_GLM5_NEXT_MODEL_MLA_QK_HEAD_DIMENSION \
	(SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + \
	 SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_MLA_QUERY_B_DIMENSION \
	(SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_MLA_QK_HEAD_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION \
	(SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION + \
	 SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_MLA_KV_B_DIMENSION \
	(SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * \
	 (SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + \
	  SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION))
#define SPARK_GLM5_NEXT_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION \
	(SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * \
	 SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION)

/* -- DSA indexer (glm52 donor) + kpool compressor (dsv4 mechanism) ----------
 * Scores POOLS, not tokens: 4-token k-pools mixed by
 * softmax(gate_j + ape_j) over pool positions; per-head relu scores at
 * 128**-0.5; head weights weights_proj(x) * heads**-0.5 summed across
 * heads; select topk/kpool = 512 pools, expand to 2048 tokens plus the
 * incomplete tail (max 3), output width 2051, invalid = -1. NoPE: no rope
 * on index q or k. DELTA vs dsv4-0731: only ape[4,128] + gate[128,hidden]
 * exist (dsv4 carries ape+wkv+wgate+norm at 256 channels on ratio-4
 * layers). */
#define SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT 32u
#define SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION 128u
#define SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION \
	(SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K 2048u
#define SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL 4u
#define SPARK_GLM5_NEXT_MODEL_INDEX_POOL_SELECT_COUNT \
	(SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL)
#define SPARK_GLM5_NEXT_MODEL_INDEX_TAIL_MAX (SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL - 1u)
#define SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH \
	(SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K + SPARK_GLM5_NEXT_MODEL_INDEX_TAIL_MAX)
#define SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE 0.08838834764831845f /* 128 ** -0.5 */
#define SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE 0.17677669529663687f /* 32 ** -0.5 */
#define SPARK_GLM5_NEXT_MODEL_INDEX_NORM_EPSILON 1e-06f  /* LayerNorm k_norm(w, b) */
/* The indexer cache packs [k(128) | gate(128) | valid(1)] per token. */
#define SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION \
	(2u * SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION + 1u)

/* -- hyper-connections (dsv4 donor) -----------------------------------------
 * mHC: fn [24, 4*hidden] over the UNWEIGHTED-RMSNorm'd flattened streams;
 * pre = sigmoid(w*s0 + b0) + eps; post = 2*sigmoid(w*s1 + b1);
 * comb = softmax + eps, one column norm, then 19 row/col norm pairs
 * (sinkhorn 20 total). Streams init to the embedding expanded across 4;
 * the FINAL head collapse is an UNWEIGHTED MEAN (dsv4 uses a weighted
 * hc_head) then RMSNorm. The MTP layer carries no hc_* tensors. */
#define SPARK_GLM5_NEXT_MODEL_HC_MULT 4u
#define SPARK_GLM5_NEXT_MODEL_HC_SINKHORN_ITERATIONS 20u
#define SPARK_GLM5_NEXT_MODEL_HC_EPSILON 1e-06f
#define SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION \
	((2u + SPARK_GLM5_NEXT_MODEL_HC_MULT) * SPARK_GLM5_NEXT_MODEL_HC_MULT)
#define SPARK_GLM5_NEXT_MODEL_HC_FN_COLUMNS \
	(SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT 3u

/* -- MoE (glm52 donor) -------------------------------------------------------
 * Sigmoid router with frozen e_score_correction bias, noaux_tc at
 * n_group = 1, norm_topk_prob, routed scaling 2.5 - the glm52 config
 * verbatim except 288 experts. Router logits in fp32. Dense MLP (int
 * 12288) on the first 3 layers; shared expert intermediate is 2048 (one
 * shared expert). */
#define SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT 288u
#define SPARK_GLM5_NEXT_MODEL_MOE_TOP_K 8u
#define SPARK_GLM5_NEXT_MODEL_MOE_SHARED_EXPERT_COUNT 1u
#define SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION 2048u
#define SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION 12288u
#define SPARK_GLM5_NEXT_MODEL_FIRST_DENSE_LAYER_COUNT 3u
#define SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER 3u
#define SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_SCALING_FACTOR 2.5f
#define SPARK_GLM5_NEXT_MODEL_MOE_NORM_TOPK_PROB 1u
#define SPARK_GLM5_NEXT_MODEL_MOE_W1_COMPONENT_COUNT 2u

/* -- quantisation ------------------------------------------------------------
 * FP8 e4m3 dynamic with [128,128] blocks on: routed experts, shared
 * experts, dense MLP, and the MLA q_a/q_b/kv_a/o projections. KDA tensors,
 * indexer tensors, hc_*, router, norms, embed/lm_head stay BF16/F32 (the
 * contract's precision.fp8_quantized_patterns is the exact set). */
#define SPARK_GLM5_NEXT_MODEL_FP8_SCALE_BLOCK 128u

/* -- KV geometry -------------------------------------------------------------
 * DSA slots hold the pure 512-wide latent (1024 B at bf16; no rope
 * segment). KDA state is fp32 64x128x128 = 4 MiB per layer (~140 MiB
 * across 34 layers). The indexer cache holds the packed 257-float row per
 * token per DSA layer. Page policy matches glm52. */
#define SPARK_GLM5_NEXT_MODEL_KV_BITS 16u
#define SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS 64u
#define SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES \
	((SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION * SPARK_GLM5_NEXT_MODEL_KV_BITS) / 8u)

/* -- derived ---------------------------------------------------------------- */
#define SPARK_GLM5_NEXT_MODEL_ROUTED_LAYERS \
	(SPARK_GLM5_NEXT_MODEL_LAYER_COUNT - SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER)
#define SPARK_GLM5_NEXT_MODEL_GATE_UP_DIMENSION \
	(SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION * \
	 SPARK_GLM5_NEXT_MODEL_MOE_W1_COMPONENT_COUNT)
#define SPARK_GLM5_NEXT_MODEL_WEIGHT_LAYER_COUNT \
	(SPARK_GLM5_NEXT_MODEL_LAYER_COUNT + 1u)  /* + the MTP layer */

/* -- compile-time sanity ----------------------------------------------------- */
#if SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION != 0u
#error glm5_next is the rope-0 MLA instantiation; a nonzero rope dim belongs to another family
#endif
#if SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION % 128u != 0u
#error kda qkv rows must stay whole-head at the TP16 slice
#endif
#if (SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT % 16u) != 0u || \
	(SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT % 16u) != 0u || \
	(SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT % 16u) != 0u
#error glm5_next assumes TP16: every head count must divide by 16
#endif
#if (SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT % 16u) != 0u
#error glm5_next assumes TP16: the expert count must divide into 16 ranks
#endif
#if SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH != 2051u
#error pool expansion width changed; update the indexer consumers
#endif

#endif
