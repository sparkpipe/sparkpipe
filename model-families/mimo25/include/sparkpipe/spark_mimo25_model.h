#ifndef SPARKPIPE_SPARK_MIMO25_MODEL_H
#define SPARKPIPE_SPARK_MIMO25_MODEL_H

#include <stdint.h>

/*
 * MiMo-V2.5 (Base, multimodal checkpoint; text stack only) model constants, derived 2026-07-19 from the released
 * config.json, modeling_mimo_v2.py and the safetensors headers (ranged
 * fetch of shard JSON - exact dtypes and shapes, not inference).
 *
 * REFERENCE-PINs resolved against modeling_mimo_v2.py:
 * - fused qkv split order is [q | k | v] flat, head-major within each
 *   segment: sizes heads*192, kv*192, kv*128 (attention_projection_layout
 *   is fused_qkv in BOTH released configs; the split layout exists in the
 *   class but no released checkpoint uses it).
 * - value scale multiplies value_states BEFORE the cache write
 *   (_forward_attention scales, then past_key_values.update) - cached
 *   values are pre-scaled; mathematically a constant fold, pinned as
 *   cached-scaled to match the reference byte flow.
 * - rope is HALF-SPLIT rotate_half pairing (i, i + rope_dim/2) applied to
 *   the FIRST rope_dim dims of each head; the remaining dims pass
 *   through. rope_dim = int(head_dim * 0.334) = 64. Full-attention
 *   branch theta 1e7, SWA and MTP branch theta 1e4, no interpolation
 *   (rope_scaling default), attention_scaling 1.0.
 * - the sink bias is one bf16 per QUERY head, appended as an EXTRA LOGIT
 *   COLUMN before the softmax and dropped after - it joins the
 *   denominator only. Present on SWA layers only
 *   (add_swa_attention_sink_bias true, add_full false); layer-0 tensors
 *   confirm no sink on the full branch. MTP layers carry it.
 * - the router is sigmoid scores from an F32 gate linear; noaux_tc
 *   selection on scores + e_score_correction_bias with n_group =
 *   topk_group = 1 (the group machinery degenerates to a plain top-k);
 *   weights gather the ORIGINAL sigmoid scores, sum-normalize with
 *   +1e-20, and routed_scaling_factor is null = 1.0. The routing weight
 *   multiplies the expert OUTPUT (after down_proj), not the
 *   intermediate.
 * - expert MLP is plain silu(gate)*up -> down, NO clamp anywhere; layer
 *   zero is a dense MLP at the full intermediate size on both variants.
 * - FP8 weights are e4m3 with F32 scale_inv per [128,128] 2-D BLOCK
 *   (dequant multiplies by scale_inv); o_proj, embedding, head, norms
 *   and sinks are bf16; the router gate and its bias are f32. Every fp8
 *   dimension in both checkpoints divides 128 exactly.
 * - MTP is THREE sequential draft layers (model.mtp.layers.0..2), each:
 *   eh_proj [hidden, 2*hidden] bf16 over the concat
 *   [enorm(embed(token)) | hnorm(hidden)] (halves pinned by the
 *   DeepSeek-V3 convention the shapes follow), then an SWA-branch
 *   attention block WITH sink and a dense MLP at the full intermediate
 *   size, pre_mlp_layernorm as the post-attention norm name, and a
 *   per-layer final_layernorm. Shares the main embedding and lm_head.
 *   The HF remote code ignores model.mtp.* - the checkpoint index and
 *   shard headers are the ground truth for the structure.
 */

#define SPARK_MIMO25_MODEL_HIDDEN_DIMENSION 4096u             /* CONFIG hidden_size */
#define SPARK_MIMO25_MODEL_LAYER_COUNT 48u                    /* CONFIG num_hidden_layers */
#define SPARK_MIMO25_MODEL_VOCAB_COUNT 152576u             /* CONFIG vocab_size */
#define SPARK_MIMO25_MODEL_MAX_POSITIONS 1048576u          /* CONFIG max_position_embeddings */
#define SPARK_MIMO25_MODEL_RMS_NORM_EPSILON 1e-5f               /* CONFIG layernorm_epsilon */
#define SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES 2u

#define SPARK_MIMO25_MODEL_ATTN_HEAD_COUNT 64u
#define SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION 192u             /* CONFIG head_dim (q and k) */
#define SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION 128u            /* CONFIG v_head_dim */
#define SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION 64u              /* int(head_dim * partial_rotary_factor) */
#define SPARK_MIMO25_MODEL_FULL_KV_HEAD_COUNT 4u
#define SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT 8u
#define SPARK_MIMO25_MODEL_FULL_QKV_DIMENSION 13568u          /* checkpoint qkv rows, full branch */
#define SPARK_MIMO25_MODEL_SWA_QKV_DIMENSION 14848u           /* checkpoint qkv rows, SWA branch */
#define SPARK_MIMO25_MODEL_Q_DIMENSION 12288u
#define SPARK_MIMO25_MODEL_O_INPUT_DIMENSION 8192u           /* heads * v_head_dim */
#define SPARK_MIMO25_MODEL_SLIDING_WINDOW_TOKENS 128u
#define SPARK_MIMO25_MODEL_ATTN_VALUE_SCALE 0.707f
#define SPARK_MIMO25_MODEL_FULL_ROPE_THETA 10000000.0f
#define SPARK_MIMO25_MODEL_SWA_ROPE_THETA 10000.0f

#define SPARK_MIMO25_MODEL_DENSE_INTERMEDIATE_DIMENSION 16384u
#define SPARK_MIMO25_MODEL_EXPERT_INTERMEDIATE_DIMENSION 2048u
#define SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT 256u
#define SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN 8u
#define SPARK_MIMO25_MODEL_ROUTED_SCALING_FACTOR 1.0f           /* CONFIG routed_scaling_factor null */
#define SPARK_MIMO25_MODEL_ROUTER_NORM_EPSILON 1e-20f           /* topk weight sum guard */

#define SPARK_MIMO25_MODEL_FP8_SCALE_BLOCK 128u                 /* [128,128] f32 scale_inv blocks */
#define SPARK_MIMO25_MODEL_MTP_LAYER_COUNT 3u

#define SPARK_MIMO25_MODEL_LAYER_KIND_FULL 0u
#define SPARK_MIMO25_MODEL_LAYER_KIND_SWA 1u

/* CONFIG hybrid_layer_pattern: 1 = SWA, 0 = full attention. */
static const uint8_t SPARK_MIMO25_MODEL_LAYER_KIND[SPARK_MIMO25_MODEL_LAYER_COUNT] =
{
	0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,
	1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,
	1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0
};

/* CONFIG moe_layer_freq: 1 = MoE, 0 = dense MLP. */
static const uint8_t SPARK_MIMO25_MODEL_LAYER_IS_MOE[SPARK_MIMO25_MODEL_LAYER_COUNT] =
{
	0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

static inline uint32_t SparkMimo25ModelLayerKind(uint32_t layer_index)
{
	return(layer_index < SPARK_MIMO25_MODEL_LAYER_COUNT ? (uint32_t)SPARK_MIMO25_MODEL_LAYER_KIND[layer_index] : SPARK_MIMO25_MODEL_LAYER_KIND_SWA);
}

static inline uint32_t SparkMimo25ModelLayerIsMoe(uint32_t layer_index)
{
	return(layer_index < SPARK_MIMO25_MODEL_LAYER_COUNT ? (uint32_t)SPARK_MIMO25_MODEL_LAYER_IS_MOE[layer_index] : 1u);
}

#endif
