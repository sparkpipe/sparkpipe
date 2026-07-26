#ifndef SPARKPIPE_SPARK_DSV4_MODEL_H
#define SPARKPIPE_SPARK_DSV4_MODEL_H

#include <stdint.h>

/*
 * DeepSeek V4 PRO geometry, pinned against the checkpoint config
 * (huggingface.co/deepseek-ai/DeepSeek-V4-Pro config.json, fetched
 * 2026-07-19). This header defines the SAME macro names as the Flash header
 * with Pro values, under the SAME include guard: a translation unit selects
 * its variant by which header it includes first, and the shared dsv4 module
 * compiles once per variant against one consistent geometry - mixing the
 * two in a TU is structurally impossible.
 *
 * Where Pro diverges from Flash, all CONFIG-pinned: 61 layers over hidden
 * 7168; the attention map opens with TWO HCA layers (not SWA - the config
 * compress_ratios begins 128,128) then alternates CSA/HCA through layer 60
 * for 30 CSA and 31 HCA, with NO sliding-window layers in the main map even
 * though sliding_window 128 remains configured (the SWA branch is a
 * REFERENCE-PIN); 128 query heads x 512 (query dimension 65536) over one kv
 * head, query compression rank 1536, output grouped 16 ways (group width
 * 4096, unchanged); the indexer keeps 64 x 128 heads but selects top-1024;
 * 384 fp4 routed experts top-6 at intermediate 3072 with routed scale 2.5.
 * Everything else matches Flash: dual rope thetas yarn-scaled x16 to 1M,
 * hash-routed first three layers, one shared expert, swiglu clamp 10,
 * hc_mult 4 (boundary stream 4 x 7168 = 28672 elements), one MTP layer,
 * vocab 129280, eps 1e-6. The REFERENCE-PIN set is RESOLVED in the Flash
 * header (one shared model.py serves both repos) and applies verbatim;
 * the Pro-specific answer: the main map has ZERO sliding-window layers,
 * but the SWA branch still exists in Pro because compress_ratios carries
 * n_layers+1 entries and the final 0 makes the MTP layer window-only.
 */

#define SPARK_DSV4_MODEL_HIDDEN_DIMENSION 7168u                 /* CONFIG hidden_size */
#define SPARK_DSV4_MODEL_LAYER_COUNT 61u                        /* CONFIG num_hidden_layers */
#define SPARK_DSV4_MODEL_VOCAB_COUNT 129280u                    /* CONFIG vocab_size */
#define SPARK_DSV4_MODEL_MAX_POSITIONS 1048576u                 /* CONFIG max_position_embeddings */
#define SPARK_DSV4_MODEL_RMS_NORM_EPSILON 1e-6f                 /* CONFIG rms_norm_eps */
#define SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT 128u             /* CONFIG num_attention_heads */
#define SPARK_DSV4_MODEL_ATTN_KV_HEAD_COUNT 1u                  /* CONFIG num_key_value_heads */
#define SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION 512u               /* CONFIG head_dim */
#define SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION 64u                /* CONFIG qk_rope_head_dim */
#define SPARK_DSV4_MODEL_ATTN_ROPE_THETA 10000.0f               /* CONFIG rope_theta */
#define SPARK_DSV4_MODEL_ATTN_YARN_FACTOR 16u                   /* CONFIG rope_scaling.factor */
#define SPARK_DSV4_MODEL_ATTN_YARN_ORIGINAL_POSITIONS 65536u    /* CONFIG rope_scaling.original */
#define SPARK_DSV4_MODEL_COMPRESS_ROPE_THETA 160000.0f          /* CONFIG compress_rope_theta */
#define SPARK_DSV4_MODEL_QUERY_LORA_RANK 1536u                  /* CONFIG q_lora_rank */
#define SPARK_DSV4_MODEL_OUTPUT_LORA_RANK 1024u                 /* CONFIG o_lora_rank */
#define SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT 16u                 /* CONFIG o_groups */
#define SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS 128u             /* CONFIG sliding_window */
#define SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO 4u                  /* CONFIG compress_ratios */
#define SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO 128u                /* CONFIG compress_ratios */
#define SPARK_DSV4_MODEL_INDEX_HEAD_COUNT 64u                   /* CONFIG index_n_heads */
#define SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION 128u              /* CONFIG index_head_dim */
#define SPARK_DSV4_MODEL_INDEX_TOP_K 1024u                      /* CONFIG index_topk */
#define SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT 384u               /* CONFIG n_routed_experts */
#define SPARK_DSV4_MODEL_SHARED_EXPERT_COUNT 1u                 /* CONFIG n_shared_experts */
#define SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN 6u                   /* CONFIG num_experts_per_tok */
#define SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION 3072u    /* CONFIG moe_intermediate_size */
#define SPARK_DSV4_MODEL_HASH_ROUTED_LAYER_COUNT 3u             /* CONFIG num_hash_layers */
#define SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR 2.5f             /* CONFIG routed_scaling_factor */
#define SPARK_DSV4_MODEL_SWIGLU_LIMIT 10.0f                     /* CONFIG swiglu_limit */
#define SPARK_DSV4_MODEL_HC_STREAM_COUNT 4u                     /* CONFIG hc_mult */
#define SPARK_DSV4_MODEL_MTP_LAYER_COUNT 1u                     /* CONFIG num_nextn_predict_layers */
#define SPARK_DSV4_MODEL_ROPE_BETA_FAST 32u                     /* CONFIG beta_fast */
#define SPARK_DSV4_MODEL_ROPE_BETA_SLOW 1u                      /* CONFIG beta_slow */
#define SPARK_DSV4_MODEL_HC_SINKHORN_ITERATIONS 20u             /* CONFIG hc_sinkhorn_iters, RUNS AT INFERENCE */
#define SPARK_DSV4_MODEL_HC_EPSILON 1e-6f                       /* CONFIG hc_eps */
#define SPARK_DSV4_MODEL_MTP_LAYER_KIND SPARK_DSV4_MODEL_LAYER_KIND_SWA /* compress_ratios[n_layers] == 0 */
#define SPARK_DSV4_MODEL_KV_QUANT_BLOCK 64u                     /* act_quant(kv nope dims, 64) */
#define SPARK_DSV4_MODEL_ACT_QUANT_BLOCK 128u                   /* linear activation quant group */
#define SPARK_DSV4_MODEL_FP4_QUANT_BLOCK 32u                    /* fp4_block_size */
#define SPARK_DSV4_MODEL_FP8_MAX 448.0f
#define SPARK_DSV4_MODEL_FP4_MAX 6.0f
#define SPARK_DSV4_MODEL_QUANT_AMAX_FLOOR 1e-4f
#define SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR 2u                  /* ratio-4 compressor doubles channels */
#define SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES 2u

#define SPARK_DSV4_MODEL_HC_MIX_ROWS ((2u + SPARK_DSV4_MODEL_HC_STREAM_COUNT) * SPARK_DSV4_MODEL_HC_STREAM_COUNT)

#define SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION (SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION (SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION / SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT)
#define SPARK_DSV4_MODEL_INDEX_DIMENSION (SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION)
#define SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS (SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HIDDEN_DIMENSION)

#define SPARK_DSV4_MODEL_LAYER_KIND_SWA 0u
#define SPARK_DSV4_MODEL_LAYER_KIND_CSA 1u
#define SPARK_DSV4_MODEL_LAYER_KIND_HCA 2u

// Transcribed from CONFIG compress_ratios: two HCA openers, then CSA on
// even indices and HCA on odd through layer 60; the array's 62nd entry
// belongs to the MTP layer and is a REFERENCE-PIN.
static const uint8_t SPARK_DSV4_MODEL_LAYER_KIND[SPARK_DSV4_MODEL_LAYER_COUNT] =
{
	2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,
	2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1
};

static inline uint32_t SparkDsv4ModelLayerKind(uint32_t layer_index)
{
	return(layer_index < SPARK_DSV4_MODEL_LAYER_COUNT ? (uint32_t)SPARK_DSV4_MODEL_LAYER_KIND[layer_index] : SPARK_DSV4_MODEL_LAYER_KIND_HCA);
}

#endif
