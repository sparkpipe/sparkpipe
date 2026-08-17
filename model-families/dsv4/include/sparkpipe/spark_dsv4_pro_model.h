#pragma once

#include <stdint.h>

#include "sparkpipe/spark_weight_codec.h"

/* Generated from the exact source revision by tools/generate_dsv4_contracts.py. */
#define SPARK_DSV4_PRO_ID "deepseek-ai/DeepSeek-V4-Pro-0813"
#define SPARK_DSV4_PRO_SOURCE_REVISION "GA release deepseek-ai/DeepSeek-V4-Pro-0813 (HF, 2026-08-13)"

#define SPARK_DSV4_PRO_HIDDEN_DIMENSION 7168u
#define SPARK_DSV4_PRO_LAYER_COUNT 61u
#define SPARK_DSV4_PRO_MTP_LAYER_COUNT 3u
/* DSpark speculative stage. */
#define SPARK_DSV4_PRO_DSPARK_BLOCK_SIZE 5u
#define SPARK_DSV4_PRO_DSPARK_TARGET_LAYER_COUNT 3u
#define SPARK_DSV4_PRO_DSPARK_MARKOV_RANK 512u
#define SPARK_DSV4_PRO_DSPARK_NOISE_TOKEN_ID 128799u
/* Spec step: the anchor's speculative chain is SPEC_STEP drafts + the
 * anchor itself, so the batched verify island is SPEC_STEP+1 = 8 rows.
 * Matches the Flash GA geometry (block 5, spec 7). */
#define SPARK_DSV4_PRO_DSPARK_SPEC_STEP 7u
#define SPARK_DSV4_PRO_VOCAB_COUNT 129280u
#define SPARK_DSV4_PRO_QUERY_LORA_RANK 1536u
#define SPARK_DSV4_PRO_OUTPUT_LORA_RANK 1024u
#define SPARK_DSV4_PRO_OUTPUT_GROUP_COUNT 16u
#define SPARK_DSV4_PRO_INDEX_HEAD_COUNT 64u
#define SPARK_DSV4_PRO_INDEX_HEAD_DIMENSION 128u
#define SPARK_DSV4_PRO_INDEX_TOP_K 1024u
#define SPARK_DSV4_PRO_SLIDING_WINDOW_TOKENS 128u
#define SPARK_DSV4_PRO_YARN_FACTOR 16u
#define SPARK_DSV4_PRO_ROUTED_EXPERT_COUNT 384u
#define SPARK_DSV4_PRO_SHARED_EXPERT_COUNT 1u
#define SPARK_DSV4_PRO_EXPERTS_PER_TOKEN 6u
#define SPARK_DSV4_PRO_EXPERT_INTERMEDIATE_DIMENSION 3072u
#define SPARK_DSV4_PRO_HASH_ROUTED_LAYER_COUNT 3u
#define SPARK_DSV4_PRO_MAXIMUM_CONTEXT_TOKENS 1048576u
#define SPARK_DSV4_PRO_ATTENTION_HEAD_COUNT 128u
#define SPARK_DSV4_PRO_KV_HEAD_COUNT 1u
#define SPARK_DSV4_PRO_HEAD_DIMENSION 512u
#define SPARK_DSV4_PRO_QK_ROPE_HEAD_DIMENSION 64u
#define SPARK_DSV4_PRO_YARN_ORIGINAL_CONTEXT_TOKENS 65536u
#define SPARK_DSV4_PRO_HYPER_CONNECTION_STREAM_COUNT 4u
#define SPARK_DSV4_PRO_HYPER_CONNECTION_SINKHORN_ITERATIONS 20u
#define SPARK_DSV4_PRO_RMS_NORM_EPSILON 1e-06f
#define SPARK_DSV4_PRO_ROPE_THETA 10000.0f
#define SPARK_DSV4_PRO_COMPRESSED_ROPE_THETA 160000.0f
#define SPARK_DSV4_PRO_HYPER_CONNECTION_EPSILON 1e-06f
#define SPARK_DSV4_PRO_ROUTED_SCALING_FACTOR 2.5f
#define SPARK_DSV4_PRO_SWIGLU_LIMIT 10.0f
#if defined(SPARK_DSV4_PRO_EXPERT_CODEC_FP8_E4M3)
/* Variant builds: FP8-E4M3 expert weights (requires the FP8 expert kernel
 * variant and an FP8-expert pack; default remains MXFP4-E2M1). */
#define SPARK_DSV4_PRO_EXPERT_WEIGHT_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3
#else
#define SPARK_DSV4_PRO_EXPERT_WEIGHT_CODEC SPARK_WEIGHT_CODEC_MXFP4_E2M1
#endif
#define SPARK_DSV4_PRO_NON_EXPERT_WEIGHT_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3
#if defined(SPARK_DSV4_PRO_KV_CODEC_FP8_E4M3)
/* Variant builds: E4M3 KV cache with UE8M0 block scales (block 64, rope
 * tail kept BF16) - matches the reference act_quant layout; requires the
 * FP8-KV cache kernels and an FP8-KV pack header. Default remains BF16. */
#define SPARK_DSV4_PRO_KV_CACHE_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3
#else
#define SPARK_DSV4_PRO_KV_CACHE_CODEC SPARK_WEIGHT_CODEC_BF16
#endif
#define SPARK_DSV4_PRO_NON_EXPERT_ACTIVATION_CODEC SPARK_ACTIVATION_CODEC_NONE /* first-light: BF16 activations, matching the Flash-validated kernel set */
#define SPARK_DSV4_PRO_EXPERT_ACTIVATION_CODEC SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0
#define SPARK_DSV4_PRO_OUTPUT_COMPOSITION_ACTIVATION_CODEC SPARK_ACTIVATION_CODEC_NONE

static const uint16_t SparkDsv4ProCompressionRatios[64u] =
{
    128u, 128u, 4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u,
    4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u,
    4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u,
    4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u,
    4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u, 4u, 128u,
    4u, 0u, 0u, 0u
};

static inline uint16_t SparkDsv4ProBackboneCompressionRatio(uint32_t layer_index)
{
	if ( layer_index >= SPARK_DSV4_PRO_LAYER_COUNT )
		return(UINT16_MAX);
	return(SparkDsv4ProCompressionRatios[layer_index]);
}

static inline uint16_t SparkDsv4ProMtpCompressionRatio(void)
{
	return(SparkDsv4ProCompressionRatios[SPARK_DSV4_PRO_LAYER_COUNT]);
}
