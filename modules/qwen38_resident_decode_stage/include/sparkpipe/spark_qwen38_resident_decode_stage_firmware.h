#ifndef SPARKPIPE_SPARK_QWEN38_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_QWEN38_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <stdint.h>

#include "sparkpipe/spark_qwen38_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Qwen 3.8 Max (Qwen3.8-2.4T-A95B) resident decode stage, pipeline- and
 * tensor-parallel from version 1. A node owns a contiguous layer slice and a
 * TP shard of it; the canonical layout is TP4 x PP4 across sixteen nodes.
 *
 * The layer structure is the qwen36 family at larger geometry plus routed
 * MoE: 92 layers = 23 x (3 GatedDeltaNet+MoE, 1 GatedAttention+MoE). The
 * GDN/attention shapes differ from 3.6 only in counts (16/128 GDN heads,
 * 64/4 attention heads, hidden 8192), so the qwen36 kernel family carries
 * over unchanged; the dense FFN is replaced by the routed-MoE path using the
 * common grouped-MoE kernels (MXFP4-E2M1 experts, BF16 spine).
 *
 * Hidden state crossing a stage boundary: rows x 8192 bf16. GDN state, conv
 * tails and KV stay resident on their stage. The head stage owns the final
 * norm and LM head.
 */

#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 1u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 1u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION 1u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION 1u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION 1u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION 1u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_GDN_SNAPSHOT_VIEW_ABI_VERSION 1u

#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION SPARK_QWEN38_MODEL_HIDDEN_DIMENSION
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT SPARK_QWEN38_MODEL_LAYER_COUNT
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT 16u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_TP_DEGREE 16u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP 4096u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT 1024u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS 64u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID UINT32_MAX
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_NO_BLOCK 0xffffffffu

#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 0u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 1u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 2u
#define SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 3u

/*
 * One linear projection, bf16 or MXFP4 payload with per-group E8M0 scales.
 * Same contract as the K3/Qwen36 LinearView; the shared Linear kernel family
 * consumes it.
 */
typedef struct SparkQwen38LinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t input_dimension;
	uint32_t output_dimension;
	const void *weight_payload;
	const uint8_t *weight_scale_e8m0;
	uint64_t weight_payload_bytes;
	uint64_t weight_scale_bytes;
} SparkQwen38LinearView;

/*
 * Gated DeltaNet layer weights, PINNED against the HF checkpoint index at
 * the contract revision: fused in_proj_qkv rows in conv channel order
 * (q 2048 | k 2048 | v 16384), separate 128-row beta/decay projections,
 * depthwise conv kernel 4 with no bias, per-value-head A_log/dt_bias/norm.
 */
typedef struct SparkQwen38GdnLayerWeights
{
	SparkQwen38LinearView qkv;
	SparkQwen38LinearView gate;
	SparkQwen38LinearView beta;
	SparkQwen38LinearView decay;
	SparkQwen38LinearView output;
	const void *conv_weight_bf16;
	const float *a_log_f32;
	const float *dt_bias_f32;
	const void *gdn_norm_weight_bf16;
} SparkQwen38GdnLayerWeights;

/*
 * Full attention layer weights: the query projection fuses a per-head output
 * gate (each head's 512 rows are 256 query then 256 gate), same as qwen36.
 */
typedef struct SparkQwen38AttentionLayerWeights
{
	SparkQwen38LinearView query;
	SparkQwen38LinearView key;
	SparkQwen38LinearView value;
	SparkQwen38LinearView output;
	const void *query_norm_bf16;
	const void *key_norm_bf16;
} SparkQwen38AttentionLayerWeights;

/*
 * Routed MoE weights. Experts are MXFP4-E2M1 with E8M0 group-32 scales;
 * the router, the shared expert and the shared gate stay BF16 (quality-first
 * spine). w1 and w3 are the two halves of the checkpoint's fused gate_up_proj.
 */
typedef struct SparkQwen38MoeWeights
{
	SparkQwen38LinearView gate;
	SparkQwen38LinearView experts_w1;
	SparkQwen38LinearView experts_w3;
	SparkQwen38LinearView experts_w2;
	SparkQwen38LinearView shared_gate;
	SparkQwen38LinearView shared_up;
	SparkQwen38LinearView shared_down;
	const void *shared_gate_weight_bf16;
} SparkQwen38MoeWeights;

/*
 * One layer's complete weight set; the kind fields name which of the three
 * sub-structs are live for this layer.
 */
typedef struct SparkQwen38LayerWeights
{
	uint32_t layer_index;
	uint32_t layer_is_gdn;
	uint32_t layer_is_full_attention;
	SparkQwen38GdnLayerWeights gdn;
	SparkQwen38AttentionLayerWeights attention;
	SparkQwen38MoeWeights moe;
	const void *attention_norm_bf16;
	const void *mlp_norm_bf16;
} SparkQwen38LayerWeights;

#ifdef __cplusplus
}
#endif

#endif
