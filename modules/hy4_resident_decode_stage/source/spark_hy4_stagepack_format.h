#pragma once

#include <stdint.h>

#include "sparkpipe/spark_hy4_model.h"
#include "sparkpipe/spark_stagepack_format.h"
#include "sparkpipe/spark_status.h"

/* hy4 TP16 stagepack wire format. One pack per rank; the fused routed
 * experts are dim0 range-split (16 local experts per rank), the lm_head
 * is vocab range-split (7552 rows per rank), attention output is dim1
 * column-gathered, everything else is replicated. Weights are F8_E4M3
 * payloads with U8 E8M0 group-32 scales (kind 6, the checkpoint's own
 * quantization), norms/router F32. */

#define SPARK_HY4_STAGEPACK_MAGIC 0x50533448u
#define SPARK_HY4_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_HY4_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_HY4_STAGEPACK_PAYLOAD_ALIGNMENT 256u

#define SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_BF16 SPARK_STAGEPACK_FORMAT_WEIGHT_BF16
#define SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_F32 SPARK_STAGEPACK_FORMAT_WEIGHT_F32
#define SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_I64 SPARK_STAGEPACK_FORMAT_WEIGHT_I64
#define SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_FP8_E4M3_E8M0B32 9u

typedef enum SparkHy4StagePackTensorKind
{
	SPARK_HY4_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_HY4_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_HY4_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_HY4_STAGEPACK_TENSOR_ATTN_NORM = 3,
	SPARK_HY4_STAGEPACK_TENSOR_ATTN_SINKS = 4,
	SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_A = 5,
	SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_A_NORM = 6,
	SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_B = 7,
	SPARK_HY4_STAGEPACK_TENSOR_ATTN_KV_A = 8,
	SPARK_HY4_STAGEPACK_TENSOR_ATTN_KV_A_NORM = 9,
	SPARK_HY4_STAGEPACK_TENSOR_ATTN_K_B = 10,
	SPARK_HY4_STAGEPACK_TENSOR_ATTN_V_B = 11,
	SPARK_HY4_STAGEPACK_TENSOR_ATTN_GATE = 12,
	SPARK_HY4_STAGEPACK_TENSOR_ATTN_OUTPUT = 13,
	SPARK_HY4_STAGEPACK_TENSOR_INDEX_Q = 14,
	SPARK_HY4_STAGEPACK_TENSOR_INDEX_K = 15,
	SPARK_HY4_STAGEPACK_TENSOR_INDEX_Q_NORM = 16,
	SPARK_HY4_STAGEPACK_TENSOR_INDEX_K_NORM = 17,
	SPARK_HY4_STAGEPACK_TENSOR_INDEX_WQ_B = 18,
	SPARK_HY4_STAGEPACK_TENSOR_INDEX_WK = 19,
	SPARK_HY4_STAGEPACK_TENSOR_INDEX_PROJ = 20,
	SPARK_HY4_STAGEPACK_TENSOR_INDEX_NORM = 21,
	SPARK_HY4_STAGEPACK_TENSOR_MLP_NORM = 22,
	SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_FN = 23,
	SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_SCALE = 24,
	SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_BASE = 25,
	SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_FN = 26,
	SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_SCALE = 27,
	SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_BASE = 28,
	SPARK_HY4_STAGEPACK_TENSOR_MOE_GATE = 29,
	SPARK_HY4_STAGEPACK_TENSOR_MOE_GATE_BIAS = 30,
	SPARK_HY4_STAGEPACK_TENSOR_MOE_W1 = 31,
	SPARK_HY4_STAGEPACK_TENSOR_MOE_W3 = 32,
	SPARK_HY4_STAGEPACK_TENSOR_MOE_DOWN = 33,
	SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_GATE = 34,
	SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_UP = 35,
	SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_DOWN = 36,
	SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_FN = 37,
	SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_SCALE = 38,
	SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_BASE = 39,
	SPARK_HY4_STAGEPACK_TENSOR_KIND_COUNT = 40
} SparkHy4StagePackTensorKind;

typedef struct SparkHy4StagePackHeader
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t header_bytes;
	uint32_t directory_entry_bytes;
	uint32_t tensor_count;
	uint32_t hidden_dimension;
	uint32_t layer_count;
	uint32_t first_layer_index;
	uint32_t total_layer_count;
	uint32_t attention_period;
	uint32_t full_attention_phase;
	uint32_t gdn_key_head_count;
	uint32_t gdn_value_head_count;
	uint32_t gdn_head_key_dimension;
	uint32_t gdn_head_value_dimension;
	uint32_t gdn_conv_kernel;
	uint32_t attn_query_head_count;
	uint32_t attn_kv_head_count;
	uint32_t attn_head_dimension;
	uint32_t attn_rope_dimension;
	uint32_t routed_expert_count;
	uint32_t experts_per_token;
	uint32_t expert_intermediate_dimension;
	uint32_t output_vocab_count;
	uint32_t mxfp4_group_size;
	uint32_t mtp_layer_count;
	uint64_t directory_offset;
	uint64_t file_bytes;
} SparkHy4StagePackHeader;

typedef struct SparkHy4StagePackEntry
{
	uint32_t tensor_kind;
	uint32_t layer_index;
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	uint32_t scale_group_size;
	uint64_t payload_offset;
	uint64_t payload_bytes;
	uint64_t scale_offset;
	uint64_t scale_bytes;
} SparkHy4StagePackEntry;

#define SPARK_HY4_STAGEPACK_HEADER_BYTES 120u
#define SPARK_HY4_STAGEPACK_ENTRY_BYTES 56u
_Static_assert(sizeof(SparkHy4StagePackHeader) == SPARK_HY4_STAGEPACK_HEADER_BYTES,"hy4 stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkHy4StagePackEntry) == SPARK_HY4_STAGEPACK_ENTRY_BYTES,"hy4 stage pack directory entry must be 56 wire bytes");

_Static_assert(SPARK_HY4_MODEL_ATTN_QUERY_HEAD_COUNT == (SPARK_HY4_MODEL_TP_RANKS * SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK),"hy4 query heads must tile the 16 ranks");
_Static_assert((SPARK_HY4_MODEL_QK_HEAD_DIMENSION % 2u) == 0u,"hy4 rope dimension must pair");
_Static_assert(SPARK_HY4_MODEL_QK_NOPE_HEAD_DIMENSION + SPARK_HY4_MODEL_QK_ROPE_HEAD_DIMENSION == SPARK_HY4_MODEL_QK_HEAD_DIMENSION,"hy4 nope+rope must cover the head");
_Static_assert((SPARK_HY4_MODEL_VOCAB_COUNT % SPARK_HY4_MODEL_TP_RANKS) == 0u,"hy4 vocab must tile the ranks");
_Static_assert((SPARK_HY4_MODEL_ROUTED_EXPERT_COUNT % SPARK_HY4_MODEL_TP_RANKS) == 0u,"hy4 experts must tile the ranks");
_Static_assert((SPARK_HY4_MODEL_EXPERT_INTERMEDIATE_DIMENSION % SPARK_HY4_MODEL_EXPERT_SCALE_GROUP_SIZE) == 0u,"hy4 expert intermediate must tile scale groups");
_Static_assert((SPARK_HY4_MODEL_HIDDEN_DIMENSION % SPARK_HY4_MODEL_EXPERT_SCALE_GROUP_SIZE) == 0u,"hy4 hidden must tile scale groups");

typedef SparkStagePackTensorShape SparkHy4StagePackTensorShape;

static const SparkStagePackGeometryTable SparkHy4StagePackGeometry =
{
	.norm_width = SPARK_HY4_MODEL_HIDDEN_DIMENSION,
	.hidden_dimension = SPARK_HY4_MODEL_HIDDEN_DIMENSION,
	.routed_expert_count = SPARK_HY4_MODEL_ROUTED_EXPERT_COUNT,
	.expert_intermediate_dimension = SPARK_HY4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,
	.gdn_conv_channels = 0u,
	.gdn_value_dimension = 0u,
	.gdn_value_head_count = 0u,
	.gdn_head_value_dimension = 0u,
	.gdn_conv_kernel = 0u
};

/* Per-rank (tp_degree 16) shapes. Row/column are the stored 2-D
 * dims of the packed tensor as written by the sharder. */
static inline int32_t SparkHy4StagePackShapeGlobal(uint32_t tensor_kind,
	SparkHy4StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL;
	switch ( tensor_kind )
	{
	case SPARK_HY4_STAGEPACK_TENSOR_EMBEDDING:
		shape->rows = SPARK_HY4_MODEL_VOCAB_PER_RANK;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows = SPARK_HY4_MODEL_VOCAB_PER_RANK;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_FN:
		shape->rows = SPARK_HY4_MODEL_HC_FN_OUTPUT_ROWS;
		shape->columns = SPARK_HY4_MODEL_HC_FLAT_WIDTH;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_SCALE:
		shape->rows = 1u;
		shape->columns = 1u;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_BASE:
		shape->rows = SPARK_HY4_MODEL_HC_STREAM_COUNT;
		shape->columns = 1u;
		return 0;
	default:
		return -1;
	}
}

static inline int32_t SparkHy4StagePackShapeEveryLayer(
	uint32_t tensor_kind, SparkHy4StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_NORM:
	case SPARK_HY4_STAGEPACK_TENSOR_MLP_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_SINKS:
		shape->rows = SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK;
		shape->columns = 1u;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_A:
		shape->rows = SPARK_HY4_MODEL_QUERY_LORA_RANK;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_A_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_HY4_MODEL_QUERY_LORA_RANK;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_B:
		shape->rows = SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK * SPARK_HY4_MODEL_QK_HEAD_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_QUERY_LORA_RANK;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_KV_A:
		shape->rows = SPARK_HY4_MODEL_KV_LORA_RANK + SPARK_HY4_MODEL_QK_ROPE_HEAD_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_KV_A_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_HY4_MODEL_KV_LORA_RANK;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_K_B:
		shape->rows = SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK * SPARK_HY4_MODEL_KV_LORA_RANK * SPARK_HY4_MODEL_QK_NOPE_HEAD_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_QUERY_LORA_RANK;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_V_B:
		shape->rows = SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK * SPARK_HY4_MODEL_V_HEAD_DIMENSION * SPARK_HY4_MODEL_KV_LORA_RANK;
		shape->columns = SPARK_HY4_MODEL_QUERY_LORA_RANK;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_GATE:
		shape->rows = SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK * SPARK_HY4_MODEL_V_HEAD_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->rows = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK * SPARK_HY4_MODEL_V_HEAD_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_Q:
		shape->rows = SPARK_HY4_MODEL_INDEX_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_Q_NORM:
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_K_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_HY4_MODEL_INDEX_HEAD_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_WQ_B:
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_WK:
		shape->rows = SPARK_HY4_MODEL_INDEX_HEADS_PER_RANK * SPARK_HY4_MODEL_INDEX_HEAD_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_PROJ:
		shape->rows = SPARK_HY4_MODEL_INDEX_HEADS_PER_RANK * SPARK_HY4_MODEL_INDEX_HEAD_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_KV_LORA_RANK;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_FN:
	case SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_FN:
		shape->rows = SPARK_HY4_MODEL_HC_FN_OUTPUT_ROWS;
		shape->columns = SPARK_HY4_MODEL_HC_FLAT_WIDTH;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_SCALE:
	case SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_SCALE:
		shape->rows = 1u;
		shape->columns = 2u;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_BASE:
	case SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_BASE:
		shape->rows = 2u * SPARK_HY4_MODEL_HC_STREAM_COUNT;
		shape->columns = 1u;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_GATE:
		shape->rows = SPARK_HY4_MODEL_ROUTED_EXPERT_COUNT;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_GATE_BIAS:
		shape->rows = SPARK_HY4_MODEL_ROUTED_EXPERT_COUNT;
		shape->columns = 1u;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_W1:
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_W3:
		shape->rows = SPARK_HY4_MODEL_EXPERTS_PER_RANK * SPARK_HY4_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_DOWN:
		shape->rows = SPARK_HY4_MODEL_EXPERTS_PER_RANK * SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_GATE:
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_UP:
		shape->rows = SPARK_HY4_MODEL_DENSE_FFN_INTERMEDIATE_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_DOWN:
		shape->rows = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_HY4_MODEL_DENSE_FFN_INTERMEDIATE_DIMENSION;
		return 0;
	default:
		return -1;
	}
}

static inline int32_t SparkHy4StagePackTensorShapeOf(uint32_t tensor_kind,
	SparkHy4StagePackTensorShape *shape)
{
	shape->natural_format = SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_F32;
	if ( SparkHy4StagePackShapeGlobal(tensor_kind,shape) == 0 )
		return 0;
	if ( SparkHy4StagePackShapeEveryLayer(tensor_kind,shape) == 0 )
		return 0;
	return -1;
}

/* Natural (checkpoint-side) format per class. The rank pack stores
 * exactly these bytes; there is no on-load requantization. */
static inline uint32_t SparkHy4StagePackNaturalFormat(uint32_t tensor_kind)
{
	switch ( tensor_kind )
	{
	case SPARK_HY4_STAGEPACK_TENSOR_EMBEDDING:
		return SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_BF16;
	case SPARK_HY4_STAGEPACK_TENSOR_LM_HEAD:
	case SPARK_HY4_STAGEPACK_TENSOR_FINAL_NORM:
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_NORM:
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_SINKS:
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_A_NORM:
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_KV_A_NORM:
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_Q_NORM:
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_K_NORM:
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_NORM:
	case SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_FN:
	case SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_SCALE:
	case SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_BASE:
	case SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_FN:
	case SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_SCALE:
	case SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_BASE:
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_GATE_BIAS:
	case SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_FN:
	case SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_SCALE:
	case SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_BASE:
		return SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_F32;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_B:
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_K_B:
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_V_B:
		return SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_BF16;
	default:
		return SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_FP8_E4M3_E8M0B32;
	}
}

static inline uint32_t SparkHy4StagePackScaleGroup(uint32_t tensor_kind)
{
	if ( SparkHy4StagePackNaturalFormat(tensor_kind) ==
		SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_FP8_E4M3_E8M0B32 )
		return SPARK_HY4_MODEL_EXPERT_SCALE_GROUP_SIZE;
	return 0u;
}

static inline uint64_t SparkHy4StagePackPayloadBytes(uint32_t weight_format,
	uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_FP8_E4M3_E8M0B32 )
		return elements;
	if ( weight_format == SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_F32 )
		return elements * 4u;
	if ( weight_format == SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_I64 )
		return elements * 8u;
	return elements * 2u;
}

static inline uint64_t SparkHy4StagePackScaleBytes(uint32_t weight_format,
	uint32_t rows, uint32_t columns)
{
	if ( weight_format == SPARK_HY4_STAGEPACK_WEIGHT_FORMAT_FP8_E4M3_E8M0B32 )
	{
		uint64_t groups = ((uint64_t)rows * (uint64_t)columns) /
			SPARK_HY4_MODEL_EXPERT_SCALE_GROUP_SIZE;
		/* scale rows follow the payload row layout: per stored row,
		 * columns/group entries */
		return ((uint64_t)rows * ((uint64_t)columns /
			SPARK_HY4_MODEL_EXPERT_SCALE_GROUP_SIZE)) != 0ull ?
			((uint64_t)rows * ((uint64_t)columns /
			SPARK_HY4_MODEL_EXPERT_SCALE_GROUP_SIZE)) : groups;
	}
	return 0u;
}

static inline int32_t SparkHy4StagePackResolvedShape(uint32_t tensor_kind,
	uint32_t layer_index, uint32_t is_global,
	SparkHy4StagePackTensorShape *shape)
{
	if ( SparkHy4StagePackTensorShapeOf(tensor_kind,shape) < 0 )
		return -1;
	if ( (shape->layer_class == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL)
		!= (is_global != 0u) )
		return -2;
	if ( is_global != 0u )
		return 0;
	if ( layer_index >= SPARK_HY4_MODEL_LAYER_COUNT )
		return -3;
	return 0;
}

static inline uint32_t SparkHy4StagePackExpectedTensorCount(
	uint32_t first_layer_index, uint32_t moe_layer_count)
{
	/* 33 every-layer kinds per MoE layer; the globals ride only on
	 * the slice that contains them, mirroring the common directory
	 * builder: embedding on first_layer_index==0, and final_norm +
	 * lm_head + the three output hc globals on the closing slice. */
	uint32_t tensors = moe_layer_count * 33u;
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + moe_layer_count ==
		SPARK_HY4_MODEL_LAYER_COUNT )
		tensors += 5u;
	return tensors;
}

static inline void SparkHy4StagePackExpectedGeometry(
	SparkHy4StagePackHeader *header, uint32_t moe_layer_count)
{
	header->magic = SPARK_HY4_STAGEPACK_MAGIC;
	header->format_version = SPARK_HY4_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_HY4_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_HY4_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkHy4StagePackExpectedTensorCount(0u,
		moe_layer_count);
	header->hidden_dimension = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
	header->layer_count = SPARK_HY4_MODEL_LAYER_COUNT;
	header->first_layer_index = 0u;
	header->total_layer_count = SPARK_HY4_MODEL_LAYER_COUNT;
	header->attention_period = SPARK_HY4_MODEL_INDEXER_FULL_PERIOD;
	header->full_attention_phase = 0u;
	header->gdn_key_head_count = 0u;
	header->gdn_value_head_count = 0u;
	header->gdn_head_key_dimension = 0u;
	header->gdn_head_value_dimension = 0u;
	header->gdn_conv_kernel = 0u;
	header->attn_query_head_count = SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK;
	header->attn_kv_head_count = SPARK_HY4_MODEL_ATTN_KV_HEADS_PER_RANK;
	header->attn_head_dimension = SPARK_HY4_MODEL_QK_HEAD_DIMENSION;
	header->attn_rope_dimension = SPARK_HY4_MODEL_QK_ROPE_HEAD_DIMENSION;
	header->routed_expert_count = SPARK_HY4_MODEL_EXPERTS_PER_RANK;
	header->experts_per_token = SPARK_HY4_MODEL_EXPERTS_PER_TOKEN;
	header->expert_intermediate_dimension =
		SPARK_HY4_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
	header->output_vocab_count = SPARK_HY4_MODEL_VOCAB_PER_RANK;
	header->mxfp4_group_size =
		SPARK_HY4_MODEL_EXPERT_SCALE_GROUP_SIZE;
	header->mtp_layer_count = SPARK_HY4_MODEL_MTP_LAYER_COUNT;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

SPARK_STAGEPACK_HEADER_LAYOUT_PROOF(SparkHy4StagePackHeader);

static inline int32_t SparkHy4StagePackHeaderMatches(
	const SparkHy4StagePackHeader *file_header,
	const SparkHy4StagePackHeader *expected)
{
	return SparkStagePackHeaderMatches(
		(const SparkStagePackHeaderCommon *)file_header,
		(const SparkStagePackHeaderCommon *)expected);
}
