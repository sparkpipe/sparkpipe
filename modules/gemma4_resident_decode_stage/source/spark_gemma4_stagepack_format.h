#pragma once

#include <stdint.h>

#include "sparkpipe/spark_gemma4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stagepack_format.h"
#include "sparkpipe/spark_status.h"


#define SPARK_GEMMA4_STAGEPACK_MAGIC 0x50533451u
#define SPARK_GEMMA4_STAGEPACK_FORMAT_VERSION 2u
#define SPARK_GEMMA4_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_GEMMA4_STAGEPACK_MTP_LAYER (UINT32_MAX - 1u)
#define SPARK_GEMMA4_STAGEPACK_PAYLOAD_ALIGNMENT 256u

typedef enum SparkGemma4StagePackTensorKind
{
	SPARK_GEMMA4_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_GEMMA4_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_GEMMA4_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_QUERY = 22,
	SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_KEY = 23,
	SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_VALUE = 24,
	SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_OUTPUT = 25,
	SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_QUERY_NORM = 26,
	SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_KEY_NORM = 27,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_FC = 28,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_EMBED_NORM = 29,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_HIDDEN_NORM = 30,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_FINAL_NORM = 31,
	SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_HC_DOWN = 32,
	SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_HC_UP = 33,
	SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_HC_INJECT = 34,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_HC_DOWN = 35,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_HC_UP = 36,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_HC_INJECT = 37,
	SPARK_GEMMA4_STAGEPACK_TENSOR_INDEXER_QK = 38,
	SPARK_GEMMA4_STAGEPACK_TENSOR_INDEXER_Q_NORM = 39,
	SPARK_GEMMA4_STAGEPACK_TENSOR_INDEXER_K_NORM = 40,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MIXER_DOWN = 41,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MIXER_UP = 42,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_MIXER_DOWN = 43,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_MIXER_UP = 44,
	SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_KEY = 45,
	SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_VALUE = 46,
	SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_NORM_KEY = 47,
	SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_NORM_QUERY = 48,
	SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_NORM_CONV = 49,
	SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_CONV = 50,
	SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_MULTIPLIERS = 51,
	SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_HEAD_VOCABS = 52,
	SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_HEAD_OFFSETS = 53,
	SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_NGRAM = 54,
	SPARK_GEMMA4_STAGEPACK_TENSOR_KIND_COUNT = 55
} SparkGemma4StagePackTensorKind;

#define SPARK_GEMMA4_STAGEPACK_TENSOR_ATTENTION_NORM SPARK_STAGEPACK_TENSOR_ATTENTION_NORM
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_NORM SPARK_STAGEPACK_TENSOR_MLP_NORM
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_GATE SPARK_STAGEPACK_TENSOR_MOE_GATE
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_W1 SPARK_STAGEPACK_TENSOR_MOE_W1
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_W3 SPARK_STAGEPACK_TENSOR_MOE_W3
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_DOWN SPARK_STAGEPACK_TENSOR_MOE_DOWN
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_SHARED_GATE SPARK_STAGEPACK_TENSOR_MOE_SHARED_GATE
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_SHARED_UP SPARK_STAGEPACK_TENSOR_MOE_SHARED_UP
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_SHARED_DOWN SPARK_STAGEPACK_TENSOR_MOE_SHARED_DOWN
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT SPARK_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT
#define SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_QKV SPARK_STAGEPACK_TENSOR_GDN_QKV
#define SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_GATE SPARK_STAGEPACK_TENSOR_GDN_GATE
#define SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_BETA SPARK_STAGEPACK_TENSOR_GDN_BETA
#define SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_DECAY SPARK_STAGEPACK_TENSOR_GDN_DECAY
#define SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_OUTPUT SPARK_STAGEPACK_TENSOR_GDN_OUTPUT
#define SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_CONV_WEIGHT SPARK_STAGEPACK_TENSOR_GDN_CONV_WEIGHT
#define SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_A_LOG SPARK_STAGEPACK_TENSOR_GDN_A_LOG
#define SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_DT_BIAS SPARK_STAGEPACK_TENSOR_GDN_DT_BIAS
#define SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_NORM SPARK_STAGEPACK_TENSOR_GDN_NORM

#define SPARK_GEMMA4_STAGEPACK_CLASS_GLOBAL SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL
#define SPARK_GEMMA4_STAGEPACK_CLASS_EVERY_LAYER SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER
#define SPARK_GEMMA4_STAGEPACK_CLASS_GDN_LAYER SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GDN_LAYER
#define SPARK_GEMMA4_STAGEPACK_CLASS_ATTN_LAYER SPARK_STAGEPACK_FORMAT_LAYER_CLASS_ATTN_LAYER
#define SPARK_GEMMA4_STAGEPACK_CLASS_PLE_LAYER 4u

typedef struct SparkGemma4StagePackHeader
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
} SparkGemma4StagePackHeader;

typedef struct SparkGemma4StagePackEntry
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
} SparkGemma4StagePackEntry;

#define SPARK_GEMMA4_STAGEPACK_HEADER_BYTES 120u
#define SPARK_GEMMA4_STAGEPACK_ENTRY_BYTES 56u
_Static_assert(sizeof(SparkGemma4StagePackHeader) == SPARK_GEMMA4_STAGEPACK_HEADER_BYTES,"gemma4 stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkGemma4StagePackEntry) == SPARK_GEMMA4_STAGEPACK_ENTRY_BYTES,"gemma4 stage pack directory entry must be 56 wire bytes");

_Static_assert(SPARK_GEMMA4_MODEL_GDN_LAYER_COUNT + SPARK_GEMMA4_MODEL_FULL_ATTENTION_LAYER_COUNT == SPARK_GEMMA4_MODEL_LAYER_COUNT,"gemma4 layer split must cover the stack");
_Static_assert((SPARK_GEMMA4_MODEL_LAYER_COUNT % SPARK_GEMMA4_MODEL_ATTENTION_PERIOD) == 0u,"gemma4 layer count must be whole periods");
_Static_assert(SPARK_GEMMA4_MODEL_GDN_LAYER_COUNT == (SPARK_GEMMA4_MODEL_LAYER_COUNT / SPARK_GEMMA4_MODEL_ATTENTION_PERIOD) * (SPARK_GEMMA4_MODEL_ATTENTION_PERIOD - 1u),"gemma4 gdn count must match the 3:1 period");
_Static_assert((SPARK_GEMMA4_MODEL_GDN_VALUE_HEAD_COUNT % SPARK_GEMMA4_MODEL_GDN_KEY_HEAD_COUNT) == 0u,"gemma4 value heads must group evenly onto key heads");
_Static_assert(SPARK_GEMMA4_MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD == 3u,"gemma4 grouped-value ratio is three per config (48 value heads over 16 key heads)");
_Static_assert((SPARK_GEMMA4_MODEL_ATTN_QUERY_HEAD_COUNT % SPARK_GEMMA4_MODEL_ATTN_KV_HEAD_COUNT) == 0u,"gemma4 query heads must group evenly onto kv heads");
_Static_assert(SPARK_GEMMA4_MODEL_ATTN_ROPE_DIMENSION == SPARK_GEMMA4_MODEL_ATTN_HEAD_DIMENSION / 4u,"gemma4 rope covers a quarter of the head");
_Static_assert((SPARK_GEMMA4_MODEL_ATTN_ROPE_DIMENSION % 2u) == 0u,"gemma4 rope dimension must pair");
_Static_assert(SPARK_GEMMA4_MODEL_GDN_CONV_CHANNELS == 10240u,"gemma4 conv width is q+k+v concatenated");
_Static_assert(SPARK_GEMMA4_MODEL_GDN_QK_DIMENSION == 2048u && SPARK_GEMMA4_MODEL_GDN_VALUE_DIMENSION == 6144u,"gemma4 gdn projection widths per config");
_Static_assert(SPARK_GEMMA4_MODEL_ATTN_QUERY_DIMENSION == 6144u && SPARK_GEMMA4_MODEL_ATTN_KV_DIMENSION == 512u,"gemma4 attention projection widths per config");
_Static_assert(SPARK_GEMMA4_MODEL_MXFP4_GROUP_SIZE == 32u,"gemma4 mxfp4 group size must be 32");
_Static_assert((SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION % SPARK_GEMMA4_MODEL_MXFP4_GROUP_SIZE) == 0u,"gemma4 expert intermediate must tile for mxfp4 groups");
_Static_assert((SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION % SPARK_GEMMA4_MODEL_MXFP4_GROUP_SIZE) == 0u,"gemma4 hidden must tile for mxfp4 groups");

static inline uint32_t SparkGemma4StagePackFullAttentionLayersBelow(uint32_t layer_count)
{
	return(layer_count / SPARK_GEMMA4_MODEL_ATTENTION_PERIOD);
}

static inline uint32_t SparkGemma4StagePackExpectedTensorCount(uint32_t first_layer_index, uint32_t layer_count)
{
	uint32_t full = SparkGemma4StagePackFullAttentionLayersBelow(first_layer_index + layer_count) - SparkGemma4StagePackFullAttentionLayersBelow(first_layer_index);
	uint32_t gdn = layer_count - full;
	uint32_t tensors = (layer_count * 16u) + (gdn * 9u) + (full * 9u);
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == SPARK_GEMMA4_MODEL_LAYER_COUNT )
		tensors += 2u + 4u + 4u + 25u + (first_layer_index != 0u ? 1u : 0u);
	return(tensors);
}

static inline uint32_t SparkGemma4StagePackExpectedTensorCountWithPle(uint32_t first_layer_index, uint32_t layer_count, uint32_t include_ple)
{
	uint32_t tensors = SparkGemma4StagePackExpectedTensorCount(first_layer_index,layer_count);
	if ( include_ple != 0u && first_layer_index <= SPARK_GEMMA4_MODEL_PLE_LAYER_INDEX && first_layer_index + layer_count > SPARK_GEMMA4_MODEL_PLE_LAYER_INDEX )
		tensors += 10u;
	return(tensors);
}

static inline void SparkGemma4StagePackExpectedGeometry(SparkGemma4StagePackHeader *header, uint32_t first_layer_index, uint32_t layer_count, uint32_t include_ple)
{
	header->magic = SPARK_GEMMA4_STAGEPACK_MAGIC;
	header->format_version = SPARK_GEMMA4_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_GEMMA4_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_GEMMA4_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkGemma4StagePackExpectedTensorCountWithPle(first_layer_index,layer_count,include_ple);
	header->hidden_dimension = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
	header->layer_count = layer_count;
	header->first_layer_index = first_layer_index;
	header->total_layer_count = SPARK_GEMMA4_MODEL_LAYER_COUNT;
	header->attention_period = SPARK_GEMMA4_MODEL_ATTENTION_PERIOD;
	header->full_attention_phase = SPARK_GEMMA4_MODEL_FULL_ATTENTION_PHASE;
	header->gdn_key_head_count = SPARK_GEMMA4_MODEL_GDN_KEY_HEAD_COUNT;
	header->gdn_value_head_count = SPARK_GEMMA4_MODEL_GDN_VALUE_HEAD_COUNT;
	header->gdn_head_key_dimension = SPARK_GEMMA4_MODEL_GDN_HEAD_KEY_DIMENSION;
	header->gdn_head_value_dimension = SPARK_GEMMA4_MODEL_GDN_HEAD_VALUE_DIMENSION;
	header->gdn_conv_kernel = SPARK_GEMMA4_MODEL_GDN_CONV_KERNEL;
	header->attn_query_head_count = SPARK_GEMMA4_MODEL_ATTN_QUERY_HEAD_COUNT;
	header->attn_kv_head_count = SPARK_GEMMA4_MODEL_ATTN_KV_HEAD_COUNT;
	header->attn_head_dimension = SPARK_GEMMA4_MODEL_ATTN_HEAD_DIMENSION;
	header->attn_rope_dimension = SPARK_GEMMA4_MODEL_ATTN_ROPE_DIMENSION;
	header->routed_expert_count = SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT;
	header->experts_per_token = SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN;
	header->expert_intermediate_dimension = SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
	header->output_vocab_count = SPARK_GEMMA4_MODEL_OUTPUT_VOCAB_COUNT;
	header->mxfp4_group_size = SPARK_GEMMA4_MODEL_MXFP4_GROUP_SIZE;
	header->mtp_layer_count = SPARK_GEMMA4_MODEL_MTP_LAYER_COUNT;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

SPARK_STAGEPACK_HEADER_LAYOUT_PROOF(SparkGemma4StagePackHeader);
static inline int32_t SparkGemma4StagePackHeaderMatches(const SparkGemma4StagePackHeader *file_header, const SparkGemma4StagePackHeader *expected)
{
	return(SparkStagePackHeaderMatches(
		(const SparkStagePackHeaderCommon *)file_header,
		(const SparkStagePackHeaderCommon *)expected));
}

typedef SparkStagePackTensorShape SparkGemma4StagePackTensorShape;

_Static_assert(SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 == SPARK_STAGEPACK_FORMAT_WEIGHT_BF16,"qwen4 bf16 weight code must match the shared format");
_Static_assert(SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 == SPARK_STAGEPACK_FORMAT_WEIGHT_F32,"qwen4 f32 weight code must match the shared format");
_Static_assert(SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 == SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128,"qwen4 fp8 weight code must match the shared format");
_Static_assert(SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64 == SPARK_STAGEPACK_FORMAT_WEIGHT_I64,"qwen4 i64 weight code must match the shared format");
_Static_assert(SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED == SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED,"qwen4 nvfp4 weight code must match the shared format");

static const SparkStagePackGeometryTable SparkGemma4StagePackGeometry =
{
	.norm_width = SPARK_GEMMA4_MODEL_HC_STREAM_WIDTH,
	.hidden_dimension = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION,
	.routed_expert_count = SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT,
	.expert_intermediate_dimension = SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,
	.gdn_conv_channels = SPARK_GEMMA4_MODEL_GDN_CONV_CHANNELS,
	.gdn_value_dimension = SPARK_GEMMA4_MODEL_GDN_VALUE_DIMENSION,
	.gdn_value_head_count = SPARK_GEMMA4_MODEL_GDN_VALUE_HEAD_COUNT,
	.gdn_head_value_dimension = SPARK_GEMMA4_MODEL_GDN_HEAD_VALUE_DIMENSION,
	.gdn_conv_kernel = SPARK_GEMMA4_MODEL_GDN_CONV_KERNEL
};

static inline int32_t SparkGemma4StagePackShapeGlobal(uint32_t tensor_kind, SparkGemma4StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_GEMMA4_STAGEPACK_CLASS_GLOBAL;
	switch ( tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows = SPARK_GEMMA4_MODEL_OUTPUT_VOCAB_COUNT;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_HC_STREAM_WIDTH;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_FC:
		shape->rows = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		shape->columns = 2u * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MIXER_DOWN:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_MIXER_DOWN:
		shape->rows = SPARK_GEMMA4_MODEL_HC_LOWRANK_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_HC_STREAM_WIDTH;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MIXER_UP:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_MIXER_UP:
		shape->rows = SPARK_GEMMA4_MODEL_HC_STREAM_WIDTH;
		shape->columns = SPARK_GEMMA4_MODEL_HC_LOWRANK_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_EMBED_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_HIDDEN_NORM:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_HC_STREAM_WIDTH;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkGemma4StagePackShapeEveryLayer(uint32_t tensor_kind, SparkGemma4StagePackTensorShape *shape)
{
	if ( SparkStagePackShapeEveryLayerCommon(tensor_kind,
		&SparkGemma4StagePackGeometry,shape) == 0 )
		return(0);
	shape->layer_class = SPARK_GEMMA4_STAGEPACK_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_HC_DOWN:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_HC_DOWN:
		shape->rows = SPARK_GEMMA4_MODEL_HC_LOWRANK_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_HC_STREAM_WIDTH;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_HC_UP:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_HC_UP:
		shape->rows = SPARK_GEMMA4_MODEL_HC_STREAM_WIDTH;
		shape->columns = SPARK_GEMMA4_MODEL_HC_LOWRANK_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_HC_INJECT:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_HC_INJECT:
		shape->rows = SPARK_GEMMA4_MODEL_HC_STREAM_COUNT;
		shape->columns = SPARK_GEMMA4_MODEL_HC_STREAM_WIDTH;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkGemma4StagePackShapeAttn(uint32_t tensor_kind, SparkGemma4StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_GEMMA4_STAGEPACK_CLASS_ATTN_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_QUERY:
		shape->rows = 2u * SPARK_GEMMA4_MODEL_ATTN_QUERY_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_KEY:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_VALUE:
		shape->rows = SPARK_GEMMA4_MODEL_ATTN_KV_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->rows = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_ATTN_QUERY_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_QUERY_NORM:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_ATTN_HEAD_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_INDEXER_QK:
		shape->rows = (SPARK_GEMMA4_MODEL_INDEXER_HEAD_COUNT + SPARK_GEMMA4_MODEL_INDEXER_KV_HEAD_COUNT) * SPARK_GEMMA4_MODEL_INDEXER_HEAD_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_INDEXER_Q_NORM:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_INDEXER_K_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_INDEXER_HEAD_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkGemma4StagePackShapePle(uint32_t tensor_kind, SparkGemma4StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_GEMMA4_STAGEPACK_CLASS_PLE_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_KEY:
		shape->rows = SPARK_GEMMA4_MODEL_HC_STREAM_WIDTH;
		shape->columns = SPARK_GEMMA4_MODEL_PLE_EMBED_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_VALUE:
		shape->rows = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_PLE_EMBED_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_NORM_KEY:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_NORM_QUERY:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_NORM_CONV:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_HC_STREAM_WIDTH;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_CONV:
		shape->rows = SPARK_GEMMA4_MODEL_HC_STREAM_WIDTH;
		shape->columns = SPARK_GEMMA4_MODEL_PLE_CONV_KERNEL;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_MULTIPLIERS:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_PLE_NGRAM_SIZE;
		shape->natural_format = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_HEAD_VOCABS:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_HEAD_OFFSETS:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_PLE_NGRAM_HEAD_COUNT;
		shape->natural_format = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_NGRAM:
		shape->rows = SPARK_GEMMA4_MODEL_PLE_NGRAM_ROW_COUNT;
		shape->columns = SPARK_GEMMA4_MODEL_PLE_NGRAM_HEAD_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkGemma4StagePackTensorShapeOf(uint32_t tensor_kind, SparkGemma4StagePackTensorShape *shape)
{
	SparkStagePackShapeInit(shape);
	if ( SparkGemma4StagePackShapeGlobal(tensor_kind,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkGemma4StagePackShapeEveryLayer(tensor_kind,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkStagePackShapeGdnCommon(tensor_kind,&SparkGemma4StagePackGeometry,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkGemma4StagePackShapeAttn(tensor_kind,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkGemma4StagePackShapePle(tensor_kind,shape) == 0 )
		return(0);
	return(-1);
}

static inline void SparkGemma4StagePackNarrowShape(SparkGemma4StagePackTensorShape *shape, uint32_t tensor_kind, uint32_t tp_degree, uint32_t tp_rank)
{
	uint32_t key_heads, value_heads, experts;
	if ( tp_degree <= 1u )
		return;
	key_heads = SPARK_GEMMA4_MODEL_GDN_KEY_HEAD_COUNT / tp_degree;
	value_heads = SPARK_GEMMA4_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree;
	experts = SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT / tp_degree;
	(void)tp_rank;
	switch ( tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_QUERY:
		shape->rows = (SPARK_GEMMA4_MODEL_ATTN_QUERY_HEAD_COUNT / tp_degree) * 2u * SPARK_GEMMA4_MODEL_ATTN_HEAD_DIMENSION;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_KEY:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_VALUE:
		if ( (SPARK_GEMMA4_MODEL_ATTN_KV_HEAD_COUNT % tp_degree) != 0u )
			break;
		shape->rows = (SPARK_GEMMA4_MODEL_ATTN_KV_HEAD_COUNT / tp_degree) * SPARK_GEMMA4_MODEL_ATTN_HEAD_DIMENSION;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->columns = SPARK_GEMMA4_MODEL_ATTN_QUERY_DIMENSION / tp_degree;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_QKV:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_CONV_WEIGHT:
		shape->rows = (2u * (key_heads * SPARK_GEMMA4_MODEL_GDN_HEAD_KEY_DIMENSION))
			+ (value_heads * SPARK_GEMMA4_MODEL_GDN_HEAD_VALUE_DIMENSION);
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_GATE:
		shape->rows = SPARK_GEMMA4_MODEL_GDN_VALUE_DIMENSION / tp_degree;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_BETA:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_DECAY:
		shape->rows = value_heads;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_A_LOG:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_DT_BIAS:
		shape->columns = value_heads;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_GDN_OUTPUT:
		shape->columns = SPARK_GEMMA4_MODEL_GDN_VALUE_DIMENSION / tp_degree;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_GATE:
		shape->rows = experts;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_W1:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_W3:
		shape->rows = experts * SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_DOWN:
		shape->rows = experts * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_SHARED_GATE:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_SHARED_UP:
		shape->rows = SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION / tp_degree;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MOE_SHARED_DOWN:
		shape->columns = SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION / tp_degree;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows = SPARK_GEMMA4_MODEL_OUTPUT_VOCAB_COUNT / tp_degree;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PLE_NGRAM:
		shape->rows = SPARK_GEMMA4_MODEL_PLE_NGRAM_ROW_COUNT / tp_degree;
		break;
	default:
		break;
	}
}

static inline int32_t SparkGemma4StagePackResolvedShape(uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, SparkGemma4StagePackTensorShape *shape)
{
	if ( SparkGemma4StagePackTensorShapeOf(tensor_kind,shape) < 0 )
		return(-1);
	if ( layer_index == SPARK_GEMMA4_STAGEPACK_MTP_LAYER )
		return((is_global == 0u && (shape->layer_class == SPARK_GEMMA4_STAGEPACK_CLASS_EVERY_LAYER || shape->layer_class == SPARK_GEMMA4_STAGEPACK_CLASS_ATTN_LAYER)) ? 0 : -6);
	if ( (shape->layer_class == SPARK_GEMMA4_STAGEPACK_CLASS_GLOBAL) != (is_global != 0u) )
		return(-2);
	if ( is_global != 0u )
		return(0);
	if ( layer_index >= SPARK_GEMMA4_MODEL_LAYER_COUNT )
		return(-3);
	if ( shape->layer_class == SPARK_GEMMA4_STAGEPACK_CLASS_GDN_LAYER && SPARK_GEMMA4_MODEL_LAYER_IS_GDN(layer_index) == 0u )
		return(-4);
	if ( shape->layer_class == SPARK_GEMMA4_STAGEPACK_CLASS_ATTN_LAYER && SPARK_GEMMA4_MODEL_LAYER_IS_GDN(layer_index) != 0u )
		return(-5);
	if ( shape->layer_class == SPARK_GEMMA4_STAGEPACK_CLASS_PLE_LAYER && layer_index != SPARK_GEMMA4_MODEL_PLE_LAYER_INDEX )
		return(-7);
	return(0);
}

static inline uint64_t SparkGemma4StagePackPayloadBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 ||
		weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED )
		return(elements / 2u);
	if ( weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 ||
		weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
		return(elements);
	if ( weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 || weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 )
		return(elements * 4u);
	if ( weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64 )
		return(elements * 8u);
	return(elements * (uint64_t)SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES);
}

static inline uint64_t SparkGemma4StagePackScaleBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	if ( weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(((uint64_t)rows * (uint64_t)columns) / SPARK_GEMMA4_MODEL_MXFP4_GROUP_SIZE);
	if ( weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED )
	{
		/* per-expert segment: rows x columns/16 e4m3 plane + the F32
		 * input_scale + F32 weight_scale_2 globals. The expert count
		 * derives from the fused geometry: gate/up carry columns ==
		 * hidden and rows = experts x intermediate; down carries
		 * columns == intermediate and rows = experts x hidden. */
		uint64_t plane = (uint64_t)rows * ((uint64_t)columns / 16u);
		uint64_t per_expert,experts;
		if ( columns == SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION )
			per_expert = SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
		else if ( columns == SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION )
			per_expert = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		else
			return(0u);
		if ( per_expert == 0u || (rows % per_expert) != 0u )
			return(0u);
		experts = (uint64_t)rows / per_expert;
		return(plane + experts * 8u);
	}
	if ( weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(((uint64_t)rows / 128u) * ((uint64_t)columns / 128u) * 4u);
	if ( weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
		return((uint64_t)rows * ((uint64_t)columns / 128u));
	return(0u);
}
