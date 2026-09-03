#pragma once

#include <stdint.h>

#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stagepack_format.h"
#include "sparkpipe/spark_status.h"


#define SPARK_QWEN38_MAX_STAGEPACK_MAGIC 0x50533851u
#define SPARK_QWEN38_MAX_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_QWEN38_MAX_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_QWEN38_MAX_STAGEPACK_MTP_LAYER (UINT32_MAX - 1u)
#define SPARK_QWEN38_MAX_STAGEPACK_PAYLOAD_ALIGNMENT 256u

typedef enum SparkQwen38MaxStagePackTensorKind
{
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_QUERY = 22,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY = 23,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_VALUE = 24,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_OUTPUT = 25,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_QUERY_NORM = 26,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY_NORM = 27,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_FC = 28,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_EMBED_NORM = 29,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_HIDDEN_NORM = 30,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_FINAL_NORM = 31,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_KIND_COUNT = 32
} SparkQwen38MaxStagePackTensorKind;

#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTENTION_NORM SPARK_STAGEPACK_TENSOR_ATTENTION_NORM
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MLP_NORM SPARK_STAGEPACK_TENSOR_MLP_NORM
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_GATE SPARK_STAGEPACK_TENSOR_MOE_GATE
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W1 SPARK_STAGEPACK_TENSOR_MOE_W1
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W3 SPARK_STAGEPACK_TENSOR_MOE_W3
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_DOWN SPARK_STAGEPACK_TENSOR_MOE_DOWN
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_GATE SPARK_STAGEPACK_TENSOR_MOE_SHARED_GATE
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_UP SPARK_STAGEPACK_TENSOR_MOE_SHARED_UP
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_DOWN SPARK_STAGEPACK_TENSOR_MOE_SHARED_DOWN
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT SPARK_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_QKV SPARK_STAGEPACK_TENSOR_GDN_QKV
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_GATE SPARK_STAGEPACK_TENSOR_GDN_GATE
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_BETA SPARK_STAGEPACK_TENSOR_GDN_BETA
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_DECAY SPARK_STAGEPACK_TENSOR_GDN_DECAY
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_OUTPUT SPARK_STAGEPACK_TENSOR_GDN_OUTPUT
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_CONV_WEIGHT SPARK_STAGEPACK_TENSOR_GDN_CONV_WEIGHT
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_A_LOG SPARK_STAGEPACK_TENSOR_GDN_A_LOG
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_DT_BIAS SPARK_STAGEPACK_TENSOR_GDN_DT_BIAS
#define SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_NORM SPARK_STAGEPACK_TENSOR_GDN_NORM

#define SPARK_QWEN38_MAX_STAGEPACK_CLASS_GLOBAL SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL
#define SPARK_QWEN38_MAX_STAGEPACK_CLASS_EVERY_LAYER SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER
#define SPARK_QWEN38_MAX_STAGEPACK_CLASS_GDN_LAYER SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GDN_LAYER
#define SPARK_QWEN38_MAX_STAGEPACK_CLASS_ATTN_LAYER SPARK_STAGEPACK_FORMAT_LAYER_CLASS_ATTN_LAYER

typedef struct SparkQwen38MaxStagePackHeader
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
} SparkQwen38MaxStagePackHeader;

typedef struct SparkQwen38MaxStagePackEntry
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
} SparkQwen38MaxStagePackEntry;

#define SPARK_QWEN38_MAX_STAGEPACK_HEADER_BYTES 120u
#define SPARK_QWEN38_MAX_STAGEPACK_ENTRY_BYTES 56u
_Static_assert(sizeof(SparkQwen38MaxStagePackHeader) == SPARK_QWEN38_MAX_STAGEPACK_HEADER_BYTES,"qwen38 stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkQwen38MaxStagePackEntry) == SPARK_QWEN38_MAX_STAGEPACK_ENTRY_BYTES,"qwen38 stage pack directory entry must be 56 wire bytes");

_Static_assert(SPARK_QWEN38_MAX_MODEL_GDN_LAYER_COUNT + SPARK_QWEN38_MAX_MODEL_FULL_ATTENTION_LAYER_COUNT == SPARK_QWEN38_MAX_MODEL_LAYER_COUNT,"qwen38 layer split must cover the stack");
_Static_assert((SPARK_QWEN38_MAX_MODEL_LAYER_COUNT % SPARK_QWEN38_MAX_MODEL_ATTENTION_PERIOD) == 0u,"qwen38 layer count must be whole periods");
_Static_assert(SPARK_QWEN38_MAX_MODEL_GDN_LAYER_COUNT == (SPARK_QWEN38_MAX_MODEL_LAYER_COUNT / SPARK_QWEN38_MAX_MODEL_ATTENTION_PERIOD) * (SPARK_QWEN38_MAX_MODEL_ATTENTION_PERIOD - 1u),"qwen38 gdn count must match the 3:1 period");
_Static_assert((SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT % SPARK_QWEN38_MAX_MODEL_GDN_KEY_HEAD_COUNT) == 0u,"qwen38 value heads must group evenly onto key heads");
_Static_assert(SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD == 8u,"qwen38 grouped-value ratio is eight per config");
_Static_assert((SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT % SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT) == 0u,"qwen38 query heads must group evenly onto kv heads");
_Static_assert(SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION == SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION / 4u,"qwen38 rope covers a quarter of the head");
_Static_assert((SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION % 2u) == 0u,"qwen38 rope dimension must pair");
_Static_assert(SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS == 20480u,"qwen38 conv width is q+k+v concatenated");
_Static_assert(SPARK_QWEN38_MAX_MODEL_GDN_QK_DIMENSION == 2048u && SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION == 16384u,"qwen38 gdn projection widths per config");
_Static_assert(SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION == 16384u && SPARK_QWEN38_MAX_MODEL_ATTN_KV_DIMENSION == 1024u,"qwen38 attention projection widths per config");
_Static_assert(SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE == 32u,"qwen38 mxfp4 group size must be 32");
_Static_assert((SPARK_QWEN38_MAX_MODEL_EXPERT_INTERMEDIATE_DIMENSION % SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE) == 0u,"qwen38 expert intermediate must tile for mxfp4 groups");
_Static_assert((SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION % SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE) == 0u,"qwen38 hidden must tile for mxfp4 groups");

static inline uint32_t SparkQwen38MaxStagePackFullAttentionLayersBelow(uint32_t layer_count)
{
	return(layer_count / SPARK_QWEN38_MAX_MODEL_ATTENTION_PERIOD);
}

static inline uint32_t SparkQwen38MaxStagePackExpectedTensorCount(uint32_t first_layer_index, uint32_t layer_count)
{
	uint32_t full = SparkQwen38MaxStagePackFullAttentionLayersBelow(first_layer_index + layer_count) - SparkQwen38MaxStagePackFullAttentionLayersBelow(first_layer_index);
	uint32_t gdn = layer_count - full;
	uint32_t tensors = (layer_count * 10u) + (gdn * 9u) + (full * 6u);
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == SPARK_QWEN38_MAX_MODEL_LAYER_COUNT )
		tensors += 2u + 4u + 16u + (first_layer_index != 0u ? 1u : 0u);
	return(tensors);
}

static inline void SparkQwen38MaxStagePackExpectedGeometry(SparkQwen38MaxStagePackHeader *header, uint32_t first_layer_index, uint32_t layer_count)
{
	header->magic = SPARK_QWEN38_MAX_STAGEPACK_MAGIC;
	header->format_version = SPARK_QWEN38_MAX_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_QWEN38_MAX_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_QWEN38_MAX_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkQwen38MaxStagePackExpectedTensorCount(first_layer_index,layer_count);
	header->hidden_dimension = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
	header->layer_count = layer_count;
	header->first_layer_index = first_layer_index;
	header->total_layer_count = SPARK_QWEN38_MAX_MODEL_LAYER_COUNT;
	header->attention_period = SPARK_QWEN38_MAX_MODEL_ATTENTION_PERIOD;
	header->full_attention_phase = SPARK_QWEN38_MAX_MODEL_FULL_ATTENTION_PHASE;
	header->gdn_key_head_count = SPARK_QWEN38_MAX_MODEL_GDN_KEY_HEAD_COUNT;
	header->gdn_value_head_count = SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT;
	header->gdn_head_key_dimension = SPARK_QWEN38_MAX_MODEL_GDN_HEAD_KEY_DIMENSION;
	header->gdn_head_value_dimension = SPARK_QWEN38_MAX_MODEL_GDN_HEAD_VALUE_DIMENSION;
	header->gdn_conv_kernel = SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL;
	header->attn_query_head_count = SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT;
	header->attn_kv_head_count = SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT;
	header->attn_head_dimension = SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
	header->attn_rope_dimension = SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION;
	header->routed_expert_count = SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT;
	header->experts_per_token = SPARK_QWEN38_MAX_MODEL_EXPERTS_PER_TOKEN;
	header->expert_intermediate_dimension = SPARK_QWEN38_MAX_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
	header->output_vocab_count = SPARK_QWEN38_MAX_MODEL_OUTPUT_VOCAB_COUNT;
	header->mxfp4_group_size = SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE;
	header->mtp_layer_count = SPARK_QWEN38_MAX_MODEL_MTP_LAYER_COUNT;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

SPARK_STAGEPACK_HEADER_LAYOUT_PROOF(SparkQwen38MaxStagePackHeader);
static inline int32_t SparkQwen38MaxStagePackHeaderMatches(const SparkQwen38MaxStagePackHeader *file_header, const SparkQwen38MaxStagePackHeader *expected)
{
	return(SparkStagePackHeaderMatches(
		(const SparkStagePackHeaderCommon *)file_header,
		(const SparkStagePackHeaderCommon *)expected));
}

typedef SparkStagePackTensorShape SparkQwen38MaxStagePackTensorShape;

_Static_assert(SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 == SPARK_STAGEPACK_FORMAT_WEIGHT_BF16,"qwen38 bf16 weight code must match the shared format");
_Static_assert(SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 == SPARK_STAGEPACK_FORMAT_WEIGHT_F32,"qwen38 f32 weight code must match the shared format");
_Static_assert(SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 == SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128,"qwen38 fp8 weight code must match the shared format");

static const SparkStagePackGeometryTable SparkQwen38MaxStagePackGeometry =
{
	.norm_width = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION,
	.hidden_dimension = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION,
	.routed_expert_count = SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT,
	.expert_intermediate_dimension = SPARK_QWEN38_MAX_MODEL_EXPERT_INTERMEDIATE_DIMENSION,
	.gdn_conv_channels = SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS,
	.gdn_value_dimension = SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION,
	.gdn_value_head_count = SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT,
	.gdn_head_value_dimension = SPARK_QWEN38_MAX_MODEL_GDN_HEAD_VALUE_DIMENSION,
	.gdn_conv_kernel = SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL
};

static inline int32_t SparkQwen38MaxStagePackShapeGlobal(uint32_t tensor_kind, SparkQwen38MaxStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN38_MAX_STAGEPACK_CLASS_GLOBAL;
	switch ( tensor_kind )
	{
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows = SPARK_QWEN38_MAX_MODEL_OUTPUT_VOCAB_COUNT;
		shape->columns = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_FC:
		shape->rows = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		shape->columns = 2u * SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_EMBED_NORM:
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_HIDDEN_NORM:
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen38MaxStagePackShapeAttn(uint32_t tensor_kind, SparkQwen38MaxStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN38_MAX_STAGEPACK_CLASS_ATTN_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_QUERY:
		shape->rows = 2u * SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION;
		shape->columns = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY:
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_VALUE:
		shape->rows = SPARK_QWEN38_MAX_MODEL_ATTN_KV_DIMENSION;
		shape->columns = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->rows = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_QUERY_NORM:
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen38MaxStagePackTensorShapeOf(uint32_t tensor_kind, SparkQwen38MaxStagePackTensorShape *shape)
{
	SparkStagePackShapeInit(shape);
	if ( SparkQwen38MaxStagePackShapeGlobal(tensor_kind,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkStagePackShapeEveryLayerCommon(tensor_kind,&SparkQwen38MaxStagePackGeometry,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkStagePackShapeGdnCommon(tensor_kind,&SparkQwen38MaxStagePackGeometry,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkQwen38MaxStagePackShapeAttn(tensor_kind,shape) == 0 )
		return(0);
	return(-1);
}

static inline int32_t SparkQwen38MaxStagePackResolvedShape(uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, SparkQwen38MaxStagePackTensorShape *shape)
{
	if ( SparkQwen38MaxStagePackTensorShapeOf(tensor_kind,shape) < 0 )
		return(-1);
	if ( layer_index == SPARK_QWEN38_MAX_STAGEPACK_MTP_LAYER )
		return((is_global == 0u && (shape->layer_class == SPARK_QWEN38_MAX_STAGEPACK_CLASS_EVERY_LAYER || shape->layer_class == SPARK_QWEN38_MAX_STAGEPACK_CLASS_ATTN_LAYER)) ? 0 : -6);
	if ( (shape->layer_class == SPARK_QWEN38_MAX_STAGEPACK_CLASS_GLOBAL) != (is_global != 0u) )
		return(-2);
	if ( is_global != 0u )
		return(0);
	if ( layer_index >= SPARK_QWEN38_MAX_MODEL_LAYER_COUNT )
		return(-3);
	if ( shape->layer_class == SPARK_QWEN38_MAX_STAGEPACK_CLASS_GDN_LAYER && SPARK_QWEN38_MAX_MODEL_LAYER_IS_GDN(layer_index) == 0u )
		return(-4);
	if ( shape->layer_class == SPARK_QWEN38_MAX_STAGEPACK_CLASS_ATTN_LAYER && SPARK_QWEN38_MAX_MODEL_LAYER_IS_GDN(layer_index) != 0u )
		return(-5);
	return(0);
}

static inline uint64_t SparkQwen38MaxStagePackPayloadBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(elements / 2u);
	if ( weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(elements);
	if ( weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 || weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 )
		return(elements * 4u);
	return(elements * (uint64_t)SPARK_QWEN38_MAX_MODEL_BF16_ELEMENT_BYTES);
}

static inline uint64_t SparkQwen38MaxStagePackScaleBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	if ( weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(((uint64_t)rows * (uint64_t)columns) / SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE);
	if ( weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(((uint64_t)rows / 128u) * ((uint64_t)columns / 128u) * 4u);
	return(0u);
}
