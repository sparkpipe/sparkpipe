#pragma once

#include <stdint.h>

#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stagepack_format.h"
#include "sparkpipe/spark_status.h"


#define SPARK_QWEN38_27B_STAGEPACK_MAGIC 0x50533651u
#define SPARK_QWEN38_27B_STAGEPACK_FORMAT_VERSION 3u
#define SPARK_QWEN38_27B_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_QWEN38_27B_STAGEPACK_MTP_LAYER (UINT32_MAX - 1u)
#define SPARK_QWEN38_27B_STAGEPACK_PAYLOAD_ALIGNMENT 256u

typedef enum SparkQwen38_27bStagePackTensorKind
{
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTENTION_NORM = 3,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_MLP_NORM = 4,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE = 5,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_UP = 6,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_DOWN = 7,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_QKV = 8,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_GATE = 9,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_BETA = 10,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_DECAY = 11,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_OUTPUT = 12,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_CONV_WEIGHT = 13,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_A_LOG = 14,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_DT_BIAS = 15,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_NORM = 16,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY = 17,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY = 18,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_VALUE = 19,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_OUTPUT = 20,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY_NORM = 21,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY_NORM = 22,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FC = 23,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_EMBED_NORM = 24,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_HIDDEN_NORM = 25,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FINAL_NORM = 26,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_KIND_COUNT = 27
} SparkQwen38_27bStagePackTensorKind;

#define SPARK_QWEN38_27B_STAGEPACK_CLASS_GLOBAL SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL
#define SPARK_QWEN38_27B_STAGEPACK_CLASS_EVERY_LAYER SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER
#define SPARK_QWEN38_27B_STAGEPACK_CLASS_GDN_LAYER SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GDN_LAYER
#define SPARK_QWEN38_27B_STAGEPACK_CLASS_ATTN_LAYER SPARK_STAGEPACK_FORMAT_LAYER_CLASS_ATTN_LAYER

typedef struct SparkQwen38_27bStagePackHeader
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
	uint32_t ffn_intermediate_dimension;
	uint32_t output_vocab_count;
	uint32_t mxfp4_group_size;
	uint32_t mtp_layer_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint64_t directory_offset;
	uint64_t file_bytes;
} SparkQwen38_27bStagePackHeader;

typedef struct SparkQwen38_27bStagePackEntry
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
} SparkQwen38_27bStagePackEntry;

#define SPARK_QWEN38_27B_STAGEPACK_HEADER_BYTES 120u
#define SPARK_QWEN38_27B_STAGEPACK_ENTRY_BYTES 56u
_Static_assert(sizeof(SparkQwen38_27bStagePackHeader) == SPARK_QWEN38_27B_STAGEPACK_HEADER_BYTES,"qwen38_27b stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkQwen38_27bStagePackEntry) == SPARK_QWEN38_27B_STAGEPACK_ENTRY_BYTES,"qwen38_27b stage pack directory entry must be 56 wire bytes");

_Static_assert(SPARK_QWEN38_27B_MODEL_GDN_LAYER_COUNT + SPARK_QWEN38_27B_MODEL_FULL_ATTENTION_LAYER_COUNT == SPARK_QWEN38_27B_MODEL_LAYER_COUNT,"qwen38_27b layer split must cover the stack");
_Static_assert((SPARK_QWEN38_27B_MODEL_LAYER_COUNT % SPARK_QWEN38_27B_MODEL_ATTENTION_PERIOD) == 0u,"qwen38_27b layer count must be whole periods");
_Static_assert(SPARK_QWEN38_27B_MODEL_GDN_LAYER_COUNT == (SPARK_QWEN38_27B_MODEL_LAYER_COUNT / SPARK_QWEN38_27B_MODEL_ATTENTION_PERIOD) * (SPARK_QWEN38_27B_MODEL_ATTENTION_PERIOD - 1u),"qwen38_27b gdn count must match the 3:1 period");
_Static_assert((SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT % SPARK_QWEN38_27B_MODEL_GDN_KEY_HEAD_COUNT) == 0u,"qwen38_27b value heads must group evenly onto key heads");
_Static_assert(SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD == 3u,"qwen38_27b grouped-value ratio is three per config");
_Static_assert((SPARK_QWEN38_27B_MODEL_ATTN_QUERY_HEAD_COUNT % SPARK_QWEN38_27B_MODEL_ATTN_KV_HEAD_COUNT) == 0u,"qwen38_27b query heads must group evenly onto kv heads");
_Static_assert(SPARK_QWEN38_27B_MODEL_ATTN_ROPE_DIMENSION == SPARK_QWEN38_27B_MODEL_ATTN_HEAD_DIMENSION / 4u,"qwen38_27b rope covers a quarter of the head");
_Static_assert((SPARK_QWEN38_27B_MODEL_ATTN_ROPE_DIMENSION % 2u) == 0u,"qwen38_27b rope dimension must pair");
_Static_assert(SPARK_QWEN38_27B_MODEL_GDN_CONV_CHANNELS == 10240u,"qwen38_27b conv width is q+k+v concatenated");
_Static_assert(SPARK_QWEN38_27B_MODEL_GDN_QK_DIMENSION == 2048u && SPARK_QWEN38_27B_MODEL_GDN_VALUE_DIMENSION == 6144u,"qwen38_27b gdn projection widths per config");
_Static_assert(SPARK_QWEN38_27B_MODEL_ATTN_QUERY_DIMENSION == 6144u && SPARK_QWEN38_27B_MODEL_ATTN_KV_DIMENSION == 1024u,"qwen38_27b attention projection widths per config");
_Static_assert((SPARK_QWEN38_27B_MODEL_GDN_CHUNK_TOKENS % 16u) == 0u,"qwen38_27b chunk must tile for wmma");

static inline uint32_t SparkQwen38_27bStagePackFullAttentionLayersBelow(uint32_t layer_count)
{
	return(layer_count / SPARK_QWEN38_27B_MODEL_ATTENTION_PERIOD);
}

static inline uint32_t SparkQwen38_27bStagePackExpectedTensorCount(uint32_t first_layer_index, uint32_t layer_count)
{
	uint32_t full = SparkQwen38_27bStagePackFullAttentionLayersBelow(first_layer_index + layer_count) - SparkQwen38_27bStagePackFullAttentionLayersBelow(first_layer_index);
	uint32_t gdn = layer_count - full;
	uint32_t tensors = (layer_count * 5u) + (gdn * 9u) + (full * 6u);
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == SPARK_QWEN38_27B_MODEL_LAYER_COUNT )
		tensors += 2u + 4u + 11u + (first_layer_index != 0u ? 1u : 0u);
	return(tensors);
}

static inline void SparkQwen38_27bStagePackExpectedGeometry(SparkQwen38_27bStagePackHeader *header, uint32_t first_layer_index, uint32_t layer_count)
{
	header->magic = SPARK_QWEN38_27B_STAGEPACK_MAGIC;
	header->format_version = SPARK_QWEN38_27B_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_QWEN38_27B_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_QWEN38_27B_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkQwen38_27bStagePackExpectedTensorCount(first_layer_index,layer_count);
	header->hidden_dimension = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
	header->layer_count = layer_count;
	header->first_layer_index = first_layer_index;
	header->total_layer_count = SPARK_QWEN38_27B_MODEL_LAYER_COUNT;
	header->attention_period = SPARK_QWEN38_27B_MODEL_ATTENTION_PERIOD;
	header->full_attention_phase = SPARK_QWEN38_27B_MODEL_FULL_ATTENTION_PHASE;
	header->gdn_key_head_count = SPARK_QWEN38_27B_MODEL_GDN_KEY_HEAD_COUNT;
	header->gdn_value_head_count = SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT;
	header->gdn_head_key_dimension = SPARK_QWEN38_27B_MODEL_GDN_HEAD_KEY_DIMENSION;
	header->gdn_head_value_dimension = SPARK_QWEN38_27B_MODEL_GDN_HEAD_VALUE_DIMENSION;
	header->gdn_conv_kernel = SPARK_QWEN38_27B_MODEL_GDN_CONV_KERNEL;
	header->attn_query_head_count = SPARK_QWEN38_27B_MODEL_ATTN_QUERY_HEAD_COUNT;
	header->attn_kv_head_count = SPARK_QWEN38_27B_MODEL_ATTN_KV_HEAD_COUNT;
	header->attn_head_dimension = SPARK_QWEN38_27B_MODEL_ATTN_HEAD_DIMENSION;
	header->attn_rope_dimension = SPARK_QWEN38_27B_MODEL_ATTN_ROPE_DIMENSION;
	header->ffn_intermediate_dimension = SPARK_QWEN38_27B_MODEL_FFN_INTERMEDIATE_DIMENSION;
	header->output_vocab_count = SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT;
	header->mxfp4_group_size = 32u;
	header->mtp_layer_count = SPARK_QWEN38_27B_MODEL_MTP_LAYER_COUNT;
	header->tp_degree = 1u;
	header->tp_rank = 0u;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

static inline int32_t SparkQwen38_27bStagePackCompareGeometry(const SparkQwen38_27bStagePackHeader *file_header, const SparkQwen38_27bStagePackHeader *expected)
{
	if ( file_header->magic != expected->magic )
		return(-1);
	if ( file_header->format_version != expected->format_version )
		return(-2);
	if ( file_header->header_bytes != expected->header_bytes )
		return(-3);
	if ( file_header->directory_entry_bytes != expected->directory_entry_bytes )
		return(-4);
	if ( file_header->tensor_count != expected->tensor_count )
		return(-5);
	if ( file_header->hidden_dimension != expected->hidden_dimension )
		return(-6);
	if ( file_header->layer_count != expected->layer_count )
		return(-7);
	if ( file_header->first_layer_index != expected->first_layer_index )
		return(-8);
	if ( file_header->total_layer_count != expected->total_layer_count )
		return(-9);
	if ( file_header->attention_period != expected->attention_period )
		return(-10);
	if ( file_header->full_attention_phase != expected->full_attention_phase )
		return(-11);
	if ( file_header->gdn_key_head_count != expected->gdn_key_head_count )
		return(-12);
	if ( file_header->gdn_value_head_count != expected->gdn_value_head_count )
		return(-13);
	if ( file_header->gdn_head_key_dimension != expected->gdn_head_key_dimension )
		return(-14);
	if ( file_header->gdn_head_value_dimension != expected->gdn_head_value_dimension )
		return(-15);
	if ( file_header->gdn_conv_kernel != expected->gdn_conv_kernel )
		return(-16);
	if ( file_header->attn_query_head_count != expected->attn_query_head_count )
		return(-17);
	if ( file_header->attn_kv_head_count != expected->attn_kv_head_count )
		return(-18);
	if ( file_header->attn_head_dimension != expected->attn_head_dimension )
		return(-19);
	if ( file_header->attn_rope_dimension != expected->attn_rope_dimension )
		return(-20);
	if ( file_header->ffn_intermediate_dimension != expected->ffn_intermediate_dimension )
		return(-21);
	if ( file_header->output_vocab_count != expected->output_vocab_count )
		return(-22);
	if ( file_header->mxfp4_group_size != expected->mxfp4_group_size )
		return(-23);
	if ( file_header->mtp_layer_count != expected->mtp_layer_count )
		return(-24);
	if ( file_header->tp_degree != expected->tp_degree )
		return(-25);
	if ( file_header->tp_rank != expected->tp_rank )
		return(-26);
	return(0);
}

static inline const char *SparkQwen38_27bStagePackGeometryFieldName(int32_t compare_result)
{
	static const char *names[25] =
	{
		"ok","magic","format_version","header_bytes","directory_entry_bytes",
		"tensor_count","hidden_dimension","layer_count","first_layer_index",
		"total_layer_count","attention_period","full_attention_phase",
		"gdn_key_head_count","gdn_value_head_count","gdn_head_key_dimension",
		"gdn_head_value_dimension","gdn_conv_kernel","attn_query_head_count",
		"attn_kv_head_count","attn_head_dimension","attn_rope_dimension",
		"ffn_intermediate_dimension","output_vocab_count","mxfp4_group_size",
		"mtp_layer_count"
	};
	uint32_t index = (uint32_t)(-compare_result);
	if ( compare_result > 0 || index > 24u )
		return("unknown");
	return(names[index]);
}

typedef struct SparkQwen38_27bStagePackTensorShape
{
	uint32_t rows;
	uint32_t columns;
	uint32_t natural_format;
	uint32_t quantizable;
	uint32_t layer_class;
} SparkQwen38_27bStagePackTensorShape;

static inline int32_t SparkQwen38_27bStagePackShapeGlobal(uint32_t tensor_kind, SparkQwen38_27bStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN38_27B_STAGEPACK_CLASS_GLOBAL;
	switch ( tensor_kind )
	{
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_EMBEDDING:
		shape->rows = SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT;
		shape->columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows = SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT;
		shape->columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FC:
		shape->rows = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		shape->columns = 2u * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_EMBED_NORM:
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_HIDDEN_NORM:
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen38_27bStagePackShapeEveryLayer(uint32_t tensor_kind, SparkQwen38_27bStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN38_27B_STAGEPACK_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTENTION_NORM:
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_MLP_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE:
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_UP:
		shape->rows = SPARK_QWEN38_27B_MODEL_FFN_INTERMEDIATE_DIMENSION;
		shape->columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_DOWN:
		shape->rows = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_QWEN38_27B_MODEL_FFN_INTERMEDIATE_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	default:
		return(-1);
	}
}

#define SPARK_QWEN38_27B_STAGEPACK_GDN_KIND_OFFSET (SPARK_STAGEPACK_TENSOR_GDN_QKV - SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_QKV)

_Static_assert(SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_NORM + SPARK_QWEN38_27B_STAGEPACK_GDN_KIND_OFFSET == SPARK_STAGEPACK_TENSOR_GDN_NORM,"qwen38_27b gdn kind run must track the shared axis");
_Static_assert(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 == SPARK_STAGEPACK_FORMAT_WEIGHT_BF16,"qwen38_27b bf16 weight code must match the shared format");
_Static_assert(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 == SPARK_STAGEPACK_FORMAT_WEIGHT_F32,"qwen38_27b f32 weight code must match the shared format");

static const SparkStagePackGeometryTable SparkQwen38_27bStagePackGeometry =
{
	.norm_width = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION,
	.hidden_dimension = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION,
	.gdn_conv_channels = SPARK_QWEN38_27B_MODEL_GDN_CONV_CHANNELS,
	.gdn_value_dimension = SPARK_QWEN38_27B_MODEL_GDN_VALUE_DIMENSION,
	.gdn_value_head_count = SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT,
	.gdn_head_value_dimension = SPARK_QWEN38_27B_MODEL_GDN_HEAD_VALUE_DIMENSION,
	.gdn_conv_kernel = SPARK_QWEN38_27B_MODEL_GDN_CONV_KERNEL
};

static inline int32_t SparkQwen38_27bStagePackShapeGdn(uint32_t tensor_kind, SparkQwen38_27bStagePackTensorShape *shape)
{
	SparkStagePackTensorShape common;
	SparkStagePackShapeInit(&common);
	if ( SparkStagePackShapeGdnCommon(tensor_kind + SPARK_QWEN38_27B_STAGEPACK_GDN_KIND_OFFSET,&SparkQwen38_27bStagePackGeometry,&common) != 0 )
		return(-1);
	shape->rows = common.rows;
	shape->columns = common.columns;
	shape->natural_format = common.natural_format;
	shape->layer_class = common.layer_class;
	if ( tensor_kind == SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_QKV || tensor_kind == SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_GATE || tensor_kind == SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_OUTPUT )
		shape->quantizable = 1u;
	return(0);
}

static inline int32_t SparkQwen38_27bStagePackShapeAttn(uint32_t tensor_kind, SparkQwen38_27bStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN38_27B_STAGEPACK_CLASS_ATTN_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY:
		shape->rows = 2u * SPARK_QWEN38_27B_MODEL_ATTN_QUERY_DIMENSION;
		shape->columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY:
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_VALUE:
		shape->rows = SPARK_QWEN38_27B_MODEL_ATTN_KV_DIMENSION;
		shape->columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->rows = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_QWEN38_27B_MODEL_ATTN_QUERY_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY_NORM:
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN38_27B_MODEL_ATTN_HEAD_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen38_27bStagePackTensorShapeOf(uint32_t tensor_kind, SparkQwen38_27bStagePackTensorShape *shape)
{
	shape->natural_format = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	shape->quantizable = 0u;
	if ( SparkQwen38_27bStagePackShapeGlobal(tensor_kind,shape) == 0 )
		return(0);
	if ( SparkQwen38_27bStagePackShapeEveryLayer(tensor_kind,shape) == 0 )
		return(0);
	if ( SparkQwen38_27bStagePackShapeGdn(tensor_kind,shape) == 0 )
		return(0);
	if ( SparkQwen38_27bStagePackShapeAttn(tensor_kind,shape) == 0 )
		return(0);
	return(-1);
}

static inline void SparkQwen38_27bStagePackApplyTpShard(uint32_t tensor_kind, uint32_t tp_degree, SparkQwen38_27bStagePackTensorShape *shape)
{
	if ( tp_degree <= 1u )
		return;
	switch ( tensor_kind )
	{
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_EMBEDDING:
		break;
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows /= tp_degree;
		break;
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE:
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_UP:
		shape->rows /= tp_degree;
		break;
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_DOWN:
		shape->columns /= tp_degree;
		break;
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_QKV:
		shape->rows = (2u * SPARK_QWEN38_27B_MODEL_GDN_QK_DIMENSION +
			SPARK_QWEN38_27B_MODEL_GDN_VALUE_DIMENSION) / tp_degree;
		break;
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_OUTPUT:
		shape->columns /= tp_degree;
		break;
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY:
		shape->rows /= tp_degree;
		break;
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY:
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_VALUE:
		shape->rows /= tp_degree;
		break;
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->columns /= tp_degree;
		break;
	default:
		break;
	}
}

static inline int32_t SparkQwen38_27bStagePackResolvedShape(uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, uint32_t tp_degree, SparkQwen38_27bStagePackTensorShape *shape)
{
	if ( SparkQwen38_27bStagePackTensorShapeOf(tensor_kind,shape) < 0 )
		return(-1);
	SparkQwen38_27bStagePackApplyTpShard(tensor_kind,tp_degree,shape);
	if ( layer_index == SPARK_QWEN38_27B_STAGEPACK_MTP_LAYER )
		return((is_global == 0u && (shape->layer_class == SPARK_QWEN38_27B_STAGEPACK_CLASS_EVERY_LAYER || shape->layer_class == SPARK_QWEN38_27B_STAGEPACK_CLASS_ATTN_LAYER)) ? 0 : -6);
	if ( (shape->layer_class == SPARK_QWEN38_27B_STAGEPACK_CLASS_GLOBAL) != (is_global != 0u) )
		return(-2);
	if ( is_global != 0u )
		return(0);
	if ( layer_index >= SPARK_QWEN38_27B_MODEL_LAYER_COUNT )
		return(-3);
	if ( shape->layer_class == SPARK_QWEN38_27B_STAGEPACK_CLASS_GDN_LAYER && SPARK_QWEN38_27B_MODEL_LAYER_IS_GDN(layer_index) == 0u )
		return(-4);
	if ( shape->layer_class == SPARK_QWEN38_27B_STAGEPACK_CLASS_ATTN_LAYER && SPARK_QWEN38_27B_MODEL_LAYER_IS_GDN(layer_index) != 0u )
		return(-5);
	return(0);
}

static inline uint64_t SparkQwen38_27bStagePackPayloadBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(elements / 2u);
	if ( weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 ||
		weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
		return(elements);
	if ( weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 || weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 )
		return(elements * 4u);
	return(elements * (uint64_t)SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES);
}

static inline uint64_t SparkQwen38_27bStagePackScaleBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	if ( weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(((uint64_t)rows * (uint64_t)columns) / 32u);
	if ( weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(((uint64_t)rows / 128u) * ((uint64_t)columns / 128u) * 4u);
	if ( weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
		return((uint64_t)rows * ((uint64_t)columns / 128u));
	return(0u);
}
