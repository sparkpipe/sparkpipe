#pragma once

#include <stdint.h>

#include "sparkpipe/spark_muse_glimmer_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stagepack_format.h"
#include "sparkpipe/spark_status.h"


#define SPARK_MUSE_GLIMMER_STAGEPACK_MAGIC 0x47534D55u
#define SPARK_MUSE_GLIMMER_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_MUSE_GLIMMER_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_MUSE_GLIMMER_STAGEPACK_PAYLOAD_ALIGNMENT 256u

typedef enum SparkMuseGlimmerStagePackTensorKind
{
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_INPUT_NORM = 40,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_POST_ATTENTION_NORM = 41,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_PRE_FFN_NORM = 42,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_POST_FFN_NORM = 43,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_QGKV = 44,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_ATTN_OUTPUT = 45,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_ATTN_GATE_UP = 46,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_MLP_DOWN = 47,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_KIND_COUNT = 48
} SparkMuseGlimmerStagePackTensorKind;

#define SPARK_MUSE_GLIMMER_STAGEPACK_CLASS_GLOBAL SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL
#define SPARK_MUSE_GLIMMER_STAGEPACK_CLASS_EVERY_LAYER SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER

typedef struct SparkMuseGlimmerStagePackHeader
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
} SparkMuseGlimmerStagePackHeader;

typedef struct SparkMuseGlimmerStagePackEntry
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
} SparkMuseGlimmerStagePackEntry;

#define SPARK_MUSE_GLIMMER_STAGEPACK_HEADER_BYTES 120u
#define SPARK_MUSE_GLIMMER_STAGEPACK_ENTRY_BYTES 56u
_Static_assert(sizeof(SparkMuseGlimmerStagePackHeader) == SPARK_MUSE_GLIMMER_STAGEPACK_HEADER_BYTES,"muse stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkMuseGlimmerStagePackEntry) == SPARK_MUSE_GLIMMER_STAGEPACK_ENTRY_BYTES,"muse stage pack directory entry must be 56 wire bytes");

_Static_assert(SPARK_MUSE_GLIMMER_MODEL_SLIDING_LAYER_COUNT + SPARK_MUSE_GLIMMER_MODEL_FULL_ATTENTION_LAYER_COUNT == SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT,"muse layer split must cover the stack");
_Static_assert((SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT % SPARK_MUSE_GLIMMER_MODEL_ATTENTION_PERIOD) == 0u,"muse layer count must be whole periods");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_SLIDING_LAYER_COUNT == (SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT / SPARK_MUSE_GLIMMER_MODEL_ATTENTION_PERIOD) * (SPARK_MUSE_GLIMMER_MODEL_ATTENTION_PERIOD - 1u),"muse sliding count must match the 3:1 period");
_Static_assert((SPARK_MUSE_GLIMMER_MODEL_ATTN_QUERY_HEAD_COUNT % SPARK_MUSE_GLIMMER_MODEL_ATTN_KV_HEAD_COUNT) == 0u,"muse query heads must group evenly onto kv heads");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_ATTN_ROPE_DIMENSION == SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,"muse rope covers the whole head");
_Static_assert((SPARK_MUSE_GLIMMER_MODEL_ATTN_ROPE_DIMENSION % 2u) == 0u,"muse rope dimension must pair");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_ATTN_QUERY_DIMENSION == 4096u && SPARK_MUSE_GLIMMER_MODEL_ATTN_KV_DIMENSION == 256u,"muse attention projection widths per config");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_INTERMEDIATE_DIMENSION == 19968u,"muse mlp intermediate per config");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_MTP_LAYER_COUNT == 0u,"muse has no mtp layers");
_Static_assert((SPARK_MUSE_GLIMMER_MODEL_VOCAB_COUNT % 16u) == 0u && (SPARK_MUSE_GLIMMER_MODEL_INTERMEDIATE_DIMENSION % 16u) == 0u,"muse vocab and intermediate must shard evenly across 16 ranks");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_KV_SHARD_COUNT(16u) == 2u && SPARK_MUSE_GLIMMER_MODEL_ATTN_RANK_KV_HEAD_BASE(16u,7u) == 0u && SPARK_MUSE_GLIMMER_MODEL_ATTN_RANK_KV_HEAD_BASE(16u,8u) == 1u,"muse tp16 kv ownership replicates head 0 on ranks 0-7 and head 1 on ranks 8-15");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_HEAD_COUNT(1u) == 2u,"tp1 keeps both kv heads rank-local");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_QGKV_LOCAL_ROWS(16u) == 768u,"the fused qgkv projection is 768 rank-local rows at tp16");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_GATE_DIMENSION(16u) == 512u && SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_DIMENSION(16u) == 128u,"tp16 local attention widths per the staged target");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_MLP_LOCAL_INTERMEDIATE(16u) == 1248u,"tp16 mlp intermediate per the staged target");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_VOCAB_LOCAL_ROWS(16u) == 12628u,"tp16 vocab rows per the staged target");
_Static_assert(SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION * 2u * SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_HEAD_COUNT(16u) * SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES == 512u,"the tp16 kv slot is 512 bytes");

static inline uint32_t SparkMuseGlimmerStagePackExpectedTensorCount(uint32_t first_layer_index, uint32_t layer_count)
{
	uint32_t tensors = layer_count * 8u;
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT )
		tensors += 2u;
	return(tensors);
}

static inline void SparkMuseGlimmerStagePackExpectedGeometry(SparkMuseGlimmerStagePackHeader *header, uint32_t first_layer_index, uint32_t layer_count)
{
	header->magic = SPARK_MUSE_GLIMMER_STAGEPACK_MAGIC;
	header->format_version = SPARK_MUSE_GLIMMER_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_MUSE_GLIMMER_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_MUSE_GLIMMER_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkMuseGlimmerStagePackExpectedTensorCount(first_layer_index,layer_count);
	header->hidden_dimension = SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION;
	header->layer_count = layer_count;
	header->first_layer_index = first_layer_index;
	header->total_layer_count = SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT;
	header->attention_period = SPARK_MUSE_GLIMMER_MODEL_ATTENTION_PERIOD;
	header->full_attention_phase = SPARK_MUSE_GLIMMER_MODEL_FULL_ATTENTION_PHASE;
	header->gdn_key_head_count = 0u;
	header->gdn_value_head_count = 0u;
	header->gdn_head_key_dimension = 0u;
	header->gdn_head_value_dimension = 0u;
	header->gdn_conv_kernel = 0u;
	header->attn_query_head_count = SPARK_MUSE_GLIMMER_MODEL_ATTN_QUERY_HEAD_COUNT;
	header->attn_kv_head_count = SPARK_MUSE_GLIMMER_MODEL_ATTN_KV_HEAD_COUNT;
	header->attn_head_dimension = SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION;
	header->attn_rope_dimension = SPARK_MUSE_GLIMMER_MODEL_ATTN_ROPE_DIMENSION;
	header->routed_expert_count = 0u;
	header->experts_per_token = 0u;
	header->expert_intermediate_dimension = SPARK_MUSE_GLIMMER_MODEL_INTERMEDIATE_DIMENSION;
	header->output_vocab_count = SPARK_MUSE_GLIMMER_MODEL_OUTPUT_VOCAB_COUNT;
	header->mxfp4_group_size = 0u;
	header->mtp_layer_count = SPARK_MUSE_GLIMMER_MODEL_MTP_LAYER_COUNT;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

SPARK_STAGEPACK_HEADER_LAYOUT_PROOF(SparkMuseGlimmerStagePackHeader);
static inline int32_t SparkMuseGlimmerStagePackHeaderMatches(const SparkMuseGlimmerStagePackHeader *file_header, const SparkMuseGlimmerStagePackHeader *expected)
{
	return(SparkStagePackHeaderMatches(
		(const SparkStagePackHeaderCommon *)file_header,
		(const SparkStagePackHeaderCommon *)expected));
}

typedef SparkStagePackTensorShape SparkMuseGlimmerStagePackTensorShape;

_Static_assert(SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 == SPARK_STAGEPACK_FORMAT_WEIGHT_BF16,"muse bf16 weight code must match the shared format");

static inline int32_t SparkMuseGlimmerStagePackResolvedShape(uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, uint32_t tp_degree, SparkMuseGlimmerStagePackTensorShape *shape)
{
	shape->natural_format = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	shape->layer_class = SPARK_MUSE_GLIMMER_STAGEPACK_CLASS_EVERY_LAYER;
	shape->rows = 0u;
	shape->columns = SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION;
	if ( is_global != 0u )
	{
		shape->layer_class = SPARK_MUSE_GLIMMER_STAGEPACK_CLASS_GLOBAL;
		switch ( tensor_kind )
		{
		case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_EMBEDDING:
		case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_LM_HEAD:
			shape->rows = SPARK_MUSE_GLIMMER_MODEL_VOCAB_LOCAL_ROWS(tp_degree);
			return(0);
		case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_FINAL_NORM:
			shape->rows = 1u;
			return(0);
		default:
			return(-1);
		}
	}
	if ( layer_index >= SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT )
		return(-3);
	switch ( tensor_kind )
	{
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_INPUT_NORM:
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_POST_ATTENTION_NORM:
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_PRE_FFN_NORM:
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_POST_FFN_NORM:
		shape->rows = 1u;
		return(0);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_QGKV:
		shape->rows = SPARK_MUSE_GLIMMER_MODEL_QGKV_LOCAL_ROWS(tp_degree);
		return(0);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->rows = SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_DIMENSION(tp_degree);
		return(0);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_ATTN_GATE_UP:
		shape->rows = 2u * SPARK_MUSE_GLIMMER_MODEL_MLP_LOCAL_INTERMEDIATE(tp_degree);
		return(0);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_MLP_DOWN:
		shape->rows = SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_MUSE_GLIMMER_MODEL_MLP_LOCAL_INTERMEDIATE(tp_degree);
		return(0);
	default:
		return(-1);
	}
}

static inline uint64_t SparkMuseGlimmerStagePackPayloadBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 || weight_format == SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 )
		return(elements * 4u);
	return(elements * (uint64_t)SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES);
}

static inline uint64_t SparkMuseGlimmerStagePackScaleBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	(void)weight_format;
	(void)rows;
	(void)columns;
	return(0u);
}
