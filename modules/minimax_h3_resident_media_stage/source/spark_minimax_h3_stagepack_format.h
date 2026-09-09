#pragma once

#include <stdint.h>

#include "sparkpipe/spark_minimax_h3_kv_geometry.h"
#include "sparkpipe/spark_stagepack_format.h"
#include "sparkpipe/spark_status.h"

#define SPARK_MINIMAX_H3_STAGEPACK_MAGIC 0x50533348u
#define SPARK_MINIMAX_H3_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_MINIMAX_H3_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_MINIMAX_H3_STAGEPACK_PAYLOAD_ALIGNMENT 256u

#define SPARK_MINIMAX_H3_STAGEPACK_SECTION_ENCODER 0x1000u
#define SPARK_MINIMAX_H3_STAGEPACK_SECTION_DIT 0x2000u
#define SPARK_MINIMAX_H3_STAGEPACK_SECTION_VIDEO_VAE 0x3000u
#define SPARK_MINIMAX_H3_STAGEPACK_SECTION_AUDIO_VAE 0x4000u
#define SPARK_MINIMAX_H3_STAGEPACK_SECTION_MASK 0xf000u
#define SPARK_MINIMAX_H3_STAGEPACK_KIND_MASK 0x0fffu

typedef enum SparkMinimaxH3StagePackTensorKind
{
	SPARK_MINIMAX_H3_ENCODER_EMBEDDING = 0x1000,
	SPARK_MINIMAX_H3_ENCODER_FINAL_NORM = 0x1001,
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_QUERY = 0x1010,
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_KEY = 0x1011,
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_VALUE = 0x1012,
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_OUTPUT = 0x1013,
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_QUERY_NORM = 0x1014,
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_KEY_NORM = 0x1015,
	SPARK_MINIMAX_H3_ENCODER_MLP_GATE_UP = 0x1016,
	SPARK_MINIMAX_H3_ENCODER_MLP_DOWN = 0x1017,
	SPARK_MINIMAX_H3_ENCODER_INPUT_NORM = 0x1018,
	SPARK_MINIMAX_H3_ENCODER_POST_NORM = 0x1019,
	SPARK_MINIMAX_H3_DIT_CONTEXT_EMBEDDER = 0x2000,
	SPARK_MINIMAX_H3_DIT_TIME_EMBED_LINEAR_1 = 0x2001,
	SPARK_MINIMAX_H3_DIT_TIME_EMBED_LINEAR_2 = 0x2002,
	SPARK_MINIMAX_H3_DIT_TIME_EMBED_NORM = 0x2003,
	SPARK_MINIMAX_H3_DIT_PROJ_IN = 0x2004,
	SPARK_MINIMAX_H3_DIT_PROJ_OUT = 0x2005,
	SPARK_MINIMAX_H3_DIT_AUDIO_PROJ_IN = 0x2006,
	SPARK_MINIMAX_H3_DIT_AUDIO_PROJ_OUT = 0x2007,
	SPARK_MINIMAX_H3_DIT_FINAL_NORM = 0x2008,
	SPARK_MINIMAX_H3_DIT_NORM_OUT_LINEAR = 0x2009,
	SPARK_MINIMAX_H3_DIT_ATTENTION_QUERY = 0x2010,
	SPARK_MINIMAX_H3_DIT_ATTENTION_KEY = 0x2011,
	SPARK_MINIMAX_H3_DIT_ATTENTION_VALUE = 0x2012,
	SPARK_MINIMAX_H3_DIT_ATTENTION_OUTPUT = 0x2013,
	SPARK_MINIMAX_H3_DIT_ATTENTION_QUERY_NORM = 0x2014,
	SPARK_MINIMAX_H3_DIT_ATTENTION_KEY_NORM = 0x2015,
	SPARK_MINIMAX_H3_DIT_FFN_GATE_UP = 0x2016,
	SPARK_MINIMAX_H3_DIT_FFN_DOWN = 0x2017,
	SPARK_MINIMAX_H3_DIT_ADALN = 0x2018,
	SPARK_MINIMAX_H3_DIT_NORM1 = 0x2019,
	SPARK_MINIMAX_H3_DIT_NORM2 = 0x201a,
	SPARK_MINIMAX_H3_DIT_KIND_COUNT = 0x201b
} SparkMinimaxH3StagePackTensorKind;

typedef struct SparkMinimaxH3StagePackHeader
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
} SparkMinimaxH3StagePackHeader;

typedef struct SparkMinimaxH3StagePackEntry
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
} SparkMinimaxH3StagePackEntry;

#define SPARK_MINIMAX_H3_STAGEPACK_HEADER_BYTES 120u
#define SPARK_MINIMAX_H3_STAGEPACK_ENTRY_BYTES 56u

#define SPARK_MINIMAX_H3_DIT_TOTAL_BLOCK_COUNT \
	(SPARK_MINIMAX_H3_DIT_BLOCK_COUNT + SPARK_MINIMAX_H3_DIT_REFINER_BLOCK_COUNT)

_Static_assert(sizeof(SparkMinimaxH3StagePackHeader) ==
	SPARK_MINIMAX_H3_STAGEPACK_HEADER_BYTES,
	"h3 stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkMinimaxH3StagePackEntry) ==
	SPARK_MINIMAX_H3_STAGEPACK_ENTRY_BYTES,
	"h3 stage pack directory entry must be 56 wire bytes");
SPARK_STAGEPACK_HEADER_LAYOUT_PROOF(SparkMinimaxH3StagePackHeader);
_Static_assert((SPARK_MINIMAX_H3_STAGEPACK_SECTION_DIT &
	SPARK_MINIMAX_H3_STAGEPACK_KIND_MASK) == 0u,
	"h3 section codes must occupy the kind high bits");
_Static_assert(SPARK_MINIMAX_H3_DIT_ADALN_ROWS ==
	(18u * SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION),
	"h3 adaln modulation must cover 18 hidden-wide planes");
_Static_assert(SPARK_MINIMAX_H3_DIT_TOTAL_BLOCK_COUNT == 52u,
	"h3 dit must carry the 50 main blocks plus 2 refiner blocks");

static inline uint32_t SparkMinimaxH3StagePackSection(
	uint32_t tensor_kind)
{
	return(tensor_kind & SPARK_MINIMAX_H3_STAGEPACK_SECTION_MASK);
}

static inline uint32_t SparkMinimaxH3StagePackKindInSection(
	uint32_t tensor_kind)
{
	return(tensor_kind & SPARK_MINIMAX_H3_STAGEPACK_KIND_MASK);
}

static inline uint32_t SparkMinimaxH3StagePackDitBlockCountForLayer(
	uint32_t layer_index)
{
	return(layer_index < SPARK_MINIMAX_H3_DIT_BLOCK_COUNT ? 1u : 0u);
}

static inline int32_t SparkMinimaxH3StagePackShapeEncoderGlobal(
	uint32_t tensor_kind, SparkStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL;
	switch ( tensor_kind )
	{
	case SPARK_MINIMAX_H3_ENCODER_EMBEDDING:
		shape->rows = SPARK_MINIMAX_H3_ENCODER_VOCAB_COUNT;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_ENCODER_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkMinimaxH3StagePackShapeEncoderLayer(
	uint32_t tensor_kind, SparkStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_MINIMAX_H3_ENCODER_ATTENTION_QUERY:
		shape->rows = SPARK_MINIMAX_H3_ENCODER_QUERY_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_ENCODER_ATTENTION_KEY:
	case SPARK_MINIMAX_H3_ENCODER_ATTENTION_VALUE:
		shape->rows = SPARK_MINIMAX_H3_ENCODER_KV_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_ENCODER_ATTENTION_OUTPUT:
		shape->rows = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_QUERY_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_ENCODER_ATTENTION_QUERY_NORM:
	case SPARK_MINIMAX_H3_ENCODER_ATTENTION_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_HEAD_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_ENCODER_MLP_GATE_UP:
		shape->rows = 2u * SPARK_MINIMAX_H3_ENCODER_INTERMEDIATE_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_ENCODER_MLP_DOWN:
		shape->rows = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_INTERMEDIATE_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_ENCODER_INPUT_NORM:
	case SPARK_MINIMAX_H3_ENCODER_POST_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkMinimaxH3StagePackShapeDitGlobal(
	uint32_t tensor_kind, SparkStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL;
	switch ( tensor_kind )
	{
	case SPARK_MINIMAX_H3_DIT_CONTEXT_EMBEDDER:
		shape->rows = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_CONTEXT_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_TIME_EMBED_LINEAR_1:
		shape->rows = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_TIME_FREQ_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_TIME_EMBED_LINEAR_2:
		shape->rows = SPARK_MINIMAX_H3_DIT_TIME_EMBED_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_TIME_EMBED_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_DIT_TIME_FREQ_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_PROJ_IN:
		shape->rows = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_PATCH_ELEMENTS;
		return(0);
	case SPARK_MINIMAX_H3_DIT_PROJ_OUT:
		shape->rows = SPARK_MINIMAX_H3_DIT_PATCH_ELEMENTS;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_AUDIO_PROJ_IN:
		shape->rows = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_AUDIO_PATCH_ELEMENTS;
		return(0);
	case SPARK_MINIMAX_H3_DIT_AUDIO_PROJ_OUT:
		shape->rows = SPARK_MINIMAX_H3_DIT_AUDIO_PATCH_ELEMENTS;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_NORM_OUT_LINEAR:
		shape->rows = SPARK_MINIMAX_H3_DIT_NORM_OUT_ROWS;
		shape->columns = SPARK_MINIMAX_H3_DIT_TIME_EMBED_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkMinimaxH3StagePackShapeDitBlock(
	uint32_t tensor_kind, SparkStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_MINIMAX_H3_DIT_ATTENTION_QUERY:
	case SPARK_MINIMAX_H3_DIT_ATTENTION_KEY:
	case SPARK_MINIMAX_H3_DIT_ATTENTION_VALUE:
		shape->rows = SPARK_MINIMAX_H3_DIT_QKV_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_ATTENTION_OUTPUT:
		shape->rows = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_QKV_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_ATTENTION_QUERY_NORM:
	case SPARK_MINIMAX_H3_DIT_ATTENTION_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_DIT_HEAD_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_FFN_GATE_UP:
		shape->rows = SPARK_MINIMAX_H3_DIT_FFN_FUSED_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_FFN_DOWN:
		shape->rows = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_FFN_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_ADALN:
		shape->rows = SPARK_MINIMAX_H3_DIT_ADALN_ROWS;
		shape->columns = SPARK_MINIMAX_H3_DIT_TIME_EMBED_DIMENSION;
		return(0);
	case SPARK_MINIMAX_H3_DIT_NORM1:
	case SPARK_MINIMAX_H3_DIT_NORM2:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkMinimaxH3StagePackTensorShapeOf(
	uint32_t tensor_kind, SparkStagePackTensorShape *shape)
{
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeEncoderGlobal(tensor_kind,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeEncoderLayer(tensor_kind,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeDitGlobal(tensor_kind,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeDitBlock(tensor_kind,shape) == 0 )
		return(0);
	return(-1);
}

static inline int32_t SparkMinimaxH3StagePackResolvedShape(
	uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global,
	SparkStagePackTensorShape *shape)
{
	uint32_t section;
	if ( SparkMinimaxH3StagePackTensorShapeOf(tensor_kind,shape) < 0 )
		return(-1);
	section = SparkMinimaxH3StagePackSection(tensor_kind);
	if ( section != SPARK_MINIMAX_H3_STAGEPACK_SECTION_ENCODER &&
		section != SPARK_MINIMAX_H3_STAGEPACK_SECTION_DIT )
		return(-2);
	if ( (shape->layer_class == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL) !=
		(is_global != 0u) )
		return(-3);
	if ( is_global != 0u )
		return(0);
	if ( section == SPARK_MINIMAX_H3_STAGEPACK_SECTION_ENCODER &&
		layer_index >= SPARK_MINIMAX_H3_ENCODER_LAYER_COUNT )
		return(-4);
	if ( section == SPARK_MINIMAX_H3_STAGEPACK_SECTION_DIT &&
		layer_index >= SPARK_MINIMAX_H3_DIT_TOTAL_BLOCK_COUNT )
		return(-5);
	return(0);
}

static inline uint32_t SparkMinimaxH3StagePackEncoderTensorCount(void)
{
	return(2u + (SPARK_MINIMAX_H3_ENCODER_LAYER_COUNT * 9u));
}

static inline uint32_t SparkMinimaxH3StagePackDitGlobalTensorCount(void)
{
	return(10u);
}

static inline uint32_t SparkMinimaxH3StagePackDitBlockTensorCount(void)
{
	return(11u);
}

static inline uint32_t SparkMinimaxH3StagePackDitTensorCount(void)
{
	return(SparkMinimaxH3StagePackDitGlobalTensorCount() +
		(SPARK_MINIMAX_H3_DIT_TOTAL_BLOCK_COUNT *
		SparkMinimaxH3StagePackDitBlockTensorCount()));
}

static inline void SparkMinimaxH3StagePackExpectedGeometry(
	SparkMinimaxH3StagePackHeader *header)
{
	header->magic = SPARK_MINIMAX_H3_STAGEPACK_MAGIC;
	header->format_version = SPARK_MINIMAX_H3_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_MINIMAX_H3_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_MINIMAX_H3_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkMinimaxH3StagePackEncoderTensorCount() +
		SparkMinimaxH3StagePackDitTensorCount();
	header->hidden_dimension = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
	header->layer_count = SPARK_MINIMAX_H3_DIT_BLOCK_COUNT;
	header->first_layer_index = 0u;
	header->total_layer_count = SPARK_MINIMAX_H3_DIT_BLOCK_COUNT;
	header->attention_period = 0u;
	header->full_attention_phase = 0u;
	header->gdn_key_head_count = 0u;
	header->gdn_value_head_count = 0u;
	header->gdn_head_key_dimension = 0u;
	header->gdn_head_value_dimension = 0u;
	header->gdn_conv_kernel = 0u;
	header->attn_query_head_count = SPARK_MINIMAX_H3_DIT_ATTENTION_HEAD_COUNT;
	header->attn_kv_head_count = SPARK_MINIMAX_H3_DIT_ATTENTION_HEAD_COUNT;
	header->attn_head_dimension = SPARK_MINIMAX_H3_DIT_HEAD_DIMENSION;
	header->attn_rope_dimension = SPARK_MINIMAX_H3_DIT_ROPE_FREQ_DIMENSION;
	header->routed_expert_count = 0u;
	header->experts_per_token = 0u;
	header->expert_intermediate_dimension =
		SPARK_MINIMAX_H3_DIT_FFN_DIMENSION;
	header->output_vocab_count = SPARK_MINIMAX_H3_ENCODER_VOCAB_COUNT;
	header->mxfp4_group_size = 0u;
	header->mtp_layer_count = 0u;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

static inline int32_t SparkMinimaxH3StagePackHeaderMatches(
	const SparkMinimaxH3StagePackHeader *file_header,
	const SparkMinimaxH3StagePackHeader *expected)
{
	return(SparkStagePackHeaderMatches(
		(const SparkStagePackHeaderCommon *)file_header,
		(const SparkStagePackHeaderCommon *)expected));
}

static inline uint64_t SparkMinimaxH3StagePackPayloadBytes(
	uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_F32 )
		return(elements * 4u);
	if ( weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_I64 )
		return(elements * 8u);
	return(elements * (uint64_t)SPARK_MINIMAX_H3_BF16_ELEMENT_BYTES);
}
