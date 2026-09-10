#pragma once

#include <stdint.h>

#include "sparkpipe/spark_minimax_h3_kv_geometry.h"
#include "sparkpipe/spark_stagepack_format.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_status.h"

#define SPARK_MINIMAX_H3_STAGEPACK_MAGIC 0x50533348u
#define SPARK_MINIMAX_H3_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_MINIMAX_H3_STAGEPACK_GLOBAL_LAYER 0xffffu
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
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_QUERY = 0x1001,
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_KEY = 0x1002,
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_VALUE = 0x1003,
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_OUTPUT = 0x1004,
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_QUERY_NORM = 0x1005,
	SPARK_MINIMAX_H3_ENCODER_ATTENTION_KEY_NORM = 0x1006,
	SPARK_MINIMAX_H3_ENCODER_MLP_GATE = 0x1007,
	SPARK_MINIMAX_H3_ENCODER_MLP_UP = 0x1008,
	SPARK_MINIMAX_H3_ENCODER_MLP_DOWN = 0x1009,
	SPARK_MINIMAX_H3_ENCODER_INPUT_NORM = 0x100a,
	SPARK_MINIMAX_H3_ENCODER_POST_NORM = 0x100b,
	SPARK_MINIMAX_H3_DIT_PROJ_IN = 0x2000,
	SPARK_MINIMAX_H3_DIT_AUDIO_PROJ_IN = 0x2001,
	SPARK_MINIMAX_H3_DIT_CONTEXT_EMBEDDER = 0x2002,
	SPARK_MINIMAX_H3_DIT_TIME_EMBED_LINEAR_1 = 0x2003,
	SPARK_MINIMAX_H3_DIT_TIME_EMBED_LINEAR_2 = 0x2004,
	SPARK_MINIMAX_H3_DIT_ATTENTION_QUERY = 0x2005,
	SPARK_MINIMAX_H3_DIT_ATTENTION_KEY = 0x2006,
	SPARK_MINIMAX_H3_DIT_ATTENTION_VALUE = 0x2007,
	SPARK_MINIMAX_H3_DIT_ATTENTION_OUTPUT = 0x2008,
	SPARK_MINIMAX_H3_DIT_ATTENTION_QUERY_NORM = 0x2009,
	SPARK_MINIMAX_H3_DIT_ATTENTION_KEY_NORM = 0x200a,
	SPARK_MINIMAX_H3_DIT_FFN_GATE_UP = 0x200b,
	SPARK_MINIMAX_H3_DIT_FFN_DOWN = 0x200c,
	SPARK_MINIMAX_H3_DIT_NORM1 = 0x200d,
	SPARK_MINIMAX_H3_DIT_NORM2 = 0x200e,
	SPARK_MINIMAX_H3_DIT_FINAL_NORM = 0x200f,
	SPARK_MINIMAX_H3_DIT_NORM_OUT_LINEAR = 0x2010,
	SPARK_MINIMAX_H3_DIT_PROJ_OUT = 0x2011,
	SPARK_MINIMAX_H3_DIT_AUDIO_PROJ_OUT = 0x2012,
	SPARK_MINIMAX_H3_DIT_ADALN = 0x2013,
	SPARK_MINIMAX_H3_VIDEO_PROJ_IN = 0x3000,
	SPARK_MINIMAX_H3_VIDEO_REGISTER_TOKENS = 0x3001,
	SPARK_MINIMAX_H3_VIDEO_FINAL_NORM = 0x3002,
	SPARK_MINIMAX_H3_VIDEO_PROJ_OUT = 0x3003,
	SPARK_MINIMAX_H3_VIDEO_POST_QUANT_CONV = 0x3004,
	SPARK_MINIMAX_H3_VIDEO_ATTENTION_QUERY = 0x3005,
	SPARK_MINIMAX_H3_VIDEO_ATTENTION_KEY = 0x3006,
	SPARK_MINIMAX_H3_VIDEO_ATTENTION_VALUE = 0x3007,
	SPARK_MINIMAX_H3_VIDEO_ATTENTION_OUTPUT = 0x3008,
	SPARK_MINIMAX_H3_VIDEO_FFN_GATE_UP = 0x3009,
	SPARK_MINIMAX_H3_VIDEO_FFN_DOWN = 0x300a,
	SPARK_MINIMAX_H3_VIDEO_NORM1 = 0x300b,
	SPARK_MINIMAX_H3_VIDEO_NORM2 = 0x300c,
	SPARK_MINIMAX_H3_VIDEO_SCALE1 = 0x300d,
	SPARK_MINIMAX_H3_VIDEO_SCALE2 = 0x300e,
	SPARK_MINIMAX_H3_AUDIO_IN_PROJ = 0x4000,
	SPARK_MINIMAX_H3_AUDIO_CONV_PRE = 0x4001,
	SPARK_MINIMAX_H3_AUDIO_RESBLOCK_CONV1 = 0x4002,
	SPARK_MINIMAX_H3_AUDIO_RESBLOCK_CONV2 = 0x4003,
	SPARK_MINIMAX_H3_AUDIO_RESBLOCK_SNAKE = 0x4004,
	SPARK_MINIMAX_H3_AUDIO_RESBLOCK_FILTER = 0x4005,
	SPARK_MINIMAX_H3_AUDIO_UPS = 0x4006,
	SPARK_MINIMAX_H3_AUDIO_CONV_POST = 0x4007,
	SPARK_MINIMAX_H3_AUDIO_POST_SNAKE = 0x4008,
	SPARK_MINIMAX_H3_AUDIO_POST_FILTER = 0x4009,
	SPARK_MINIMAX_H3_STAGEPACK_KIND_COUNT = 0x400a
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

SPARK_MINIMAX_H3_STATIC_ASSERT(sizeof(SparkMinimaxH3StagePackHeader) ==
	SPARK_MINIMAX_H3_STAGEPACK_HEADER_BYTES,
	"h3 stage pack header must be 120 wire bytes");
SPARK_MINIMAX_H3_STATIC_ASSERT(sizeof(SparkMinimaxH3StagePackEntry) ==
	SPARK_MINIMAX_H3_STAGEPACK_ENTRY_BYTES,
	"h3 stage pack directory entry must be 56 wire bytes");
SPARK_STAGEPACK_HEADER_LAYOUT_PROOF(SparkMinimaxH3StagePackHeader);
SPARK_MINIMAX_H3_STATIC_ASSERT((SPARK_MINIMAX_H3_STAGEPACK_SECTION_DIT &
	SPARK_MINIMAX_H3_STAGEPACK_KIND_MASK) == 0u,
	"h3 section codes must occupy the kind high bits");
SPARK_MINIMAX_H3_STATIC_ASSERT(SPARK_MINIMAX_H3_DIT_TOTAL_BLOCK_COUNT == 52u,
	"h3 dit must carry the 50 main blocks plus 2 refiner blocks");

#define SPARK_MINIMAX_H3_STAGEPACK_LAYER_MASK 0xffffu
#define SPARK_MINIMAX_H3_STAGEPACK_SUB_SHIFT 16u
#define SPARK_MINIMAX_H3_STAGEPACK_SUB_WEIGHT 0u

static inline uint32_t SparkMinimaxH3StagePackSection(
	uint32_t tensor_kind)
{
	return(tensor_kind & SPARK_MINIMAX_H3_STAGEPACK_SECTION_MASK);
}

static inline uint32_t SparkMinimaxH3StagePackLayerOf(
	uint32_t packed_layer)
{
	return(packed_layer & SPARK_MINIMAX_H3_STAGEPACK_LAYER_MASK);
}

static inline uint32_t SparkMinimaxH3StagePackSubOf(
	uint32_t packed_layer)
{
	return(packed_layer >> SPARK_MINIMAX_H3_STAGEPACK_SUB_SHIFT);
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
		return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_NOT_FOUND);
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
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_ENCODER_ATTENTION_KEY:
	case SPARK_MINIMAX_H3_ENCODER_ATTENTION_VALUE:
		shape->rows = SPARK_MINIMAX_H3_ENCODER_KV_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_ENCODER_ATTENTION_OUTPUT:
		shape->rows = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_QUERY_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_ENCODER_ATTENTION_QUERY_NORM:
	case SPARK_MINIMAX_H3_ENCODER_ATTENTION_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_HEAD_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_ENCODER_MLP_GATE:
	case SPARK_MINIMAX_H3_ENCODER_MLP_UP:
		shape->rows = SPARK_MINIMAX_H3_ENCODER_INTERMEDIATE_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_ENCODER_MLP_DOWN:
		shape->rows = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_INTERMEDIATE_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_ENCODER_INPUT_NORM:
	case SPARK_MINIMAX_H3_ENCODER_POST_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_ENCODER_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_NOT_FOUND);
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
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_TIME_EMBED_LINEAR_1:
		shape->rows = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_TIME_FREQ_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_TIME_EMBED_LINEAR_2:
		shape->rows = SPARK_MINIMAX_H3_DIT_TIME_EMBED_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_PROJ_IN:
		shape->rows = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_PATCH_ELEMENTS;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_PROJ_OUT:
		shape->rows = SPARK_MINIMAX_H3_DIT_PATCH_ELEMENTS;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_AUDIO_PROJ_IN:
		shape->rows = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_AUDIO_PATCH_ELEMENTS;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_AUDIO_PROJ_OUT:
		shape->rows = SPARK_MINIMAX_H3_DIT_AUDIO_PATCH_ELEMENTS;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_NORM_OUT_LINEAR:
		shape->rows = SPARK_MINIMAX_H3_DIT_NORM_OUT_ROWS;
		shape->columns = SPARK_MINIMAX_H3_DIT_TIME_EMBED_DIMENSION;
		return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_NOT_FOUND);
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
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_ATTENTION_OUTPUT:
		shape->rows = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_QKV_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_ATTENTION_QUERY_NORM:
	case SPARK_MINIMAX_H3_DIT_ATTENTION_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_DIT_HEAD_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_FFN_GATE_UP:
		shape->rows = SPARK_MINIMAX_H3_DIT_FFN_FUSED_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_FFN_DOWN:
		shape->rows = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_DIT_FFN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_ADALN:
		shape->rows = SPARK_MINIMAX_H3_DIT_ADALN_ROWS;
		shape->columns = SPARK_MINIMAX_H3_DIT_TIME_EMBED_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_DIT_NORM1:
	case SPARK_MINIMAX_H3_DIT_NORM2:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_NOT_FOUND);
	}
}

static inline int32_t SparkMinimaxH3StagePackShapeVideoGlobal(
	uint32_t tensor_kind, SparkStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL;
	switch ( tensor_kind )
	{
	case SPARK_MINIMAX_H3_VIDEO_PROJ_IN:
		shape->rows = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_VIDEO_VAE_LATENT_CHANNELS;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_VIDEO_REGISTER_TOKENS:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_REGISTER_TOKEN_COUNT *
			SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_VIDEO_FINAL_NORM:
		shape->rows = 2u;
		shape->columns = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_VIDEO_PROJ_OUT:
		shape->rows = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_PROJ_OUT_ELEMENTS;
		shape->columns = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_VIDEO_POST_QUANT_CONV:
		shape->rows = SPARK_MINIMAX_H3_VIDEO_VAE_LATENT_CHANNELS;
		shape->columns = SPARK_MINIMAX_H3_VIDEO_VAE_LATENT_CHANNELS;
		return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_NOT_FOUND);
	}
}

static inline int32_t SparkMinimaxH3StagePackShapeVideoBlock(
	uint32_t tensor_kind, SparkStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_MINIMAX_H3_VIDEO_ATTENTION_QUERY:
	case SPARK_MINIMAX_H3_VIDEO_ATTENTION_KEY:
	case SPARK_MINIMAX_H3_VIDEO_ATTENTION_VALUE:
	case SPARK_MINIMAX_H3_VIDEO_ATTENTION_OUTPUT:
		shape->rows = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_VIDEO_FFN_GATE_UP:
		shape->rows = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_FFN_FUSED_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_VIDEO_FFN_DOWN:
		shape->rows = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_FFN_DIMENSION;
		return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_H3_VIDEO_NORM1:
	case SPARK_MINIMAX_H3_VIDEO_NORM2:
	case SPARK_MINIMAX_H3_VIDEO_SCALE1:
	case SPARK_MINIMAX_H3_VIDEO_SCALE2:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_HIDDEN_DIMENSION;
		return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_NOT_FOUND);
	}
}

static inline int32_t SparkMinimaxH3StagePackShapeAudioGlobal(
	uint32_t tensor_kind, SparkStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL;
	shape->rows = 0u;
	shape->columns = 0u;
	switch ( tensor_kind )
	{
	case SPARK_MINIMAX_H3_AUDIO_IN_PROJ:
	case SPARK_MINIMAX_H3_AUDIO_CONV_PRE:
	case SPARK_MINIMAX_H3_AUDIO_UPS:
	case SPARK_MINIMAX_H3_AUDIO_CONV_POST:
	case SPARK_MINIMAX_H3_AUDIO_POST_SNAKE:
	case SPARK_MINIMAX_H3_AUDIO_POST_FILTER:
		return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_NOT_FOUND);
	}
}

static inline int32_t SparkMinimaxH3StagePackShapeAudioBlock(
	uint32_t tensor_kind, SparkStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER;
	shape->rows = 0u;
	shape->columns = 0u;
	switch ( tensor_kind )
	{
	case SPARK_MINIMAX_H3_AUDIO_RESBLOCK_CONV1:
	case SPARK_MINIMAX_H3_AUDIO_RESBLOCK_CONV2:
	case SPARK_MINIMAX_H3_AUDIO_RESBLOCK_SNAKE:
	case SPARK_MINIMAX_H3_AUDIO_RESBLOCK_FILTER:
		return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_NOT_FOUND);
	}
}

static inline int32_t SparkMinimaxH3StagePackTensorShapeOf(
	uint32_t tensor_kind, SparkStagePackTensorShape *shape)
{
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeEncoderGlobal(tensor_kind,shape) == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeEncoderLayer(tensor_kind,shape) == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeDitGlobal(tensor_kind,shape) == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeDitBlock(tensor_kind,shape) == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeVideoGlobal(tensor_kind,shape) == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeVideoBlock(tensor_kind,shape) == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeAudioGlobal(tensor_kind,shape) == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	SparkStagePackShapeInit(shape);
	if ( SparkMinimaxH3StagePackShapeAudioBlock(tensor_kind,shape) == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	return(SPARK_STATUS_NOT_FOUND);
}

static inline int32_t SparkMinimaxH3StagePackResolvedShape(
	uint32_t tensor_kind, uint32_t packed_layer, uint32_t is_global,
	SparkStagePackTensorShape *shape)
{
	uint32_t section,layer;
	if ( SparkMinimaxH3StagePackTensorShapeOf(tensor_kind,shape) !=
		SPARK_STATUS_OK )
		return(SPARK_STATUS_NOT_FOUND);
	if ( SparkMinimaxH3StagePackSubOf(packed_layer) !=
		SPARK_MINIMAX_H3_STAGEPACK_SUB_WEIGHT )
		return(SPARK_STATUS_OK);
	section = SparkMinimaxH3StagePackSection(tensor_kind);
	if ( (shape->layer_class == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL) !=
		(is_global != 0u) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( is_global != 0u )
		return(SPARK_STATUS_OK);
	layer = SparkMinimaxH3StagePackLayerOf(packed_layer);
	if ( section == SPARK_MINIMAX_H3_STAGEPACK_SECTION_ENCODER &&
		layer >= SPARK_MINIMAX_H3_ENCODER_LAYER_COUNT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( section == SPARK_MINIMAX_H3_STAGEPACK_SECTION_DIT &&
		layer >= SPARK_MINIMAX_H3_DIT_TOTAL_BLOCK_COUNT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( section == SPARK_MINIMAX_H3_STAGEPACK_SECTION_VIDEO_VAE &&
		layer >= SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_LAYER_COUNT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( section == SPARK_MINIMAX_H3_STAGEPACK_SECTION_AUDIO_VAE &&
		layer >= (SPARK_MINIMAX_H3_AUDIO_VAE_RESBLOCK_STAGE_COUNT *
		SPARK_MINIMAX_H3_AUDIO_VAE_RESBLOCKS_PER_STAGE) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static inline uint32_t SparkMinimaxH3StagePackEncoderTensorCount(void)
{
	return(1u + (SPARK_MINIMAX_H3_ENCODER_LAYER_COUNT * 11u));
}

static inline uint32_t SparkMinimaxH3StagePackDitGlobalTensorCount(void)
{
	return(18u);
}

static inline uint32_t SparkMinimaxH3StagePackDitBlockTensorCount(
	uint32_t layer)
{
	return(layer < SPARK_MINIMAX_H3_DIT_BLOCK_COUNT ? 12u : 10u);
}

static inline uint32_t SparkMinimaxH3StagePackDitTensorCount(void)
{
	uint32_t count,layer;
	count = SparkMinimaxH3StagePackDitGlobalTensorCount();
	for (layer=0u; layer<SPARK_MINIMAX_H3_DIT_TOTAL_BLOCK_COUNT; layer++)
		count += SparkMinimaxH3StagePackDitBlockTensorCount(layer);
	return(count);
}

static inline uint32_t SparkMinimaxH3StagePackVideoTensorCount(void)
{
	return(9u + (SPARK_MINIMAX_H3_VIDEO_VAE_DECODER_LAYER_COUNT * 16u));
}

#define SPARK_MINIMAX_H3_STAGEPACK_AUDIO_GLOBAL_TENSOR_COUNT 11u
#define SPARK_MINIMAX_H3_STAGEPACK_AUDIO_RESBLOCK_TENSOR_COUNT 42u
#define SPARK_MINIMAX_H3_STAGEPACK_AUDIO_TENSOR_COUNT \
	(SPARK_MINIMAX_H3_STAGEPACK_AUDIO_GLOBAL_TENSOR_COUNT + \
	SPARK_MINIMAX_H3_AUDIO_VAE_RESBLOCK_STAGE_COUNT * \
	SPARK_MINIMAX_H3_AUDIO_VAE_RESBLOCKS_PER_STAGE * \
	SPARK_MINIMAX_H3_STAGEPACK_AUDIO_RESBLOCK_TENSOR_COUNT + \
	SPARK_MINIMAX_H3_AUDIO_VAE_DECODER_RATE_COUNT * 3u)

SPARK_MINIMAX_H3_STATIC_ASSERT(SPARK_MINIMAX_H3_STAGEPACK_AUDIO_TENSOR_COUNT == 914u,
	"h3 audio decode-path census must bind the measured shard enumeration: "
	"914 = dec_in 2 + conv_pre 3 + 21 resblocks x 42 + ups 7x3 + conv_post 2 + post act 4; "
	"the 1087-tensor shard adds the 22-tensor encode-only pre_block and 151 encoder-arm tensors");

static inline uint32_t SparkMinimaxH3StagePackAudioTensorCount(void)
{
	return(SPARK_MINIMAX_H3_STAGEPACK_AUDIO_TENSOR_COUNT);
}

static inline void SparkMinimaxH3StagePackExpectedGeometry(
	SparkMinimaxH3StagePackHeader *header)
{
	header->magic = SPARK_MINIMAX_H3_STAGEPACK_MAGIC;
	header->format_version = SPARK_MINIMAX_H3_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_MINIMAX_H3_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_MINIMAX_H3_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkMinimaxH3StagePackEncoderTensorCount() +
		SparkMinimaxH3StagePackDitTensorCount() +
		SparkMinimaxH3StagePackVideoTensorCount() +
		SparkMinimaxH3StagePackAudioTensorCount();
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
