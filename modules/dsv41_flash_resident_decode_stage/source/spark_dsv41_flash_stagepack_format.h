#pragma once

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_dsv41_flash_model.h"
#include "sparkpipe/spark_weight_codec.h"

#define SPARK_DSV41_FLASH_STAGEPACK_MAGIC UINT32_C(0x31413444)
#define SPARK_DSV41_FLASH_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_DSV41_FLASH_STAGEPACK_ALIGNMENT_BYTES 256u
#define SPARK_DSV41_FLASH_STAGEPACK_MODEL_REVISION_BYTES 65u
#define SPARK_DSV41_FLASH_STAGEPACK_SHA256_BYTES 32u
#define SPARK_DSV41_FLASH_STAGEPACK_FLAG_DSPARK UINT32_C(0x00000001)
#define SPARK_DSV41_FLASH_STAGEPACK_KNOWN_FLAGS SPARK_DSV41_FLASH_STAGEPACK_FLAG_DSPARK

typedef enum SparkDsv41FlashStagePackPayloadType
{
	SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_BF16 = 1,
	SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_F32 = 2,
	SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_U32 = 3,
	SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_PACKED_WEIGHT = 4
} SparkDsv41FlashStagePackPayloadType;

typedef enum SparkDsv41FlashStagePackTensorKind
{
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ATTN_NORM = 3,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FFN_NORM = 4,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_Q_A = 5,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_Q_B = 6,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KV_A = 7,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_Q_NORM = 8,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KV_NORM = 9,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ATTN_SINK = 10,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_O_A = 11,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_O_B = 12,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_Q_B = 13,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_WK = 14,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_WEIGHTS_PROJ = 15,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM = 16,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_WKV = 17,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_WGATE = 18,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_NORM = 19,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_ATTN_FN = 20,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_ATTN_BASE = 21,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_ATTN_SCALE = 22,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_FFN_FN = 23,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_FFN_BASE = 24,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_FFN_SCALE = 25,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ROUTER = 26,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ROUTER_BIAS = 27,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ROUTER_BIAS_VL = 28,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W1 = 29,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W2 = 30,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W3 = 31,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W1 = 32,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W2 = 33,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W3 = 34,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT = 35
} SparkDsv41FlashStagePackTensorKind;

#pragma pack(1)

typedef struct SparkDsv41FlashStagePackHeader
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t header_bytes;
	uint32_t directory_entry_bytes;
	uint32_t codec_abi_version;
	uint32_t flags;
	uint32_t tensor_count;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t total_layer_count;
	uint32_t hidden_dimension;
	uint32_t vocab_count;
	uint32_t routed_expert_count;
	uint32_t linear_weight_codec;
	uint32_t expert_weight_codec;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t reserved0;
	uint64_t directory_offset;
	uint64_t file_bytes;
	char model_revision[SPARK_DSV41_FLASH_STAGEPACK_MODEL_REVISION_BYTES];
	uint8_t contract_sha256[SPARK_DSV41_FLASH_STAGEPACK_SHA256_BYTES];
	uint8_t source_config_sha256[SPARK_DSV41_FLASH_STAGEPACK_SHA256_BYTES];
	uint8_t pack_recipe_sha256[SPARK_DSV41_FLASH_STAGEPACK_SHA256_BYTES];
} SparkDsv41FlashStagePackHeader;

typedef struct SparkDsv41FlashStagePackEntry
{
	uint32_t tensor_kind;
	uint32_t layer_index;
	uint32_t payload_type;
	uint32_t weight_codec;
	uint32_t scale_encoding;
	uint32_t group_count;
	uint32_t rows;
	uint32_t columns;
	uint64_t payload_offset;
	uint64_t payload_bytes;
	uint64_t scale_offset;
	uint64_t scale_bytes;
} SparkDsv41FlashStagePackEntry;

typedef struct SparkDsv41FlashStagePackTensorShape
{
	uint32_t payload_type;
	uint32_t weight_codec;
	uint32_t scale_encoding;
	uint32_t group_count;
	uint32_t rows;
	uint32_t columns;
} SparkDsv41FlashStagePackTensorShape;

#pragma pack()

#define SPARK_DSV41_FLASH_STAGEPACK_HEADER_BYTES ((uint32_t)sizeof(SparkDsv41FlashStagePackHeader))
#define SPARK_DSV41_FLASH_STAGEPACK_ENTRY_BYTES ((uint32_t)sizeof(SparkDsv41FlashStagePackEntry))

typedef struct SparkDsv41FlashStagePackLayerMaskContext
{
	uint32_t expert_weight_codec;
	uint32_t tp_degree;
} SparkDsv41FlashStagePackLayerMaskContext;

static inline uint32_t SparkDsv41FlashStagePackKindIsGlobal(uint32_t tensor_kind)
{
	return(tensor_kind <= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD ? 1u : 0u);
}

static inline uint32_t SparkDsv41FlashStagePackKindIsIndexer(uint32_t tensor_kind)
{
	return(tensor_kind >= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_Q_B &&
		tensor_kind <= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM ? 1u : 0u);
}

static inline uint32_t SparkDsv41FlashStagePackKindIsCompressor(uint32_t tensor_kind)
{
	return(tensor_kind >= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_WKV &&
		tensor_kind <= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_NORM ? 1u : 0u);
}

static inline uint32_t SparkDsv41FlashStagePackKindIsHc(uint32_t tensor_kind)
{
	return(tensor_kind >= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_ATTN_FN &&
		tensor_kind <= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_FFN_SCALE ? 1u : 0u);
}

static inline uint32_t SparkDsv41FlashStagePackKindIsRouted(uint32_t tensor_kind)
{
	return(tensor_kind >= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ROUTER &&
		tensor_kind <= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W3 ? 1u : 0u);
}

static inline uint32_t SparkDsv41FlashStagePackKindIsExpert(uint32_t tensor_kind)
{
	return(tensor_kind >= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W1 &&
		tensor_kind <= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W3 ? 1u : 0u);
}

static inline uint32_t SparkDsv41FlashStagePackKindUsesIndexSourceLayer(uint32_t layer_index)
{
	return(layer_index == 2u || layer_index == 8u || layer_index == 14u ||
		layer_index == 20u || layer_index == 24u || layer_index == 28u ||
		layer_index == 32u || layer_index == 36u ? 1u : 0u);
}

static inline uint32_t SparkDsv41FlashStagePackKindUsesKvSourceLayer(uint32_t layer_index)
{
	return(layer_index == 2u || layer_index == 8u || layer_index == 14u ||
		layer_index == 20u ? 1u : 0u);
}

static inline uint32_t SparkDsv41FlashStagePackKindUsesCompressGateLayer(uint32_t layer_index)
{
	return(layer_index == 2u || layer_index == 8u || layer_index == 14u ? 1u : 0u);
}

static inline void SparkDsv41FlashStagePackShapeBf16(SparkDsv41FlashStagePackTensorShape *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
	shape->payload_type = SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_BF16;
	shape->weight_codec = SPARK_WEIGHT_CODEC_BF16;
	shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
	shape->group_count = groups;
	shape->rows = rows;
	shape->columns = columns;
}

static inline void SparkDsv41FlashStagePackShapeF32(SparkDsv41FlashStagePackTensorShape *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
	shape->payload_type = SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_F32;
	shape->weight_codec = SPARK_WEIGHT_CODEC_NONE;
	shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
	shape->group_count = groups;
	shape->rows = rows;
	shape->columns = columns;
}

static inline void SparkDsv41FlashStagePackShapeFp8(SparkDsv41FlashStagePackTensorShape *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
	shape->payload_type = SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_PACKED_WEIGHT;
	shape->weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_E8M0;
	shape->group_count = groups;
	shape->rows = rows;
	shape->columns = columns;
}

static inline void SparkDsv41FlashStagePackShapeMxfp4(SparkDsv41FlashStagePackTensorShape *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
	shape->payload_type = SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_PACKED_WEIGHT;
	shape->weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1;
	shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_E8M0;
	shape->group_count = groups;
	shape->rows = rows;
	shape->columns = columns;
}

static inline uint32_t SparkDsv41FlashStagePackExpectedShape(uint32_t tensor_kind,uint32_t expert_weight_codec,uint32_t tp_degree,SparkDsv41FlashStagePackTensorShape *shape)
{
	switch ( tensor_kind )
	{
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EMBEDDING:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,SPARK_DSV41_FLASH_MODEL_OUTPUT_VOCAB_COUNT / tp_degree,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FINAL_NORM:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,1u,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,SPARK_DSV41_FLASH_MODEL_OUTPUT_VOCAB_COUNT / tp_degree,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ATTN_NORM:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FFN_NORM:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,1u,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_Q_A:
		SparkDsv41FlashStagePackShapeFp8(shape,1u,SPARK_DSV41_FLASH_MODEL_QUERY_LORA_RANK,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_Q_B:
		SparkDsv41FlashStagePackShapeFp8(shape,1u,SPARK_DSV41_FLASH_MODEL_QUERY_B_DIMENSION / tp_degree,SPARK_DSV41_FLASH_MODEL_QUERY_LORA_RANK);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KV_A:
		SparkDsv41FlashStagePackShapeFp8(shape,1u,SPARK_DSV41_FLASH_MODEL_KV_LATENT_DIMENSION,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_Q_NORM:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,1u,SPARK_DSV41_FLASH_MODEL_QUERY_LORA_RANK);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KV_NORM:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,1u,SPARK_DSV41_FLASH_MODEL_KV_LATENT_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ATTN_SINK:
		SparkDsv41FlashStagePackShapeF32(shape,1u,1u,SPARK_DSV41_FLASH_MODEL_ATTENTION_SINK_COUNT / tp_degree);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_O_A:
		SparkDsv41FlashStagePackShapeFp8(shape,1u,SPARK_DSV41_FLASH_MODEL_OUTPUT_A_DIMENSION / tp_degree,SPARK_DSV41_FLASH_MODEL_ATTENTION_PROJECTION_DIMENSION / tp_degree);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_O_B:
		SparkDsv41FlashStagePackShapeFp8(shape,1u,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION,SPARK_DSV41_FLASH_MODEL_OUTPUT_A_DIMENSION / tp_degree);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_Q_B:
		SparkDsv41FlashStagePackShapeFp8(shape,1u,SPARK_DSV41_FLASH_MODEL_INDEX_QUERY_DIMENSION / tp_degree,SPARK_DSV41_FLASH_MODEL_QUERY_LORA_RANK);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_WK:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,SPARK_DSV41_FLASH_MODEL_INDEX_HEAD_DIMENSION,SPARK_DSV41_FLASH_MODEL_KV_LATENT_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_WEIGHTS_PROJ:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,SPARK_DSV41_FLASH_MODEL_INDEX_HEAD_COUNT / tp_degree,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,1u,SPARK_DSV41_FLASH_MODEL_INDEX_HEAD_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_WKV:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_WGATE:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,SPARK_DSV41_FLASH_MODEL_KV_LATENT_DIMENSION,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_NORM:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,1u,SPARK_DSV41_FLASH_MODEL_KV_LATENT_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_ATTN_FN:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_FFN_FN:
		SparkDsv41FlashStagePackShapeF32(shape,1u,SPARK_DSV41_FLASH_MODEL_HC_COEFFICIENT_COUNT,SPARK_DSV41_FLASH_MODEL_HC_STREAM_HIDDEN_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_ATTN_BASE:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_FFN_BASE:
		SparkDsv41FlashStagePackShapeF32(shape,1u,1u,SPARK_DSV41_FLASH_MODEL_HC_COEFFICIENT_COUNT);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_ATTN_SCALE:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_FFN_SCALE:
		SparkDsv41FlashStagePackShapeF32(shape,1u,1u,3u);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ROUTER:
		SparkDsv41FlashStagePackShapeBf16(shape,1u,SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ROUTER_BIAS:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ROUTER_BIAS_VL:
		SparkDsv41FlashStagePackShapeF32(shape,1u,1u,SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W1:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W3:
		if ( tp_degree == 0u || (SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT % tp_degree) != 0u )
			return(0u);
		if ( expert_weight_codec == SPARK_WEIGHT_CODEC_MXFP4_E2M1 )
			SparkDsv41FlashStagePackShapeMxfp4(shape,SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT / tp_degree,SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION / 2u);
		else if ( expert_weight_codec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
			SparkDsv41FlashStagePackShapeFp8(shape,SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT / tp_degree,SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION);
		else
			return(0u);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W2:
		if ( tp_degree == 0u || (SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT % tp_degree) != 0u )
			return(0u);
		if ( expert_weight_codec == SPARK_WEIGHT_CODEC_MXFP4_E2M1 )
			SparkDsv41FlashStagePackShapeMxfp4(shape,SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT / tp_degree,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION,SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION / 2u);
		else if ( expert_weight_codec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
			SparkDsv41FlashStagePackShapeFp8(shape,SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT / tp_degree,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION,SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION);
		else
			return(0u);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W1:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W3:
		SparkDsv41FlashStagePackShapeFp8(shape,1u,SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W2:
		SparkDsv41FlashStagePackShapeFp8(shape,1u,SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION,SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION);
		return(1u);
	default:
		return(0u);
	}
}

static inline uint32_t SparkDsv41FlashStagePackKindInLayer(uint32_t tensor_kind,uint32_t layer_index)
{
	if ( SparkDsv41FlashStagePackKindIsGlobal(tensor_kind) )
		return(0u);
	if ( SPARK_DSV41_FLASH_MODEL_LAYER_IS_SWA_ONLY(layer_index) &&
		SparkDsv41FlashStagePackKindIsIndexer(tensor_kind) )
		return(0u);
	if ( SparkDsv41FlashStagePackKindIsIndexer(tensor_kind) )
	{
		if ( tensor_kind == SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_WK ||
			tensor_kind == SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM )
			return(SparkDsv41FlashStagePackKindUsesKvSourceLayer(layer_index));
		return(SparkDsv41FlashStagePackKindUsesIndexSourceLayer(layer_index));
	}
	if ( SparkDsv41FlashStagePackKindIsCompressor(tensor_kind) )
	{
		if ( tensor_kind == SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_WGATE )
			return(SparkDsv41FlashStagePackKindUsesCompressGateLayer(layer_index));
		return(SparkDsv41FlashStagePackKindUsesKvSourceLayer(layer_index));
	}
	return(1u);
}

static inline uint32_t SparkDsv41FlashStagePackTpShardsRows(uint32_t tensor_kind)
{
	switch ( tensor_kind )
	{
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_Q_B:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ATTN_SINK:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_O_A:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_O_B:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_Q_B:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_WEIGHTS_PROJ:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W1:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W2:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W3:
		return(1u);
	default:
		return(0u);
	}
}
