#pragma once

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_dsv41_flash_model.h"
#include "sparkpipe/spark_weight_codec.h"
#define SPARK_FAMILY_CAMEL Dsv41Flash
#define SPARK_FAMILY_UPPER DSV41_FLASH
#define SPARK_FAMILY_LOWER dsv41_flash

#include "sparkpipe/family/spark_family.h"

#define SPARK_DSV41_FLASH_STAGEPACK_MAGIC UINT32_C(0x31413444)
#define SPARK_DSV41_FLASH_STAGEPACK_FORMAT_VERSION 2u
#define SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_DSV41_FLASH_STAGEPACK_ALIGNMENT_BYTES 256u
#define SPARK_DSV41_FLASH_STAGEPACK_MODEL_REVISION_BYTES 65u
#define SPARK_DSV41_FLASH_STAGEPACK_SHA256_BYTES 32u
#define SPARK_DSV41_FLASH_STAGEPACK_FLAG_DSPARK UINT32_C(0x00000001)
#define SPARK_DSV41_FLASH_STAGEPACK_KNOWN_FLAGS SPARK_DSV41_FLASH_STAGEPACK_FLAG_DSPARK
#define SPARK_DSV41_FLASH_STAGEPACK_FP8_BLOCK 32u
#define SPARK_DSV41_FLASH_STAGEPACK_CHECK_SHAPE (-1)
#define SPARK_DSV41_FLASH_STAGEPACK_CHECK_BYTES (-2)
#define SPARK_DSV41_FLASH_STAGEPACK_CHECK_GLOBAL (-3)
#define SPARK_DSV41_FLASH_STAGEPACK_CHECK_LAYER (-4)
#define SPARK_DSV41_FLASH_STAGEPACK_CHECK_OVERLAP (-5)
#define SPARK_DSV41_FLASH_STAGEPACK_CHECK_SELF_OVERLAP (-6)
#define SPARK_DSV41_FLASH_STAGEPACK_CHECK_GLOBALS (-7)
#define SPARK_DSV41_FLASH_STAGEPACK_CHECK_LAYER_SET (-8)

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
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W1 = 32,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W2 = 33,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W3 = 34,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W13_SLICE = 35,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W2_SLICE = 36,
	SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT = 37
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

static inline uint32_t SparkDsv41FlashStagePackKindExists(uint32_t tensor_kind)
{
	return(tensor_kind < SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT &&
		(tensor_kind <= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ROUTER_BIAS_VL ||
		tensor_kind >= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W1) ? 1u : 0u);
}

static inline uint32_t SparkDsv41FlashStagePackKindIsRouted(uint32_t tensor_kind)
{
	return(SparkDsv41FlashStagePackKindExists(tensor_kind) != 0u &&
		tensor_kind >= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ROUTER ? 1u : 0u);
}

static inline uint32_t SparkDsv41FlashStagePackKindIsExpert(uint32_t tensor_kind)
{
	return(tensor_kind == SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W13_SLICE ||
		tensor_kind == SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W2_SLICE ? 1u : 0u);
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

#include "sparkpipe/family/stagepack/spark_stagepack_shape_bf16.h"

#include "sparkpipe/family/stagepack/spark_stagepack_shape_f32.h"

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
		SparkDsv41FlashStagePackShapeF32(shape,1u,1u,SPARK_DSV41_FLASH_MODEL_ATTENTION_SINK_COUNT);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_O_A:
		SparkDsv41FlashStagePackShapeFp8(shape,1u,SPARK_DSV41_FLASH_MODEL_OUTPUT_A_DIMENSION / tp_degree,SPARK_DSV41_FLASH_MODEL_OUTPUT_GROUP_INPUT_DIMENSION);
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
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W13_SLICE:
		if ( tp_degree == 0u || (SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION % tp_degree) != 0u ||
			expert_weight_codec != SPARK_WEIGHT_CODEC_MXFP4_E2M1 )
			return(0u);
		SparkDsv41FlashStagePackShapeMxfp4(shape,SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT,
			2u * (SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION / tp_degree),SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION / 2u);
		return(1u);
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W2_SLICE:
		if ( tp_degree == 0u || (SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION % tp_degree) != 0u ||
			expert_weight_codec != SPARK_WEIGHT_CODEC_MXFP4_E2M1 )
			return(0u);
		SparkDsv41FlashStagePackShapeMxfp4(shape,SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT,
			SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION / tp_degree,SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION / 2u);
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
	if ( SparkDsv41FlashStagePackKindExists(tensor_kind) == 0u || SparkDsv41FlashStagePackKindIsGlobal(tensor_kind) )
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

static inline uint64_t SparkDsv41FlashStagePackExpectedPayloadBytes(const SparkDsv41FlashStagePackTensorShape *shape)
{
	uint64_t elements;
	elements = (uint64_t)shape->group_count * shape->rows * shape->columns;
	if ( shape->payload_type == SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_BF16 )
		return(elements * 2u);
	if ( shape->payload_type == SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_F32 )
		return(elements * 4u);
	if ( shape->payload_type == SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_PACKED_WEIGHT )
		return(elements);
	return(0u);
}

static inline uint64_t SparkDsv41FlashStagePackExpectedScaleBytes(const SparkDsv41FlashStagePackTensorShape *shape)
{
	uint32_t group;
	group = SparkWeightCodecScaleGroupSize(shape->weight_codec);
	if ( shape->payload_type != SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_PACKED_WEIGHT )
		return(0u);
	if ( shape->weight_codec == SPARK_WEIGHT_CODEC_MXFP4_E2M1 )
		return(group == 0u ? 0u : (uint64_t)shape->group_count * shape->rows * ((2u * (uint64_t)shape->columns + group - 1u) / group));
	return((uint64_t)shape->group_count * ((shape->rows + SPARK_DSV41_FLASH_STAGEPACK_FP8_BLOCK - 1u) / SPARK_DSV41_FLASH_STAGEPACK_FP8_BLOCK) *
		((shape->columns + SPARK_DSV41_FLASH_STAGEPACK_FP8_BLOCK - 1u) / SPARK_DSV41_FLASH_STAGEPACK_FP8_BLOCK));
}

static inline uint32_t SparkDsv41FlashStagePackRangesOverlap(uint64_t first,uint64_t first_bytes,uint64_t second,uint64_t second_bytes)
{
	return(first_bytes != 0u && second_bytes != 0u && first < second + second_bytes && second < first + first_bytes ? 1u : 0u);
}

static inline uint32_t SparkDsv41FlashStagePackRangeInside(uint64_t offset,uint64_t bytes,uint64_t first,uint64_t end)
{
	return(offset % SPARK_DSV41_FLASH_STAGEPACK_ALIGNMENT_BYTES == 0u && offset >= first && offset <= end && bytes <= end - offset ? 1u : 0u);
}

static inline int32_t SparkDsv41FlashStagePackDirectoryCheck(const SparkDsv41FlashStagePackHeader *header,const SparkDsv41FlashStagePackEntry *entries,uint32_t *at)
{
	uint64_t directory_end,global_seen,layer_seen[SPARK_DSV41_FLASH_MODEL_LAYER_COUNT],expected,bit,payload,scale;
	SparkDsv41FlashStagePackTensorShape shape;
	const SparkDsv41FlashStagePackEntry *entry,*right;
	uint32_t index,other,layer,kind;
	directory_end = header->directory_offset + (uint64_t)header->tensor_count * sizeof(SparkDsv41FlashStagePackEntry);
	global_seen = 0u;
	memset(layer_seen,0,sizeof(layer_seen));
	for ( index = 0u; index < header->tensor_count; index++ )
	{
		entry = &entries[index];
		*at = index;
		if ( entry->tensor_kind >= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT || entry->tensor_kind >= 64u ||
			SparkDsv41FlashStagePackExpectedShape(entry->tensor_kind,header->expert_weight_codec,header->tp_degree,&shape) == 0u ||
			entry->payload_type != shape.payload_type || entry->weight_codec != shape.weight_codec || entry->scale_encoding != shape.scale_encoding ||
			entry->group_count != shape.group_count || entry->rows != shape.rows || entry->columns != shape.columns )
			return(SPARK_DSV41_FLASH_STAGEPACK_CHECK_SHAPE);
		bit = UINT64_C(1) << entry->tensor_kind;
		payload = SparkDsv41FlashStagePackExpectedPayloadBytes(&shape);
		scale = SparkDsv41FlashStagePackExpectedScaleBytes(&shape);
		if ( payload == 0u || entry->payload_bytes != payload || entry->scale_bytes != scale ||
			SparkDsv41FlashStagePackRangeInside(entry->payload_offset,entry->payload_bytes,directory_end,header->file_bytes) == 0u ||
			(scale == 0u ? entry->scale_offset != 0u : SparkDsv41FlashStagePackRangeInside(entry->scale_offset,entry->scale_bytes,directory_end,header->file_bytes) == 0u) )
			return(SPARK_DSV41_FLASH_STAGEPACK_CHECK_BYTES);
		if ( SparkDsv41FlashStagePackKindIsGlobal(entry->tensor_kind) )
		{
			if ( entry->layer_index != SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER || (global_seen & bit) != 0u )
				return(SPARK_DSV41_FLASH_STAGEPACK_CHECK_GLOBAL);
			global_seen |= bit;
		}
		else
		{
			if ( entry->layer_index >= SPARK_DSV41_FLASH_MODEL_LAYER_COUNT || SparkDsv41FlashStagePackKindInLayer(entry->tensor_kind,entry->layer_index) == 0u ||
				(layer_seen[entry->layer_index] & bit) != 0u )
				return(SPARK_DSV41_FLASH_STAGEPACK_CHECK_LAYER);
			layer_seen[entry->layer_index] |= bit;
		}
		for ( other = 0u; other < index; other++ )
		{
			right = &entries[other];
			if ( SparkDsv41FlashStagePackRangesOverlap(entry->payload_offset,entry->payload_bytes,right->payload_offset,right->payload_bytes) != 0u ||
				SparkDsv41FlashStagePackRangesOverlap(entry->payload_offset,entry->payload_bytes,right->scale_offset,right->scale_bytes) != 0u ||
				SparkDsv41FlashStagePackRangesOverlap(entry->scale_offset,entry->scale_bytes,right->payload_offset,right->payload_bytes) != 0u ||
				SparkDsv41FlashStagePackRangesOverlap(entry->scale_offset,entry->scale_bytes,right->scale_offset,right->scale_bytes) != 0u )
				return(SPARK_DSV41_FLASH_STAGEPACK_CHECK_OVERLAP);
		}
		if ( SparkDsv41FlashStagePackRangesOverlap(entry->payload_offset,entry->payload_bytes,entry->scale_offset,entry->scale_bytes) != 0u )
			return(SPARK_DSV41_FLASH_STAGEPACK_CHECK_SELF_OVERLAP);
	}
	*at = 0u;
	if ( global_seen != ((UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EMBEDDING) | (UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FINAL_NORM) |
		(UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD)) )
		return(SPARK_DSV41_FLASH_STAGEPACK_CHECK_GLOBALS);
	for ( layer = 0u; layer < SPARK_DSV41_FLASH_MODEL_LAYER_COUNT; layer++ )
	{
		expected = 0u;
		for ( kind = 0u; kind < SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT; kind++ )
			if ( SparkDsv41FlashStagePackKindInLayer(kind,layer) != 0u )
				expected |= UINT64_C(1) << kind;
		*at = layer;
		if ( layer_seen[layer] != expected )
			return(SPARK_DSV41_FLASH_STAGEPACK_CHECK_LAYER_SET);
	}
	*at = 0u;
	return(0);
}

static inline uint32_t SparkDsv41FlashStagePackTpShardsRows(uint32_t tensor_kind)
{
	switch ( tensor_kind )
	{
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_Q_B:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_O_A:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_O_B:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_Q_B:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_WEIGHTS_PROJ:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W13_SLICE:
	case SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W2_SLICE:
		return(1u);
	default:
		return(0u);
	}
}

#include "sparkpipe/family/stagepack/spark_stagepack_kind_is_hc.h"
