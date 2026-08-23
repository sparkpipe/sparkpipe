#pragma once

#include <stdint.h>
#include <string.h>

#include "runtime/spark_stagepack_reader.h"
#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_weight_codec.h"

#define SPARK_GLM52_STAGEPACK_MAGIC UINT32_C(0x32534c47)
#define SPARK_GLM52_STAGEPACK_FORMAT_VERSION 3u
#define SPARK_GLM52_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES 256u
#define SPARK_GLM52_STAGEPACK_MODEL_REVISION_BYTES 65u
#define SPARK_GLM52_STAGEPACK_SHA256_BYTES 32u
#define SPARK_GLM52_STAGEPACK_FLAG_MTP UINT32_C(0x00000001)
#define SPARK_GLM52_STAGEPACK_KNOWN_FLAGS SPARK_GLM52_STAGEPACK_FLAG_MTP

typedef enum SparkGlm52StagePackPayloadType
{
	SPARK_GLM52_STAGEPACK_PAYLOAD_BF16 = SPARK_STAGE_PACK_PAYLOAD_BF16,
	SPARK_GLM52_STAGEPACK_PAYLOAD_F32 = SPARK_STAGE_PACK_PAYLOAD_F32,
	SPARK_GLM52_STAGEPACK_PAYLOAD_U32 = SPARK_STAGE_PACK_PAYLOAD_U32,
	SPARK_GLM52_STAGEPACK_PAYLOAD_PACKED_WEIGHT = SPARK_STAGE_PACK_PAYLOAD_PACKED_WEIGHT
} SparkGlm52StagePackPayloadType;

typedef enum SparkGlm52StagePackTensorKind
{
	SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM = 3,
	SPARK_GLM52_STAGEPACK_TENSOR_Q_A = 4,
	SPARK_GLM52_STAGEPACK_TENSOR_Q_A_NORM = 5,
	SPARK_GLM52_STAGEPACK_TENSOR_Q_B = 6,
	SPARK_GLM52_STAGEPACK_TENSOR_KV_A = 7,
	SPARK_GLM52_STAGEPACK_TENSOR_KV_A_NORM = 8,
	SPARK_GLM52_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED = 9,
	SPARK_GLM52_STAGEPACK_TENSOR_KV_B_VALUE = 10,
	SPARK_GLM52_STAGEPACK_TENSOR_ATTN_OUTPUT = 11,
	SPARK_GLM52_STAGEPACK_TENSOR_POST_ATTN_NORM = 12,
	SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q = 13,
	SPARK_GLM52_STAGEPACK_TENSOR_INDEX_K = 14,
	SPARK_GLM52_STAGEPACK_TENSOR_INDEX_HEAD = 15,
	SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT = 16,
	SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_BIAS = 17,
	SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP = 18,
	SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN = 19,
	SPARK_GLM52_STAGEPACK_TENSOR_ROUTER = 20,
	SPARK_GLM52_STAGEPACK_TENSOR_ROUTER_CORRECTION = 21,
	SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE = 22,
	SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN = 23,
	SPARK_GLM52_STAGEPACK_TENSOR_SHARED_GATE_UP = 24,
	SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN = 25,
	SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT = 26
} SparkGlm52StagePackTensorKind;

typedef struct SparkGlm52StagePackHeader
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
	uint32_t kv_cache_codec;
	uint32_t reserved0;
	uint32_t reserved1;
	uint64_t directory_offset;
	uint64_t file_bytes;
	char model_revision[SPARK_GLM52_STAGEPACK_MODEL_REVISION_BYTES];
	uint8_t contract_sha256[SPARK_GLM52_STAGEPACK_SHA256_BYTES];
	uint8_t source_config_sha256[SPARK_GLM52_STAGEPACK_SHA256_BYTES];
	uint8_t pack_recipe_sha256[SPARK_GLM52_STAGEPACK_SHA256_BYTES];
} SparkGlm52StagePackHeader;

typedef struct SparkGlm52StagePackEntry
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
} SparkGlm52StagePackEntry;

/* The normalized reader shape, under this family's historical name. */
typedef SparkStagePackShape SparkGlm52StagePackTensorShape;

#define SPARK_GLM52_STAGEPACK_HEADER_BYTES ((uint32_t)sizeof(SparkGlm52StagePackHeader))
#define SPARK_GLM52_STAGEPACK_ENTRY_BYTES ((uint32_t)sizeof(SparkGlm52StagePackEntry))

static inline uint32_t SparkGlm52StagePackLayerIsDense(uint32_t layer_index)
{
	return(layer_index < SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER ? 1u : 0u);
}

static inline uint32_t SparkGlm52StagePackLayerHasFullIndexer(uint32_t layer_index)
{
	return(SPARK_GLM52_MODEL_LAYER_HAS_FULL_INDEXER(layer_index));
}

static inline uint32_t SparkGlm52StagePackKindIsGlobal(uint32_t tensor_kind)
{
	return(tensor_kind <= SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD ? 1u : 0u);
}

static inline uint32_t SparkGlm52StagePackKindIsIndexer(uint32_t tensor_kind)
{
	return(tensor_kind >= SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q && tensor_kind <= SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_BIAS ? 1u : 0u);
}

static inline uint32_t SparkGlm52StagePackKindIsDense(uint32_t tensor_kind)
{
	return(tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP || tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN ? 1u : 0u);
}

static inline uint32_t SparkGlm52StagePackKindIsRouted(uint32_t tensor_kind)
{
	return(tensor_kind >= SPARK_GLM52_STAGEPACK_TENSOR_ROUTER && tensor_kind <= SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN ? 1u : 0u);
}

/* TP sharding policy: which kinds are row- or column-sharded across ranks.
 * Must match tools/glm52_resident_stagepack.py exactly; the mechanics live
 * in the shared reader (SparkStagePackApplyShard). */
static inline uint32_t SparkGlm52StagePackTpShardPolicy(uint32_t tensor_kind)
{
	switch ( tensor_kind )
	{
	case SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD:
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_B:
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP:
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE:
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_GATE_UP:
		return(SPARK_STAGE_PACK_SHARD_ROWS);
	case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_OUTPUT:
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN:
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN:
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN:
		return(SPARK_STAGE_PACK_SHARD_COLUMNS);
	default:
		return(SPARK_STAGE_PACK_SHARD_NONE);
	}
}

/* Pack header TP identity: reserved0 = tp_degree, reserved1 = tp_rank. */
static inline uint32_t SparkGlm52StagePackHeaderTpDegree(const SparkGlm52StagePackHeader *header)
{
	return(header != 0 ? header->reserved0 : 0u);
}

static inline uint32_t SparkGlm52StagePackHeaderTpRank(const SparkGlm52StagePackHeader *header)
{
	return(header != 0 ? header->reserved1 : 0u);
}

static inline int32_t SparkGlm52StagePackExpectedShape(uint32_t tensor_kind,uint32_t layer_index,uint32_t expert_codec,uint32_t tp_degree,SparkGlm52StagePackTensorShape *shape)
{
	uint32_t global;
	if ( shape == 0 || tensor_kind >= SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT || tp_degree == 0u )
		return(-1);
	memset(shape,0,sizeof(*shape));
	global = SparkGlm52StagePackKindIsGlobal(tensor_kind);
	if ( (global != 0u && layer_index != SPARK_GLM52_STAGEPACK_GLOBAL_LAYER) || (global == 0u && layer_index >= SPARK_GLM52_MODEL_LAYER_COUNT) )
		return(-2);
	if ( SparkGlm52StagePackKindIsIndexer(tensor_kind) != 0u && SparkGlm52StagePackLayerHasFullIndexer(layer_index) == 0u )
		return(-3);
	if ( SparkGlm52StagePackKindIsDense(tensor_kind) != SparkGlm52StagePackLayerIsDense(layer_index) && (SparkGlm52StagePackKindIsDense(tensor_kind) != 0u || SparkGlm52StagePackKindIsRouted(tensor_kind) != 0u) )
		return(-4);
	switch ( tensor_kind )
	{
	case SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM:
	case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM:
	case SPARK_GLM52_STAGEPACK_TENSOR_POST_ATTN_NORM: SparkStagePackShapeBf16(shape,1u,1u,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_A: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_QUERY_A_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_A_NORM: SparkStagePackShapeBf16(shape,1u,1u,SPARK_GLM52_MODEL_QUERY_A_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_B: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_QK_HEAD_DIMENSION,SPARK_GLM52_MODEL_QUERY_A_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_A: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_A_NORM: SparkStagePackShapeBf16(shape,1u,1u,SPARK_GLM52_MODEL_LATENT_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED: SparkStagePackShapeBf16(shape,SPARK_GLM52_MODEL_HEAD_COUNT,SPARK_GLM52_MODEL_LATENT_DIMENSION,SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_VALUE: SparkStagePackShapeBf16(shape,SPARK_GLM52_MODEL_HEAD_COUNT,SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION,SPARK_GLM52_MODEL_LATENT_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_DSA_INDEX_QUERY_DIMENSION,SPARK_GLM52_MODEL_QUERY_A_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_K: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_HEAD: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT:
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_BIAS: SparkStagePackShapeBf16(shape,1u,1u,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP: SparkStagePackShapeBf16(shape,1u,2u * SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER_CORRECTION: SparkStagePackShapeWords(shape,SPARK_GLM52_STAGEPACK_PAYLOAD_F32,1u,1u,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE:
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN:
		if ( expert_codec < SPARK_WEIGHT_CODEC_INT6 || expert_codec > SPARK_WEIGHT_CODEC_MXFP4_E2M1 )
			return(-5);
		shape->payload_type = SPARK_GLM52_STAGEPACK_PAYLOAD_PACKED_WEIGHT;
		shape->weight_codec = expert_codec;
		shape->scale_encoding = SparkWeightCodecScaleEncoding(expert_codec);
		shape->group_count = SPARK_GLM52_MODEL_MOE_EXPERT_COUNT;
		shape->rows = tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE ? 2u * SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION : SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
		shape->columns = tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE ? SPARK_GLM52_MODEL_HIDDEN_DIMENSION : SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION;
		break;
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_GATE_UP: SparkStagePackShapeBf16(shape,1u,2u * SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION); break;
	default: return(-6);
	}
	if ( SparkStagePackApplyShard(shape,SparkGlm52StagePackTpShardPolicy(tensor_kind),tp_degree) != 0 )
		return(-7);
	return(0);
}

