#ifndef SPARKPIPE_SPARK_GLM5_NEXT_STAGEPACK_H
#define SPARKPIPE_SPARK_GLM5_NEXT_STAGEPACK_H

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_weight_codec.h"

#define GLM5_NEXT_PACK_MAGIC UINT32_C(0x33584C47)
#define GLM5_NEXT_PACK_VERSION 1u
#define GLM5_NEXT_PACK_GLOBAL_LAYER UINT32_MAX
#define GLM5_NEXT_PACK_ALIGNMENT 256u
#define GLM5_NEXT_PACK_FLAG_MTP UINT32_C(0x00000001)
#define GLM5_NEXT_PACK_MAX_TENSORS 2048u

enum
{
	GLM5_NEXT_PAYLOAD_BF16 = 1,
	GLM5_NEXT_PAYLOAD_F32 = 2,
	GLM5_NEXT_PAYLOAD_U32 = 3,
	GLM5_NEXT_PAYLOAD_PACKED = 4
};

enum
{
	GLM5_NEXT_T_EMBEDDING = 0,
	GLM5_NEXT_T_FINAL_NORM,
	GLM5_NEXT_T_LM_HEAD,
	GLM5_NEXT_T_ATTN_NORM,
	GLM5_NEXT_T_Q_A,
	GLM5_NEXT_T_Q_A_NORM,
	GLM5_NEXT_T_Q_B,
	GLM5_NEXT_T_KV_A,
	GLM5_NEXT_T_KV_A_NORM,
	GLM5_NEXT_T_KV_B_KEY_T,
	GLM5_NEXT_T_KV_B_VALUE,
	GLM5_NEXT_T_ATTN_OUTPUT,
	GLM5_NEXT_T_POST_ATTN_NORM,
	GLM5_NEXT_T_INDEX_Q,
	GLM5_NEXT_T_INDEX_K,
	GLM5_NEXT_T_INDEX_HEAD,
	GLM5_NEXT_T_INDEX_NORM_WEIGHT,
	GLM5_NEXT_T_INDEX_NORM_BIAS,
	GLM5_NEXT_T_DENSE_GATE_UP,
	GLM5_NEXT_T_DENSE_DOWN,
	GLM5_NEXT_T_ROUTER,
	GLM5_NEXT_T_ROUTER_CORRECTION,
	GLM5_NEXT_T_EXPERT_UP_GATE,
	GLM5_NEXT_T_EXPERT_DOWN,
	GLM5_NEXT_T_SHARED_GATE_UP,
	GLM5_NEXT_T_SHARED_DOWN,
	GLM5_NEXT_T_KDA_QKV_BETA,
	GLM5_NEXT_T_KDA_DECAY_GATE_DOWN,
	GLM5_NEXT_T_KDA_DECAY_UP,
	GLM5_NEXT_T_KDA_GATE_UP,
	GLM5_NEXT_T_KDA_Q_CONV,
	GLM5_NEXT_T_KDA_K_CONV,
	GLM5_NEXT_T_KDA_V_CONV,
	GLM5_NEXT_T_KDA_DECAY_BIAS,
	GLM5_NEXT_T_KDA_HEAD_LOG_SCALE,
	GLM5_NEXT_T_KDA_OUT_NORM,
	GLM5_NEXT_T_KDA_OUT,
	GLM5_NEXT_T_HC_ATTN_FN,
	GLM5_NEXT_T_HC_ATTN_BASE,
	GLM5_NEXT_T_HC_ATTN_SCALE,
	GLM5_NEXT_T_HC_FFN_FN,
	GLM5_NEXT_T_HC_FFN_BASE,
	GLM5_NEXT_T_HC_FFN_SCALE,
	GLM5_NEXT_T_INDEX_COMPRESS_APE,
	GLM5_NEXT_T_INDEX_COMPRESS_GATE,
	GLM5_NEXT_T_MTP_EH_PROJ,
	GLM5_NEXT_T_MTP_ENORM,
	GLM5_NEXT_T_MTP_HNORM,
	GLM5_NEXT_T_MTP_SHARED_NORM,
	GLM5_NEXT_T_COUNT
};

typedef struct Glm5NextPackHeader
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
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint64_t directory_offset;
	uint64_t file_bytes;
	char model_revision[65];
	uint8_t contract_sha256[32];
	uint8_t source_config_sha256[32];
	uint8_t pack_recipe_sha256[32];
} Glm5NextPackHeader;

typedef struct Glm5NextPackEntry
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
} Glm5NextPackEntry;

typedef struct Glm5NextPackShape
{
	uint32_t payload_type;
	uint32_t groups;
	uint32_t rows;
	uint32_t columns;
	uint32_t split;
} Glm5NextPackShape;

#define GLM5_NEXT_SPLIT_NONE 0u
#define GLM5_NEXT_SPLIT_ROWS 1u
#define GLM5_NEXT_SPLIT_COLUMNS 2u

static inline uint32_t Glm5NextPackLayerIsDsa(uint32_t layer)
{
	return layer < SPARK_GLM5_NEXT_MODEL_LAYER_COUNT && !SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer) ? 1u : 0u;
}

static inline uint32_t Glm5NextPackLayerIsKda(uint32_t layer)
{
	return layer < SPARK_GLM5_NEXT_MODEL_LAYER_COUNT && SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer) ? 1u : 0u;
}

static inline uint32_t Glm5NextPackKindAllowed(uint32_t kind, uint32_t layer)
{
	const uint32_t mtp = layer == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX;
	const uint32_t dense = layer < SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER;
	if ( kind <= GLM5_NEXT_T_LM_HEAD )
		return layer == GLM5_NEXT_PACK_GLOBAL_LAYER;
	if ( layer > SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		return 0u;
	if ( (kind >= GLM5_NEXT_T_Q_A && kind <= GLM5_NEXT_T_ATTN_OUTPUT) ||
		(kind >= GLM5_NEXT_T_INDEX_Q && kind <= GLM5_NEXT_T_INDEX_NORM_BIAS) ||
		kind == GLM5_NEXT_T_INDEX_COMPRESS_APE || kind == GLM5_NEXT_T_INDEX_COMPRESS_GATE )
		return mtp || Glm5NextPackLayerIsDsa(layer);
	if ( kind >= GLM5_NEXT_T_KDA_QKV_BETA && kind <= GLM5_NEXT_T_KDA_OUT )
		return Glm5NextPackLayerIsKda(layer);
	if ( kind >= GLM5_NEXT_T_HC_ATTN_FN && kind <= GLM5_NEXT_T_HC_FFN_SCALE )
		return !mtp;
	if ( kind >= GLM5_NEXT_T_MTP_EH_PROJ && kind <= GLM5_NEXT_T_MTP_SHARED_NORM )
		return mtp;
	if ( kind == GLM5_NEXT_T_DENSE_GATE_UP || kind == GLM5_NEXT_T_DENSE_DOWN )
		return dense && !mtp;
	if ( kind >= GLM5_NEXT_T_ROUTER && kind <= GLM5_NEXT_T_SHARED_DOWN )
		return !dense;
	return 1u;
}

static inline uint32_t Glm5NextPackShapeOf(uint32_t kind, Glm5NextPackShape *shape)
{
	const uint32_t H = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION, V = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT;
	const uint32_t QA = SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION, L = SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION;
	const uint32_t K = SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION, R = SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK;
	const uint32_t E = SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT, I = SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION;
	const uint32_t D = SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION, X = SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION;
	const uint32_t M = SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION;
	static const uint32_t B = GLM5_NEXT_PAYLOAD_BF16, F = GLM5_NEXT_PAYLOAD_F32, P = GLM5_NEXT_PAYLOAD_PACKED;
	const uint32_t table[GLM5_NEXT_T_COUNT][5] =
	{
		{ B, 1u, V, H, GLM5_NEXT_SPLIT_ROWS },
		{ B, 1u, 1u, H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, V, H, GLM5_NEXT_SPLIT_ROWS },
		{ B, 1u, 1u, H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, QA, H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, 1u, QA, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, SPARK_GLM5_NEXT_MODEL_MLA_QUERY_B_DIMENSION, QA, GLM5_NEXT_SPLIT_ROWS },
		{ B, 1u, SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION, H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, 1u, L, GLM5_NEXT_SPLIT_NONE },
		{ B, SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT, L, SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION, GLM5_NEXT_SPLIT_NONE },
		{ B, SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT, SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION, L, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, H, SPARK_GLM5_NEXT_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION, GLM5_NEXT_SPLIT_COLUMNS },
		{ B, 1u, 1u, H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION, QA, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, X, H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT, H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, 1u, X, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, 1u, X, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, 2u * D, H, GLM5_NEXT_SPLIT_ROWS },
		{ B, 1u, H, D, GLM5_NEXT_SPLIT_COLUMNS },
		{ B, 1u, E, H, GLM5_NEXT_SPLIT_NONE },
		{ F, 1u, 1u, E, GLM5_NEXT_SPLIT_NONE },
		{ P, E, 2u * I, H, GLM5_NEXT_SPLIT_ROWS },
		{ P, E, H, I, GLM5_NEXT_SPLIT_COLUMNS },
		{ B, 1u, 2u * I, H, GLM5_NEXT_SPLIT_ROWS },
		{ B, 1u, H, I, GLM5_NEXT_SPLIT_COLUMNS },
		{ B, 1u, 3u * K + SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT, H, GLM5_NEXT_SPLIT_ROWS },
		{ B, 1u, 2u * R, H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, K, R, GLM5_NEXT_SPLIT_ROWS },
		{ B, 1u, K, R, GLM5_NEXT_SPLIT_ROWS },
		{ B, 1u, K, SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL, GLM5_NEXT_SPLIT_ROWS },
		{ B, 1u, K, SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL, GLM5_NEXT_SPLIT_ROWS },
		{ B, 1u, K, SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL, GLM5_NEXT_SPLIT_ROWS },
		{ F, 1u, 1u, K, GLM5_NEXT_SPLIT_COLUMNS },
		{ F, 1u, 1u, SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT, GLM5_NEXT_SPLIT_COLUMNS },
		{ F, 1u, 1u, SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, H, K, GLM5_NEXT_SPLIT_COLUMNS },
		{ F, 1u, M, SPARK_GLM5_NEXT_MODEL_HC_FN_COLUMNS, GLM5_NEXT_SPLIT_NONE },
		{ F, 1u, 1u, M, GLM5_NEXT_SPLIT_NONE },
		{ F, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT, GLM5_NEXT_SPLIT_NONE },
		{ F, 1u, M, SPARK_GLM5_NEXT_MODEL_HC_FN_COLUMNS, GLM5_NEXT_SPLIT_NONE },
		{ F, 1u, 1u, M, GLM5_NEXT_SPLIT_NONE },
		{ F, 1u, 1u, SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT, GLM5_NEXT_SPLIT_NONE },
		{ F, 1u, SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL, X, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, X, H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, H, 2u * H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, 1u, H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, 1u, H, GLM5_NEXT_SPLIT_NONE },
		{ B, 1u, 1u, H, GLM5_NEXT_SPLIT_NONE },
	};
	if ( kind >= GLM5_NEXT_T_COUNT )
		return 0u;
	shape->payload_type = table[kind][0];
	shape->groups = table[kind][1];
	shape->rows = table[kind][2];
	shape->columns = table[kind][3];
	shape->split = table[kind][4];
	return 1u;
}

static inline int32_t Glm5NextPackEntryCheck(const Glm5NextPackEntry *entry, uint32_t expert_codec, uint32_t tp_degree, uint64_t file_bytes)
{
	Glm5NextPackShape shape;
	uint32_t codec, encoding;
	uint64_t payload, scale;
	if ( tp_degree == 0u || Glm5NextPackShapeOf(entry->tensor_kind, &shape) == 0u || Glm5NextPackKindAllowed(entry->tensor_kind, entry->layer_index) == 0u )
		return -1;
	if ( shape.split == GLM5_NEXT_SPLIT_ROWS )
	{
		if ( shape.rows % tp_degree != 0u )
			return -2;
		shape.rows /= tp_degree;
	}
	if ( shape.split == GLM5_NEXT_SPLIT_COLUMNS )
	{
		if ( shape.columns % tp_degree != 0u )
			return -2;
		shape.columns /= tp_degree;
	}
	codec = shape.payload_type == GLM5_NEXT_PAYLOAD_PACKED ? expert_codec : shape.payload_type == GLM5_NEXT_PAYLOAD_BF16 ? SPARK_WEIGHT_CODEC_BF16 : SPARK_WEIGHT_CODEC_NONE;
	encoding = shape.payload_type == GLM5_NEXT_PAYLOAD_PACKED ? SparkWeightCodecScaleEncoding(expert_codec) : SPARK_WEIGHT_SCALE_ENCODING_NONE;
	if ( entry->payload_type != shape.payload_type || entry->weight_codec != codec || entry->scale_encoding != encoding ||
		entry->group_count != shape.groups || entry->rows != shape.rows || entry->columns != shape.columns )
		return -3;
	payload = (uint64_t)shape.groups * shape.rows * shape.columns;
	payload = shape.payload_type == GLM5_NEXT_PAYLOAD_BF16 ? payload * 2u : shape.payload_type == GLM5_NEXT_PAYLOAD_PACKED ?
		SparkWeightCodecPayloadBytes(codec, (uint64_t)shape.groups * shape.rows, shape.columns) : payload * 4u;
	scale = shape.payload_type == GLM5_NEXT_PAYLOAD_PACKED ? SparkWeightCodecScaleBytes(codec, shape.groups, shape.rows, shape.columns) : 0u;
	if ( payload == 0u || entry->payload_bytes != payload || entry->scale_bytes != scale ||
		entry->payload_offset % GLM5_NEXT_PACK_ALIGNMENT != 0u || entry->payload_offset > file_bytes || entry->payload_bytes > file_bytes - entry->payload_offset )
		return -4;
	if ( scale == 0u ? entry->scale_offset != 0u :
		(entry->scale_offset % GLM5_NEXT_PACK_ALIGNMENT != 0u || entry->scale_offset > file_bytes || entry->scale_bytes > file_bytes - entry->scale_offset) )
		return -5;
	return 0;
}

#endif
