#pragma once

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_ling_model.h"
#include "sparkpipe/spark_weight_codec.h"

#define SPARK_LING_STAGEPACK_MAGIC UINT32_C(0x33474E4C)
#define SPARK_LING_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_LING_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_LING_STAGEPACK_ALIGNMENT_BYTES 256u
#define SPARK_LING_STAGEPACK_MODEL_REVISION_BYTES 65u
#define SPARK_LING_STAGEPACK_SHA256_BYTES 32u
#define SPARK_LING_STAGEPACK_FLAG_MTP UINT32_C(0x00000001)
#define SPARK_LING_STAGEPACK_KNOWN_FLAGS SPARK_LING_STAGEPACK_FLAG_MTP

typedef enum SparkLingStagePackPayloadType
{
    SPARK_LING_STAGEPACK_PAYLOAD_BF16 = 1,
    SPARK_LING_STAGEPACK_PAYLOAD_F32 = 2,
    SPARK_LING_STAGEPACK_PAYLOAD_U32 = 3,
    SPARK_LING_STAGEPACK_PAYLOAD_PACKED_WEIGHT = 4
} SparkLingStagePackPayloadType;

typedef enum SparkLingStagePackTensorKind
{
    SPARK_LING_STAGEPACK_TENSOR_EMBEDDING = 0,
    SPARK_LING_STAGEPACK_TENSOR_FINAL_NORM = 1,
    SPARK_LING_STAGEPACK_TENSOR_LM_HEAD = 2,
    SPARK_LING_STAGEPACK_TENSOR_ATTN_NORM = 3,
    SPARK_LING_STAGEPACK_TENSOR_Q = 4,
    SPARK_LING_STAGEPACK_TENSOR_KV_A = 5,
    SPARK_LING_STAGEPACK_TENSOR_KV_A_NORM = 6,
    SPARK_LING_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED = 7,
    SPARK_LING_STAGEPACK_TENSOR_KV_B_VALUE = 8,
    SPARK_LING_STAGEPACK_TENSOR_ATTN_GATE = 9,
    SPARK_LING_STAGEPACK_TENSOR_ATTN_OUTPUT = 10,
    SPARK_LING_STAGEPACK_TENSOR_POST_ATTN_NORM = 11,
    SPARK_LING_STAGEPACK_TENSOR_DENSE_GATE_UP = 12,
    SPARK_LING_STAGEPACK_TENSOR_DENSE_DOWN = 13,
    SPARK_LING_STAGEPACK_TENSOR_ROUTER = 14,
    SPARK_LING_STAGEPACK_TENSOR_ROUTER_CORRECTION = 15,
    SPARK_LING_STAGEPACK_TENSOR_EXPERT_UP_GATE = 16,
    SPARK_LING_STAGEPACK_TENSOR_EXPERT_DOWN = 17,
    SPARK_LING_STAGEPACK_TENSOR_SHARED_GATE_UP = 18,
    SPARK_LING_STAGEPACK_TENSOR_SHARED_DOWN = 19,
    SPARK_LING_STAGEPACK_TENSOR_KDA_QKV_BETA = 20,
    SPARK_LING_STAGEPACK_TENSOR_KDA_DECAY_PROJ = 21,
    SPARK_LING_STAGEPACK_TENSOR_KDA_GATE_PROJ = 22,
    SPARK_LING_STAGEPACK_TENSOR_KDA_Q_CONV = 23,
    SPARK_LING_STAGEPACK_TENSOR_KDA_K_CONV = 24,
    SPARK_LING_STAGEPACK_TENSOR_KDA_V_CONV = 25,
    SPARK_LING_STAGEPACK_TENSOR_KDA_DECAY_BIAS = 26,
    SPARK_LING_STAGEPACK_TENSOR_KDA_HEAD_LOG_SCALE = 27,
    SPARK_LING_STAGEPACK_TENSOR_KDA_OUT_NORM = 28,
    SPARK_LING_STAGEPACK_TENSOR_KDA_OUT = 29,
    SPARK_LING_STAGEPACK_TENSOR_MTP_EH_PROJ = 30,
    SPARK_LING_STAGEPACK_TENSOR_MTP_ENORM = 31,
    SPARK_LING_STAGEPACK_TENSOR_MTP_HNORM = 32,
    SPARK_LING_STAGEPACK_TENSOR_MTP_SHARED_NORM = 33,
    SPARK_LING_STAGEPACK_TENSOR_KIND_COUNT = 34
} SparkLingStagePackTensorKind;

typedef struct SparkLingStagePackHeader
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
    char model_revision[SPARK_LING_STAGEPACK_MODEL_REVISION_BYTES];
    uint8_t contract_sha256[SPARK_LING_STAGEPACK_SHA256_BYTES];
    uint8_t source_config_sha256[SPARK_LING_STAGEPACK_SHA256_BYTES];
    uint8_t pack_recipe_sha256[SPARK_LING_STAGEPACK_SHA256_BYTES];
} SparkLingStagePackHeader;

typedef struct SparkLingStagePackEntry
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
} SparkLingStagePackEntry;

typedef struct SparkLingStagePackTensorShape
{
    uint32_t payload_type;
    uint32_t weight_codec;
    uint32_t scale_encoding;
    uint32_t group_count;
    uint32_t rows;
    uint32_t columns;
} SparkLingStagePackTensorShape;

#define SPARK_LING_STAGEPACK_HEADER_BYTES ((uint32_t)sizeof(SparkLingStagePackHeader))
#define SPARK_LING_STAGEPACK_ENTRY_BYTES ((uint32_t)sizeof(SparkLingStagePackEntry))

static inline uint32_t SparkLingStagePackLayerIsDense(uint32_t layer_index)
{
    return(layer_index < SPARK_LING_MODEL_FIRST_ROUTED_LAYER ? 1u : 0u);
}

static inline uint32_t SparkLingStagePackLayerIsKda(uint32_t layer_index)
{
    return(layer_index < SPARK_LING_MODEL_LAYER_COUNT &&
        SPARK_LING_MODEL_LAYER_IS_KDA(layer_index) ? 1u : 0u);
}

static inline uint32_t SparkLingStagePackLayerIsMla(uint32_t layer_index)
{
    return(layer_index < SPARK_LING_MODEL_LAYER_COUNT &&
        SPARK_LING_MODEL_LAYER_IS_MLA(layer_index) ? 1u : 0u);
}

static inline uint32_t SparkLingStagePackLayerIsMtp(uint32_t layer_index)
{
    return(layer_index == SPARK_LING_MODEL_MTP_LAYER_INDEX ? 1u : 0u);
}

static inline uint32_t SparkLingStagePackKindIsGlobal(uint32_t tensor_kind)
{
    return(tensor_kind <= SPARK_LING_STAGEPACK_TENSOR_LM_HEAD ? 1u : 0u);
}

static inline uint32_t SparkLingStagePackKindIsMla(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_LING_STAGEPACK_TENSOR_Q &&
           tensor_kind <= SPARK_LING_STAGEPACK_TENSOR_ATTN_OUTPUT ? 1u : 0u);
}

static inline uint32_t SparkLingStagePackKindIsKda(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_LING_STAGEPACK_TENSOR_KDA_QKV_BETA &&
           tensor_kind <= SPARK_LING_STAGEPACK_TENSOR_KDA_OUT ? 1u : 0u);
}

static inline uint32_t SparkLingStagePackKindIsMtp(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_LING_STAGEPACK_TENSOR_MTP_EH_PROJ &&
           tensor_kind <= SPARK_LING_STAGEPACK_TENSOR_MTP_SHARED_NORM ? 1u : 0u);
}

static inline uint32_t SparkLingStagePackKindIsDense(uint32_t tensor_kind)
{
    return(tensor_kind == SPARK_LING_STAGEPACK_TENSOR_DENSE_GATE_UP ||
           tensor_kind == SPARK_LING_STAGEPACK_TENSOR_DENSE_DOWN ? 1u : 0u);
}

static inline uint32_t SparkLingStagePackKindIsRouted(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_LING_STAGEPACK_TENSOR_ROUTER &&
           tensor_kind <= SPARK_LING_STAGEPACK_TENSOR_SHARED_DOWN ? 1u : 0u);
}

static inline void SparkLingStagePackShapeBf16(SparkLingStagePackTensorShape *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
    shape->payload_type = SPARK_LING_STAGEPACK_PAYLOAD_BF16;
    shape->weight_codec = SPARK_WEIGHT_CODEC_BF16;
    shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
    shape->group_count = groups;
    shape->rows = rows;
    shape->columns = columns;
}

static inline uint32_t SparkLingStagePackTpShardsRows(uint32_t tensor_kind)
{
    switch ( tensor_kind )
    {
    case SPARK_LING_STAGEPACK_TENSOR_EMBEDDING:
    case SPARK_LING_STAGEPACK_TENSOR_LM_HEAD:
    case SPARK_LING_STAGEPACK_TENSOR_Q:
    case SPARK_LING_STAGEPACK_TENSOR_ATTN_GATE:
    case SPARK_LING_STAGEPACK_TENSOR_DENSE_GATE_UP:
    case SPARK_LING_STAGEPACK_TENSOR_EXPERT_UP_GATE:
    case SPARK_LING_STAGEPACK_TENSOR_SHARED_GATE_UP:
    case SPARK_LING_STAGEPACK_TENSOR_KDA_QKV_BETA:
    case SPARK_LING_STAGEPACK_TENSOR_KDA_DECAY_PROJ:
    case SPARK_LING_STAGEPACK_TENSOR_KDA_GATE_PROJ:
    case SPARK_LING_STAGEPACK_TENSOR_KDA_Q_CONV:
    case SPARK_LING_STAGEPACK_TENSOR_KDA_K_CONV:
    case SPARK_LING_STAGEPACK_TENSOR_KDA_V_CONV:
        return(1u);
    default:
        return(0u);
    }
}

static inline uint32_t SparkLingStagePackTpShardsCols(uint32_t tensor_kind)
{
    switch ( tensor_kind )
    {
    case SPARK_LING_STAGEPACK_TENSOR_ATTN_OUTPUT:
    case SPARK_LING_STAGEPACK_TENSOR_KDA_OUT:
    case SPARK_LING_STAGEPACK_TENSOR_DENSE_DOWN:
    case SPARK_LING_STAGEPACK_TENSOR_EXPERT_DOWN:
    case SPARK_LING_STAGEPACK_TENSOR_SHARED_DOWN:
    case SPARK_LING_STAGEPACK_TENSOR_KDA_DECAY_BIAS:
    case SPARK_LING_STAGEPACK_TENSOR_KDA_HEAD_LOG_SCALE:
        return(1u);
    default:
        return(0u);
    }
}

static inline uint32_t SparkLingStagePackHeaderTpDegree(const SparkLingStagePackHeader *header)
{
    return(header != 0 ? header->reserved0 : 0u);
}

static inline uint32_t SparkLingStagePackHeaderTpRank(const SparkLingStagePackHeader *header)
{
    return(header != 0 ? header->reserved1 : 0u);
}

typedef struct SparkLingStagePackShapeSpec
{
    uint32_t payload_type;
    uint32_t weight_codec;
    uint32_t scale_encoding;
    uint32_t group_count;
    uint32_t rows;
    uint32_t columns;
    uint32_t codec_from_arg;
} SparkLingStagePackShapeSpec;

static const SparkLingStagePackShapeSpec SPARK_LING_STAGEPACK_SHAPE_TABLE[SPARK_LING_STAGEPACK_TENSOR_KIND_COUNT] = {
    [SPARK_LING_STAGEPACK_TENSOR_EMBEDDING] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_OUTPUT_VOCAB_COUNT, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_FINAL_NORM] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_LM_HEAD] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_OUTPUT_VOCAB_COUNT, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_ATTN_NORM] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_Q] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_MLA_QUERY_DIMENSION, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KV_A] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_MLA_KV_A_DIMENSION, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KV_A_NORM] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LING_MODEL_MLA_LATENT_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, SPARK_LING_MODEL_MLA_HEAD_COUNT, SPARK_LING_MODEL_MLA_LATENT_DIMENSION, SPARK_LING_MODEL_MLA_QK_NOPE_HEAD_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KV_B_VALUE] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, SPARK_LING_MODEL_MLA_HEAD_COUNT, SPARK_LING_MODEL_MLA_VALUE_HEAD_DIMENSION, SPARK_LING_MODEL_MLA_LATENT_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_ATTN_GATE] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_MLA_HEAD_COUNT, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_ATTN_OUTPUT] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_HIDDEN_DIMENSION, SPARK_LING_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_POST_ATTN_NORM] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_DENSE_GATE_UP] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 2u * SPARK_LING_MODEL_DENSE_INTERMEDIATE_DIMENSION, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_DENSE_DOWN] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_HIDDEN_DIMENSION, SPARK_LING_MODEL_DENSE_INTERMEDIATE_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_ROUTER] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_MOE_EXPERT_COUNT, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_ROUTER_CORRECTION] =
        {SPARK_LING_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LING_MODEL_MOE_EXPERT_COUNT, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_EXPERT_UP_GATE] =
        {SPARK_LING_STAGEPACK_PAYLOAD_PACKED_WEIGHT, 0u, 0u, SPARK_LING_MODEL_MOE_EXPERT_COUNT, 2u * SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION, SPARK_LING_MODEL_HIDDEN_DIMENSION, 1u},
    [SPARK_LING_STAGEPACK_TENSOR_EXPERT_DOWN] =
        {SPARK_LING_STAGEPACK_PAYLOAD_PACKED_WEIGHT, 0u, 0u, SPARK_LING_MODEL_MOE_EXPERT_COUNT, SPARK_LING_MODEL_HIDDEN_DIMENSION, SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION, 1u},
    [SPARK_LING_STAGEPACK_TENSOR_SHARED_GATE_UP] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 2u * SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_SHARED_DOWN] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_HIDDEN_DIMENSION, SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KDA_QKV_BETA] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 2u * SPARK_LING_MODEL_KDA_QKV_DIMENSION + SPARK_LING_MODEL_KDA_VALUE_DIMENSION + SPARK_LING_MODEL_KDA_HEAD_COUNT, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KDA_DECAY_PROJ] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_KDA_VALUE_DIMENSION, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KDA_GATE_PROJ] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_KDA_VALUE_DIMENSION, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KDA_Q_CONV] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_KDA_QKV_DIMENSION, SPARK_LING_MODEL_KDA_SHORT_CONV_KERNEL, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KDA_K_CONV] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_KDA_QKV_DIMENSION, SPARK_LING_MODEL_KDA_SHORT_CONV_KERNEL, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KDA_V_CONV] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_KDA_VALUE_DIMENSION, SPARK_LING_MODEL_KDA_SHORT_CONV_KERNEL, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KDA_DECAY_BIAS] =
        {SPARK_LING_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LING_MODEL_KDA_VALUE_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KDA_HEAD_LOG_SCALE] =
        {SPARK_LING_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LING_MODEL_KDA_HEAD_COUNT, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KDA_OUT_NORM] =
        {SPARK_LING_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LING_MODEL_KDA_HEAD_KEY_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_KDA_OUT] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_HIDDEN_DIMENSION, SPARK_LING_MODEL_KDA_VALUE_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_MTP_EH_PROJ] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LING_MODEL_HIDDEN_DIMENSION, 2u * SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_MTP_ENORM] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_MTP_HNORM] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LING_STAGEPACK_TENSOR_MTP_SHARED_NORM] =
        {SPARK_LING_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LING_MODEL_HIDDEN_DIMENSION, 0u},
};

static inline int32_t SparkLingStagePackCheckLayerKind(uint32_t layer_index,uint32_t tensor_kind)
{
    uint32_t mtp_layer = SPARK_LING_MODEL_MTP_LAYER_INDEX;
    uint32_t in_mtp = layer_index == mtp_layer;
    if ( SparkLingStagePackKindIsMla(tensor_kind) != 0u &&
         SparkLingStagePackLayerIsMla(layer_index) == 0u && !in_mtp )
        return(-3);
    if ( SparkLingStagePackKindIsKda(tensor_kind) != 0u &&
         SparkLingStagePackLayerIsKda(layer_index) == 0u )
        return(-3);
    if ( SparkLingStagePackKindIsMtp(tensor_kind) != 0u && !in_mtp )
        return(-3);
    if ( SparkLingStagePackKindIsDense(tensor_kind) !=
             SparkLingStagePackLayerIsDense(layer_index) &&
         (SparkLingStagePackKindIsDense(tensor_kind) != 0u ||
          SparkLingStagePackKindIsRouted(tensor_kind) != 0u) )
        return(-4);
    return(0);
}

static inline int32_t SparkLingStagePackExpectedShape(uint32_t tensor_kind,uint32_t layer_index,uint32_t expert_codec,uint32_t tp_degree,SparkLingStagePackTensorShape *shape)
{
    const SparkLingStagePackShapeSpec *spec;
    uint32_t global;
    uint32_t mtp_layer = SPARK_LING_MODEL_MTP_LAYER_INDEX;
    if ( shape == 0 || tensor_kind >= SPARK_LING_STAGEPACK_TENSOR_KIND_COUNT || tp_degree == 0u )
        return(-1);
    memset(shape,0,sizeof(*shape));
    global = SparkLingStagePackKindIsGlobal(tensor_kind);
    if ( (global != 0u && layer_index != SPARK_LING_STAGEPACK_GLOBAL_LAYER) ||
         (global == 0u && layer_index > mtp_layer) )
        return(-2);
    if ( layer_index <= mtp_layer && layer_index != SPARK_LING_STAGEPACK_GLOBAL_LAYER )
    {
        int32_t allowed = SparkLingStagePackCheckLayerKind(layer_index,tensor_kind);
        if ( allowed != 0 )
            return(allowed);
    }
    spec = &SPARK_LING_STAGEPACK_SHAPE_TABLE[tensor_kind];
    if ( spec->payload_type == 0u )
        return(-6);
    if ( spec->codec_from_arg != 0u )
    {
        if ( expert_codec < SPARK_WEIGHT_CODEC_INT6 || expert_codec > SPARK_WEIGHT_CODEC_MXFP4_E2M1 )
            return(-5);
        shape->payload_type = spec->payload_type;
        shape->weight_codec = expert_codec;
        shape->scale_encoding = SparkWeightCodecScaleEncoding(expert_codec);
    }
    else
    {
        shape->payload_type = spec->payload_type;
        shape->weight_codec = spec->weight_codec;
        shape->scale_encoding = spec->scale_encoding;
    }
    shape->group_count = spec->group_count;
    shape->rows = spec->rows;
    shape->columns = spec->columns;
    if ( SparkLingStagePackTpShardsRows(tensor_kind) != 0u )
    {
        if ( shape->rows == 0u || shape->rows % tp_degree != 0u )
            return(-7);
        shape->rows /= tp_degree;
    }
    if ( SparkLingStagePackTpShardsCols(tensor_kind) != 0u )
    {
        if ( shape->columns == 0u || shape->columns % tp_degree != 0u )
            return(-7);
        shape->columns /= tp_degree;
    }
    return(0);
}

static inline uint64_t SparkLingStagePackExpectedPayloadBytes(const SparkLingStagePackTensorShape *shape)
{
    uint64_t elements;
    if ( shape == 0 || shape->group_count == 0u || shape->rows == 0u || shape->columns == 0u || shape->group_count > UINT64_MAX / shape->rows || (uint64_t)shape->group_count * shape->rows > UINT64_MAX / shape->columns )
        return(0u);
    elements = (uint64_t)shape->group_count * shape->rows * shape->columns;
    if ( shape->payload_type == SPARK_LING_STAGEPACK_PAYLOAD_BF16 )
        return(elements > UINT64_MAX / 2u ? 0u : elements * 2u);
    if ( shape->payload_type == SPARK_LING_STAGEPACK_PAYLOAD_F32 || shape->payload_type == SPARK_LING_STAGEPACK_PAYLOAD_U32 )
        return(elements > UINT64_MAX / 4u ? 0u : elements * 4u);
    return(shape->payload_type == SPARK_LING_STAGEPACK_PAYLOAD_PACKED_WEIGHT ? SparkWeightCodecPayloadBytes(shape->weight_codec,(uint64_t)shape->group_count * shape->rows,shape->columns) : 0u);
}

static inline uint64_t SparkLingStagePackExpectedScaleBytes(const SparkLingStagePackTensorShape *shape)
{
    return(shape != 0 && shape->payload_type == SPARK_LING_STAGEPACK_PAYLOAD_PACKED_WEIGHT ? SparkWeightCodecScaleBytes(shape->weight_codec,shape->group_count,shape->rows,shape->columns) : 0u);
}
