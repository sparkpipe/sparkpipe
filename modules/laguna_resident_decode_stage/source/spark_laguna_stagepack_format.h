#pragma once

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_laguna_model.h"
#include "sparkpipe/spark_weight_codec.h"

#define SPARK_LAGUNA_STAGEPACK_MAGIC UINT32_C(0x334C4147)
#define SPARK_LAGUNA_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_LAGUNA_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_LAGUNA_STAGEPACK_ALIGNMENT_BYTES 256u
#define SPARK_LAGUNA_STAGEPACK_MODEL_REVISION_BYTES 65u
#define SPARK_LAGUNA_STAGEPACK_SHA256_BYTES 32u
#define SPARK_LAGUNA_STAGEPACK_FLAG_DFLASH UINT32_C(0x00000002)
#define SPARK_LAGUNA_STAGEPACK_KNOWN_FLAGS SPARK_LAGUNA_STAGEPACK_FLAG_DFLASH

typedef enum SparkLagunaStagePackPayloadType
{
    SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16 = 1,
    SPARK_LAGUNA_STAGEPACK_PAYLOAD_F32 = 2,
    SPARK_LAGUNA_STAGEPACK_PAYLOAD_U32 = 3,
    SPARK_LAGUNA_STAGEPACK_PAYLOAD_PACKED_WEIGHT = 4
} SparkLagunaStagePackPayloadType;

typedef enum SparkLagunaStagePackTensorKind
{
    SPARK_LAGUNA_STAGEPACK_TENSOR_EMBEDDING = 0,
    SPARK_LAGUNA_STAGEPACK_TENSOR_FINAL_NORM = 1,
    SPARK_LAGUNA_STAGEPACK_TENSOR_LM_HEAD = 2,
    SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_INPUT_NORM = 3,
    SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_POST_NORM = 4,
    SPARK_LAGUNA_STAGEPACK_TENSOR_FUSED_QKV = 5,
    SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_OUTPUT = 6,
    SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_GATE = 7,
    SPARK_LAGUNA_STAGEPACK_TENSOR_Q_NORM = 8,
    SPARK_LAGUNA_STAGEPACK_TENSOR_K_NORM = 9,
    SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_GATE_UP = 10,
    SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_DOWN = 11,
    SPARK_LAGUNA_STAGEPACK_TENSOR_ROUTER = 12,
    SPARK_LAGUNA_STAGEPACK_TENSOR_ROUTER_CORRECTION = 13,
    SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_GATE_UP = 14,
    SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_DOWN = 15,
    SPARK_LAGUNA_STAGEPACK_TENSOR_SHARED_GATE_UP = 16,
    SPARK_LAGUNA_STAGEPACK_TENSOR_SHARED_DOWN = 17,
    SPARK_LAGUNA_STAGEPACK_TENSOR_KIND_COUNT = 18
} SparkLagunaStagePackTensorKind;

typedef struct SparkLagunaStagePackHeader
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
    char model_revision[SPARK_LAGUNA_STAGEPACK_MODEL_REVISION_BYTES];
    uint8_t contract_sha256[SPARK_LAGUNA_STAGEPACK_SHA256_BYTES];
    uint8_t source_config_sha256[SPARK_LAGUNA_STAGEPACK_SHA256_BYTES];
    uint8_t pack_recipe_sha256[SPARK_LAGUNA_STAGEPACK_SHA256_BYTES];
} SparkLagunaStagePackHeader;

typedef struct SparkLagunaStagePackEntry
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
} SparkLagunaStagePackEntry;

typedef struct SparkLagunaStagePackTensorShape
{
    uint32_t payload_type;
    uint32_t weight_codec;
    uint32_t scale_encoding;
    uint32_t group_count;
    uint32_t rows;
    uint32_t columns;
} SparkLagunaStagePackTensorShape;

#define SPARK_LAGUNA_STAGEPACK_HEADER_BYTES ((uint32_t)sizeof(SparkLagunaStagePackHeader))
#define SPARK_LAGUNA_STAGEPACK_ENTRY_BYTES ((uint32_t)sizeof(SparkLagunaStagePackEntry))

static inline uint32_t SparkLagunaStagePackKindIsGlobal(uint32_t tensor_kind)
{
    return(tensor_kind <= SPARK_LAGUNA_STAGEPACK_TENSOR_LM_HEAD ? 1u : 0u);
}

static inline uint32_t SparkLagunaStagePackKindIsAttention(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_LAGUNA_STAGEPACK_TENSOR_FUSED_QKV &&
           tensor_kind <= SPARK_LAGUNA_STAGEPACK_TENSOR_K_NORM ? 1u : 0u);
}

static inline uint32_t SparkLagunaStagePackKindIsDense(uint32_t tensor_kind)
{
    return(tensor_kind == SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_GATE_UP ||
           tensor_kind == SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_DOWN ? 1u : 0u);
}

static inline uint32_t SparkLagunaStagePackKindIsRouted(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_LAGUNA_STAGEPACK_TENSOR_ROUTER &&
           tensor_kind <= SPARK_LAGUNA_STAGEPACK_TENSOR_SHARED_DOWN ? 1u : 0u);
}

static inline uint32_t SparkLagunaStagePackLayerIsDense(uint32_t layer_index)
{
    return(layer_index < SPARK_LAGUNA_MODEL_FIRST_ROUTED_LAYER ? 1u : 0u);
}

static inline void SparkLagunaStagePackShapeBf16(SparkLagunaStagePackTensorShape *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
    shape->payload_type = SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16;
    shape->weight_codec = SPARK_WEIGHT_CODEC_BF16;
    shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
    shape->group_count = groups;
    shape->rows = rows;
    shape->columns = columns;
}

static inline void SparkLagunaStagePackShapeF32(SparkLagunaStagePackTensorShape *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
    shape->payload_type = SPARK_LAGUNA_STAGEPACK_PAYLOAD_F32;
    shape->weight_codec = SPARK_WEIGHT_CODEC_NONE;
    shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
    shape->group_count = groups;
    shape->rows = rows;
    shape->columns = columns;
}

static inline uint32_t SparkLagunaStagePackTpShardsRows(uint32_t tensor_kind)
{
    switch ( tensor_kind )
    {
    case SPARK_LAGUNA_STAGEPACK_TENSOR_EMBEDDING:
    case SPARK_LAGUNA_STAGEPACK_TENSOR_LM_HEAD:
    case SPARK_LAGUNA_STAGEPACK_TENSOR_FUSED_QKV:
    case SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_GATE:
    case SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_GATE_UP:
    case SPARK_LAGUNA_STAGEPACK_TENSOR_SHARED_GATE_UP:
        return(1u);
    default:
        return(0u);
    }
}

static inline uint32_t SparkLagunaStagePackTpShardsCols(uint32_t tensor_kind)
{
    switch ( tensor_kind )
    {
    case SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_OUTPUT:
    case SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_DOWN:
    case SPARK_LAGUNA_STAGEPACK_TENSOR_SHARED_DOWN:
        return(1u);
    default:
        return(0u);
    }
}

static inline uint32_t SparkLagunaStagePackHeaderTpDegree(const SparkLagunaStagePackHeader *header)
{
    return(header != 0 ? header->reserved0 : 0u);
}

static inline uint32_t SparkLagunaStagePackHeaderTpRank(const SparkLagunaStagePackHeader *header)
{
    return(header != 0 ? header->reserved1 : 0u);
}

typedef struct SparkLagunaStagePackShapeSpec
{
    uint32_t payload_type;
    uint32_t weight_codec;
    uint32_t scale_encoding;
    uint32_t group_count;
    uint32_t rows;
    uint32_t columns;
    uint32_t codec_from_arg;
} SparkLagunaStagePackShapeSpec;

static const SparkLagunaStagePackShapeSpec SPARK_LAGUNA_STAGEPACK_SHAPE_TABLE[SPARK_LAGUNA_STAGEPACK_TENSOR_KIND_COUNT] = {
    [SPARK_LAGUNA_STAGEPACK_TENSOR_EMBEDDING] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LAGUNA_MODEL_OUTPUT_VOCAB_COUNT, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_FINAL_NORM] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_LM_HEAD] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LAGUNA_MODEL_OUTPUT_VOCAB_COUNT, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_INPUT_NORM] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_POST_NORM] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_FUSED_QKV] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 0u, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_OUTPUT] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 0u, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_GATE] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 0u, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_Q_NORM] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_K_NORM] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_GATE_UP] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 2u * SPARK_LAGUNA_MODEL_DENSE_INTERMEDIATE_DIMENSION, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_DOWN] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, SPARK_LAGUNA_MODEL_DENSE_INTERMEDIATE_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_ROUTER] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_ROUTER_CORRECTION] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_F32, SPARK_WEIGHT_CODEC_NONE, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 1u, SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_GATE_UP] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_PACKED_WEIGHT, 0u, 0u, SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT, 2u * SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 1u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_DOWN] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_PACKED_WEIGHT, 0u, 0u, SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION, 1u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_SHARED_GATE_UP] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, 2u * SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, 0u},
    [SPARK_LAGUNA_STAGEPACK_TENSOR_SHARED_DOWN] =
        {SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16, SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_SCALE_ENCODING_NONE, 1u, SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION, SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION, 0u},
};

static inline uint32_t SparkLagunaStagePackKindIsExpertPartition(uint32_t tensor_kind)
{
    return(tensor_kind == SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_GATE_UP ||
           tensor_kind == SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_DOWN ? 1u : 0u);
}

static inline int32_t SparkLagunaStagePackCheckLayerKind(uint32_t layer_index,uint32_t tensor_kind)
{
    if ( SparkLagunaStagePackKindIsDense(tensor_kind) !=
             SparkLagunaStagePackLayerIsDense(layer_index) &&
         (SparkLagunaStagePackKindIsDense(tensor_kind) != 0u ||
          SparkLagunaStagePackKindIsRouted(tensor_kind) != 0u) )
        return(-4);
    return(0);
}

static inline int32_t SparkLagunaStagePackExpectedShape(uint32_t tensor_kind,uint32_t layer_index,uint32_t expert_codec,uint32_t tp_degree,SparkLagunaStagePackTensorShape *shape)
{
    const SparkLagunaStagePackShapeSpec *spec;
    uint32_t global;
    if ( shape == 0 || tensor_kind >= SPARK_LAGUNA_STAGEPACK_TENSOR_KIND_COUNT || tp_degree == 0u )
        return(-1);
    memset(shape,0,sizeof(*shape));
    global = SparkLagunaStagePackKindIsGlobal(tensor_kind);
    if ( (global != 0u && layer_index != SPARK_LAGUNA_STAGEPACK_GLOBAL_LAYER) ||
         (global == 0u && layer_index >= SPARK_LAGUNA_MODEL_LAYER_COUNT) )
        return(-2);
    if ( global == 0u )
    {
        int32_t allowed = SparkLagunaStagePackCheckLayerKind(layer_index,tensor_kind);
        if ( allowed != 0 )
            return(allowed);
    }
    spec = &SPARK_LAGUNA_STAGEPACK_SHAPE_TABLE[tensor_kind];
    if ( spec->payload_type == 0u )
        return(-6);
    if ( SparkLagunaStagePackKindIsExpertPartition(tensor_kind) != 0u &&
         expert_codec == SPARK_WEIGHT_CODEC_BF16 )
    {
        shape->payload_type = SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16;
        shape->weight_codec = SPARK_WEIGHT_CODEC_BF16;
        shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
    }
    else if ( spec->codec_from_arg != 0u )
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
    if ( tensor_kind == SPARK_LAGUNA_STAGEPACK_TENSOR_FUSED_QKV ||
         tensor_kind == SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_OUTPUT ||
         tensor_kind == SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_GATE )
    {
        uint32_t heads = SPARK_LAGUNA_MODEL_LAYER_HEAD_COUNT(layer_index);
        if ( tensor_kind == SPARK_LAGUNA_STAGEPACK_TENSOR_FUSED_QKV )
            shape->rows = heads * SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION +
                2u * SPARK_LAGUNA_MODEL_ATTENTION_KV_HEAD_COUNT * SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION;
        else if ( tensor_kind == SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_OUTPUT )
            shape->columns = heads * SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION;
        else
            shape->rows = heads;
    }
    if ( SparkLagunaStagePackKindIsExpertPartition(tensor_kind) != 0u )
    {
        if ( shape->group_count % tp_degree != 0u )
            return(-7);
        shape->group_count /= tp_degree;
    }
    if ( SparkLagunaStagePackTpShardsRows(tensor_kind) != 0u )
    {
        if ( shape->rows == 0u || shape->rows % tp_degree != 0u )
            return(-7);
        shape->rows /= tp_degree;
    }
    if ( SparkLagunaStagePackTpShardsCols(tensor_kind) != 0u )
    {
        if ( shape->columns == 0u || shape->columns % tp_degree != 0u )
            return(-7);
        shape->columns /= tp_degree;
    }
    return(0);
}

static inline uint64_t SparkLagunaStagePackExpectedPayloadBytes(const SparkLagunaStagePackTensorShape *shape)
{
    uint64_t elements;
    if ( shape == 0 || shape->group_count == 0u || shape->rows == 0u || shape->columns == 0u || shape->group_count > UINT64_MAX / shape->rows || (uint64_t)shape->group_count * shape->rows > UINT64_MAX / shape->columns )
        return(0u);
    elements = (uint64_t)shape->group_count * shape->rows * shape->columns;
    if ( shape->payload_type == SPARK_LAGUNA_STAGEPACK_PAYLOAD_BF16 )
        return(elements > UINT64_MAX / 2u ? 0u : elements * 2u);
    if ( shape->payload_type == SPARK_LAGUNA_STAGEPACK_PAYLOAD_F32 || shape->payload_type == SPARK_LAGUNA_STAGEPACK_PAYLOAD_U32 )
        return(elements > UINT64_MAX / 4u ? 0u : elements * 4u);
    return(shape->payload_type == SPARK_LAGUNA_STAGEPACK_PAYLOAD_PACKED_WEIGHT ? SparkWeightCodecPayloadBytes(shape->weight_codec,(uint64_t)shape->group_count * shape->rows,shape->columns) : 0u);
}

static inline uint64_t SparkLagunaStagePackExpectedScaleBytes(const SparkLagunaStagePackTensorShape *shape)
{
    return(shape != 0 && shape->payload_type == SPARK_LAGUNA_STAGEPACK_PAYLOAD_PACKED_WEIGHT ? SparkWeightCodecScaleBytes(shape->weight_codec,shape->group_count,shape->rows,shape->columns) : 0u);
}
