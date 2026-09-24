#pragma once

static inline uint32_t SPARK_FAMILY(StagePackLayerIsDense)(uint32_t layer_index)
{
    return(layer_index < SPARK_FAMILY_CONST(MODEL_FIRST_ROUTED_LAYER) ? 1u : 0u);
}

static inline uint32_t SPARK_FAMILY(StagePackKindIsDense)(uint32_t tensor_kind)
{
    return(tensor_kind == SPARK_FAMILY_CONST(STAGEPACK_TENSOR_DENSE_GATE_UP) ||
           tensor_kind == SPARK_FAMILY_CONST(STAGEPACK_TENSOR_DENSE_DOWN) ? 1u : 0u);
}

static inline uint32_t SPARK_FAMILY(StagePackKindIsRouted)(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_FAMILY_CONST(STAGEPACK_TENSOR_ROUTER) &&
           tensor_kind <= SPARK_FAMILY_CONST(STAGEPACK_TENSOR_SHARED_DOWN) ? 1u : 0u);
}

static inline uint32_t SPARK_FAMILY(StagePackHeaderTpDegree)(const SPARK_FAMILY(StagePackHeader) *header)
{
    return(header != 0 ? header->reserved0 : 0u);
}

static inline uint32_t SPARK_FAMILY(StagePackHeaderTpRank)(const SPARK_FAMILY(StagePackHeader) *header)
{
    return(header != 0 ? header->reserved1 : 0u);
}

static inline uint64_t SPARK_FAMILY(StagePackExpectedPayloadBytes)(const SPARK_FAMILY(StagePackTensorShape) *shape)
{
    uint64_t elements;
    if ( shape == 0 || shape->group_count == 0u || shape->rows == 0u || shape->columns == 0u || shape->group_count > UINT64_MAX / shape->rows || (uint64_t)shape->group_count * shape->rows > UINT64_MAX / shape->columns )
        return(0u);
    elements = (uint64_t)shape->group_count * shape->rows * shape->columns;
    if ( shape->payload_type == SPARK_FAMILY_CONST(STAGEPACK_PAYLOAD_BF16) )
        return(elements > UINT64_MAX / 2u ? 0u : elements * 2u);
    if ( shape->payload_type == SPARK_FAMILY_CONST(STAGEPACK_PAYLOAD_F32) || shape->payload_type == SPARK_FAMILY_CONST(STAGEPACK_PAYLOAD_U32) )
        return(elements > UINT64_MAX / 4u ? 0u : elements * 4u);
    return(shape->payload_type == SPARK_FAMILY_CONST(STAGEPACK_PAYLOAD_PACKED_WEIGHT) ? SparkWeightCodecPayloadBytes(shape->weight_codec,(uint64_t)shape->group_count * shape->rows,shape->columns) : 0u);
}

static inline uint64_t SPARK_FAMILY(StagePackExpectedScaleBytes)(const SPARK_FAMILY(StagePackTensorShape) *shape)
{
    return(shape != 0 && shape->payload_type == SPARK_FAMILY_CONST(STAGEPACK_PAYLOAD_PACKED_WEIGHT) ? SparkWeightCodecScaleBytes(shape->weight_codec,shape->group_count,shape->rows,shape->columns) : 0u);
}
