#pragma once

static inline uint32_t SPARK_FAMILY(StagePackLayerIsKda)(uint32_t layer_index)
{
    return(layer_index < SPARK_FAMILY_CONST(MODEL_LAYER_COUNT) &&
        SPARK_FAMILY_CONST(MODEL_LAYER_IS_KDA)(layer_index) ? 1u : 0u);
}

static inline uint32_t SPARK_FAMILY(StagePackLayerIsMtp)(uint32_t layer_index)
{
    return(layer_index == SPARK_FAMILY_CONST(MODEL_MTP_LAYER_INDEX) ? 1u : 0u);
}

static inline uint32_t SPARK_FAMILY(StagePackKindIsKda)(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_FAMILY_CONST(STAGEPACK_TENSOR_KDA_QKV_BETA) &&
           tensor_kind <= SPARK_FAMILY_CONST(STAGEPACK_TENSOR_KDA_OUT) ? 1u : 0u);
}

static inline uint32_t SPARK_FAMILY(StagePackKindIsMtp)(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_FAMILY_CONST(STAGEPACK_TENSOR_MTP_EH_PROJ) &&
           tensor_kind <= SPARK_FAMILY_CONST(STAGEPACK_TENSOR_MTP_SHARED_NORM) ? 1u : 0u);
}

static inline uint32_t SPARK_FAMILY(StagePackTpShardsCols)(uint32_t tensor_kind)
{
    switch ( tensor_kind )
    {
    case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_ATTN_OUTPUT):
    case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_KDA_OUT):
    case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_DENSE_DOWN):
    case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_EXPERT_DOWN):
    case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_SHARED_DOWN):
    case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_KDA_DECAY_BIAS):
    case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_KDA_HEAD_LOG_SCALE):
        return(1u);
    default:
        return(0u);
    }
}
