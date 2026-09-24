#pragma once

static inline uint32_t SPARK_FAMILY(StagePackKindIsHc)(uint32_t tensor_kind)
{
    return(tensor_kind >= SPARK_FAMILY_CONST(STAGEPACK_TENSOR_HC_ATTN_FN) &&
           tensor_kind <= SPARK_FAMILY_CONST(STAGEPACK_TENSOR_HC_FFN_SCALE) ? 1u : 0u);
}
