#pragma once

static inline uint32_t SPARK_FAMILY(StagePackKindIsGlobal)(uint32_t tensor_kind)
{
    return(tensor_kind <= SPARK_FAMILY_CONST(STAGEPACK_TENSOR_LM_HEAD) ? 1u : 0u);
}

static inline void SPARK_FAMILY(StagePackShapeBf16)(SPARK_FAMILY(StagePackTensorShape) *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
    shape->payload_type = SPARK_FAMILY_CONST(STAGEPACK_PAYLOAD_BF16);
    shape->weight_codec = SPARK_WEIGHT_CODEC_BF16;
    shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
    shape->group_count = groups;
    shape->rows = rows;
    shape->columns = columns;
}
