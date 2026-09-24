#pragma once

static inline void SPARK_FAMILY(StagePackShapeF32)(SPARK_FAMILY(StagePackTensorShape) *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
    shape->payload_type = SPARK_FAMILY_CONST(STAGEPACK_PAYLOAD_F32);
    shape->weight_codec = SPARK_WEIGHT_CODEC_NONE;
    shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
    shape->group_count = groups;
    shape->rows = rows;
    shape->columns = columns;
}
