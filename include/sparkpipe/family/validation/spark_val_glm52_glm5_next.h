#pragma once

static uint32_t SPARK_FAMILY(ValCodecUsesSignedIntGrid)(uint32_t codec)
{
	return(codec == SPARK_FAMILY_CONST(VAL_CODEC_INT6) || codec == SPARK_FAMILY_CONST(VAL_CODEC_INT7) ||
		codec == SPARK_FAMILY_CONST(VAL_CODEC_INT8) ? 1u : 0u);
}

static uint64_t SPARK_FAMILY(ValPayloadBytesPerExpert)(uint32_t codec,uint32_t rows,uint32_t columns)
{
	return((uint64_t)rows * SPARK_FAMILY(ValPayloadRowBytes)(codec,columns));
}

static uint64_t SPARK_FAMILY(ValPayloadExpertOffset)(uint32_t codec,uint32_t expert,uint32_t rows,uint32_t columns)
{
	return((uint64_t)expert * SPARK_FAMILY(ValPayloadBytesPerExpert)(codec,rows,columns));
}
