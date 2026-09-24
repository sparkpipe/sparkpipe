#pragma once

static float SPARK_FAMILY(ValFromBf16)(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float converted;
	memcpy(&converted,&bits,sizeof(converted));
	return(converted);
}
