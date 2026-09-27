#pragma once

static float SPARK_FAMILY(ValUniform)(float scale)
{
	return(((float)(int32_t)(SPARK_FAMILY(ValNext)() & 0xffffu) - 32768.0f) * scale / 32768.0f);
}

static uint16_t SPARK_FAMILY(ValBf16)(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	bits += 0x8000u;
	return((uint16_t)(bits >> 16u));
}

static void SPARK_FAMILY(ValFillBf16)(uint16_t *packed, float *exact, uint64_t count, float scale)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		packed[index] = SPARK_FAMILY(ValBf16)(SPARK_FAMILY(ValUniform)(scale));
		if (exact != 0)
			exact[index] = SPARK_FAMILY(ValFromBf16)(packed[index]);
	}
}
