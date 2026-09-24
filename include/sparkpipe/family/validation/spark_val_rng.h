#pragma once

static uint32_t SPARK_FAMILY(ValNext)(void)
{
	SPARK_FAMILY(ValRandomState) = SPARK_FAMILY(ValRandomState) * 1664525u + 1013904223u;
	return(SPARK_FAMILY(ValRandomState) >> 8u);
}
