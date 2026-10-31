#pragma once

#include <stdint.h>

#define SPARK_CLOCK_CAP 0u
#define SPARK_CLOCK_MAX 1u
#define SPARK_CLOCK_MODES 2u

static inline const char *SparkClockModeName(uint32_t mode)
{
	return(mode == SPARK_CLOCK_CAP ? "cap" : mode == SPARK_CLOCK_MAX ? "max" : "unknown");
}
