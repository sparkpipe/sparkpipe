#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define SPARK_PRECISION_FULL 0u
#define SPARK_PRECISION_FP8_ATTENTION 1u
#define SPARK_PRECISION_MODES 2u

static inline const char *SparkPrecisionName(uint32_t precision)
{
	return(precision == SPARK_PRECISION_FULL ? "full" : precision == SPARK_PRECISION_FP8_ATTENTION ? "fp8" : "unknown");
}

static inline int SparkPrecisionParse(const char *text, size_t length, uint32_t *precision)
{
	uint32_t mode;
	for (mode = 0u; mode < SPARK_PRECISION_MODES; mode++)
		if ( strlen(SparkPrecisionName(mode)) == length && memcmp(SparkPrecisionName(mode),text,length) == 0 )
		{
			*precision = mode;
			return(1);
		}
	return(0);
}
