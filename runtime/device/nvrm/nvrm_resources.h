#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkNvrmResourceCounts
{
	uint64_t rm_objects;
	uint32_t shared_memory;
	uint32_t reservations;
	uint32_t mappings;
} SparkNvrmResourceCounts;

SparkStatus SparkNvrmResourceCountsRead(SparkNvrmResourceCounts *counts);

#ifdef __cplusplus
}
#endif
