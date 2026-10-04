#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t (*SparkWeightdPacerClock)(void *context);
typedef void (*SparkWeightdPacerSleep)(void *context,uint64_t nanoseconds);

typedef struct SparkWeightdPacer
{
	uint64_t bytes_per_second;
	uint64_t next_ns;
	uint64_t paced_bytes;
	uint64_t wait_ns;
	uint64_t refused_count;
	SparkWeightdPacerClock clock;
	SparkWeightdPacerSleep sleep;
	void *context;
} SparkWeightdPacer;

SparkStatus SparkWeightdPacerInitialize(SparkWeightdPacer *pacer,uint64_t bytes_per_second,SparkWeightdPacerClock clock,SparkWeightdPacerSleep sleep,void *context);
SparkStatus SparkWeightdPacerTake(SparkWeightdPacer *pacer,uint64_t bytes,uint32_t serving);

#ifdef __cplusplus
}
#endif
