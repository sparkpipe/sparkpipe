#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_OCCUPANCY_BLOCKS_PER_SM 24u
#define SPARK_NVRM_OCCUPANCY_REGISTERS_PER_SM 65536u
#define SPARK_NVRM_OCCUPANCY_SHARED_PER_SM (100u * 1024u)
#define SPARK_NVRM_OCCUPANCY_REGISTER_UNIT 256u
#define SPARK_NVRM_OCCUPANCY_SHARED_UNIT 128u
#define SPARK_NVRM_WARP_THREADS 32u

typedef struct SparkNvrmOccupancyLimits
{
	uint32_t warps_per_sm;
	uint32_t registers_per_sm;
	uint32_t blocks_per_sm;
	uint32_t shared_per_sm;
	uint32_t register_unit;
	uint32_t shared_unit;
} SparkNvrmOccupancyLimits;

uint32_t SparkNvrmOccupancy(const SparkNvrmOccupancyLimits *limits, uint32_t threads, uint32_t registers_per_thread, uint32_t shared_bytes);
int SparkNvrmOccupancyLimitsOf(SparkNvrmOccupancyLimits *limits);

#ifdef __cplusplus
}
#endif
